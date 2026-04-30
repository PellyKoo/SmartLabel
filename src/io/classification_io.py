import csv
import os
import shutil
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from src.engine.base import ClassificationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_classification_results(results: list[ClassificationResult],
                                 output_dir: str,
                                 output_format: str = "both",
                                 file_operation: str = "copy",
                                 num_workers: int = 4):
    """
    保存分类结果。

    Args:
        results: 分类结果列表
        output_dir: 输出目录
        output_format: "folder" | "csv" | "both"
        file_operation: "copy" | "symlink"
        num_workers: IO 线程数
    """
    os.makedirs(output_dir, exist_ok=True)

    if output_format in ("folder", "both"):
        _save_as_folders(results, output_dir, file_operation, num_workers)

    if output_format in ("csv", "both"):
        _save_as_csv(results, output_dir)

    logger.info(f"分类结果已保存: {output_dir} (格式={output_format})")


def _save_as_folders(results: list[ClassificationResult], output_dir: str,
                      file_operation: str, num_workers: int):
    """按类别分文件夹存放"""
    def _process_one(result: ClassificationResult):
        class_dir = os.path.join(output_dir, result.predicted_class)
        os.makedirs(class_dir, exist_ok=True)
        filename = os.path.basename(result.image_path)
        dst = os.path.join(class_dir, filename)

        if os.path.exists(dst):
            return

        if file_operation == "symlink":
            os.symlink(os.path.abspath(result.image_path), dst)
        else:
            shutil.copy2(result.image_path, dst)

    # 用 submit + as_completed 收集每个文件的结果
    # executor.map 是惰性的，不迭代不会触发异常 → 文件复制失败会被静默吞掉
    failures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one, r): r for r in results}
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                r = futures[fut]
                failures.append((r.image_path, str(e)))
                logger.warning(f"文件操作失败: {r.image_path} -> {e}")

    if failures:
        logger.error(f"共 {len(failures)} 个文件保存失败（前 5 个）:")
        for path, err in failures[:5]:
            logger.error(f"  {path}: {err}")


def _save_as_csv(results: list[ClassificationResult], output_dir: str):
    """保存为 CSV 文件"""
    csv_path = os.path.join(output_dir, "classification_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_class", "confidence", "is_uncertain", "raw_output"])
        for r in results:
            writer.writerow([
                r.image_path,
                r.predicted_class,
                r.confidence if r.confidence is not None else "",
                r.is_uncertain,
                r.raw_output
            ])


def read_classification_folders(annotation_dir: str) -> dict[str, str]:
    """
    从文件夹结构读取分类标注。

    目录结构：
        annotation_dir/
        ├── class_A/
        │   ├── img001.jpg
        │   └── img002.jpg
        └── class_B/
            └── img003.jpg

    Returns:
        {filename: class_name} 映射，如 {"img001.jpg": "class_A"}
    """
    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(f"标注目录不存在: {annotation_dir}")

    annotations = {}
    for class_name in os.listdir(annotation_dir):
        class_dir = os.path.join(annotation_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            annotations[filename] = class_name

    logger.info(f"读取分类标注: {len(annotations)} 条，来自 {annotation_dir}")
    return annotations


def relabel_classification(class_root_dir: str, filename: str,
                            old_label: str, new_label: str,
                            csv_path: Optional[str] = None) -> str:
    """
    把已分类的文件从 old_label 子目录移到 new_label 子目录。

    同时处理三种情况：
    - 常规文件：rename 移动
    - 软链接：删除旧链接，在新目录创建指向同一目标的新链接
    - 目标已存在：抛 FileExistsError（由调用方决定如何提示）

    若提供 csv_path，同步更新该文件对应记录的 predicted_class 列。

    Args:
        class_root_dir: 类别根目录（下面是 old_label/ new_label/ 等子目录）
        filename: 文件名（不带目录）
        old_label: 原类别
        new_label: 新类别
        csv_path: 可选，分类 CSV 路径；存在时同步更新

    Returns:
        新位置的绝对路径
    """
    if old_label == new_label:
        raise ValueError("新旧标签相同，无需移动")

    old_path = os.path.join(class_root_dir, old_label, filename)
    new_dir = os.path.join(class_root_dir, new_label)
    new_path = os.path.join(new_dir, filename)

    if not os.path.lexists(old_path):
        raise FileNotFoundError(f"源文件不存在: {old_path}")
    if os.path.lexists(new_path):
        raise FileExistsError(f"目标已存在: {new_path}")

    os.makedirs(new_dir, exist_ok=True)

    # symlink 处理：删旧链接 + 新目录重建，指向同一真实目标
    if os.path.islink(old_path):
        target = os.readlink(old_path)
        # 若原链接是相对路径，转成绝对路径避免新位置断链
        if not os.path.isabs(target):
            target = os.path.abspath(
                os.path.join(os.path.dirname(old_path), target)
            )
        os.unlink(old_path)
        os.symlink(target, new_path)
    else:
        # 常规文件：shutil.move 能跨卷（os.rename 跨卷会抛 OSError）
        shutil.move(old_path, new_path)

    # 同步 CSV
    if csv_path and os.path.isfile(csv_path):
        _update_csv_label(csv_path, filename, new_label)

    logger.info(f"重标注: {old_label} -> {new_label}, {filename}")
    return new_path


def _update_csv_label(csv_path: str, filename: str, new_label: str) -> int:
    """
    更新分类 CSV 中匹配 filename（basename）行的 predicted_class / label 列。

    Returns: 更新的行数
    """
    # 读全部
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0

    header = rows[0]
    # 找列位置：优先匹配 predicted_class / label
    try:
        path_idx = next(i for i, h in enumerate(header)
                        if h.lower() in ("image_path", "path", "filename"))
    except StopIteration:
        path_idx = 0
    try:
        label_idx = next(i for i, h in enumerate(header)
                          if h.lower() in ("predicted_class", "label", "class"))
    except StopIteration:
        label_idx = 1

    updated = 0
    for row in rows[1:]:
        if len(row) <= max(path_idx, label_idx):
            continue
        if os.path.basename(row[path_idx]) == filename:
            row[label_idx] = new_label
            updated += 1

    # 写回
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return updated


def read_classification_csv(csv_path: str) -> dict[str, str]:
    """
    从 CSV 文件读取分类标注。

    CSV 格式：image_path, label（或 filename, label）

    Returns:
        {filename: class_name} 映射
    """
    annotations = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filepath, label = row[0], row[1]
            filename = os.path.basename(filepath)
            annotations[filename] = label.strip()

    logger.info(f"读取 CSV 标注: {len(annotations)} 条，来自 {csv_path}")
    return annotations

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

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(_process_one, results)


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

"""
SmartLabel CLI 命令集合（Typer）。

所有命令统一：
- --config 指向 profile yaml（与 configs/default.yaml 深合并）
- 输出目录自动创建
- 进度条基于 rich.progress
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)

from src.engine.base import Capability
from src.engine.engine_factory import EngineFactory
from src.utils.logger import get_logger

app = typer.Typer(
    name="smartlabel",
    help="SmartLabel AI 辅助标注与质检平台 CLI",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__)

# ---------- 项目路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# ==================== 配置加载 ====================

def _deep_merge(base: dict, override: dict) -> dict:
    """
    深合并两个 dict。

    规则：
    - 两边都是 dict 时递归合并
    - 其他情况（含 list）以 override 覆盖 base
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(profile_path: Optional[str]) -> dict:
    """
    加载配置。

    1. 若 default.yaml 存在，先加载作为基础
    2. 若传入 profile 路径，与 default 深合并后覆盖
    3. 仅传 profile 且 default 不存在时，直接返回 profile 内容
    """
    base = {}
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    if profile_path:
        profile_file = Path(profile_path)
        if not profile_file.is_file():
            console.print(f"[red]配置文件不存在:[/red] {profile_path}")
            raise typer.Exit(code=2)
        with open(profile_file, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}
        merged = _deep_merge(base, profile)
    else:
        merged = base

    if not merged:
        console.print("[red]配置为空：既未找到 default.yaml 也未提供 --config[/red]")
        raise typer.Exit(code=2)

    # 附带 prompts 目录，供 VLM 引擎使用
    prompts_dir = merged.get("prompts", {}).get("dir", "configs/prompts")
    if not os.path.isabs(prompts_dir):
        prompts_dir = str(PROJECT_ROOT / prompts_dir)
    merged.setdefault("engine", {})["prompts_dir"] = prompts_dir

    return merged


def _build_engine(cfg: dict):
    """根据配置构建并加载引擎"""
    engine_cfg = cfg.get("engine", {})
    engine = EngineFactory.create(engine_cfg)
    console.print(f"[cyan]加载引擎:[/cyan] {engine_cfg.get('type', 'vlm')}")
    engine.load()
    return engine


def _build_vlm_engine_from_profile(vlm_profile_path: str) -> Optional[object]:
    """从另一个 profile 构建 VLM 辅助引擎"""
    cfg = load_config(vlm_profile_path)
    engine_cfg = cfg.get("engine", {})
    if engine_cfg.get("type") != "vlm":
        console.print(f"[red]--vlm-config 指向的引擎类型不是 vlm[/red]")
        raise typer.Exit(code=2)

    engine = EngineFactory.create(engine_cfg)
    console.print(f"[cyan]加载 VLM 辅助引擎:[/cyan] {engine_cfg.get('vlm', {}).get('model_path', '?')}")
    engine.load()
    return engine


def _make_progress_callback(progress: Progress, task_id):
    """将 rich Progress 包装成 pipeline 的 progress_callback 接口"""
    def _cb(current: int, total: int, latest):
        progress.update(task_id, completed=current, total=total)
    return _cb


def _progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _print_dict_summary(title: str, data: dict, keys: Optional[list] = None):
    """打印指标摘要"""
    console.rule(f"[bold green]{title}[/bold green]")
    items = keys if keys else list(data.keys())
    for k in items:
        if k in data:
            console.print(f"  [cyan]{k}[/cyan]: {data[k]}")


# ==================== 命令：preannotate ====================

@app.command("preannotate")
def cmd_preannotate(
    config: str = typer.Option(..., "--config", "-c", help="Profile YAML 配置文件路径"),
    image_dir: str = typer.Option(..., "--image-dir", "-i", help="输入图片目录"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="输出目录"),
    task: str = typer.Option("classification", "--task", "-t",
                              help="任务类型: classification | detection"),
    categories: Optional[str] = typer.Option(
        None, "--categories",
        help="覆盖配置中的类别列表，逗号分隔。示例: normal,fatigue,distracted"
    ),
):
    """预标注：对图片执行分类或检测，输出标注文件。"""
    cfg = load_config(config)

    # CLI 参数覆盖
    if categories:
        cats_list = [c.strip() for c in categories.split(",") if c.strip()]
        if task == "classification":
            cfg.setdefault("task", {}).setdefault("classification", {})["categories"] = cats_list
        else:
            cfg.setdefault("task", {}).setdefault("detection", {})["targets"] = cats_list

    # 从 pipeline 包按需导入，避免无关命令时加载全部
    from src.pipeline.preannotate import PreAnnotatePipeline

    engine = _build_engine(cfg)

    try:
        pipeline_cfg = {
            **cfg.get("task", {}).get("classification", {}),
            **cfg.get("task", {}).get("detection", {}),
            "ui_update_interval": cfg.get("runtime", {}).get("ui_update_interval", 10),
            "num_io_workers": cfg.get("runtime", {}).get("num_io_workers", 4),
        }
        pipeline = PreAnnotatePipeline(engine, pipeline_cfg)

        with _progress_bar() as progress:
            if task == "classification":
                cats = cfg.get("task", {}).get("classification", {}).get("categories", [])
                if not cats:
                    console.print("[red]未提供分类类别，请在配置中设置或使用 --categories[/red]")
                    raise typer.Exit(code=2)
                task_id = progress.add_task("[green]分类预标注", total=100)
                cb = _make_progress_callback(progress, task_id)
                result = pipeline.run_classification(image_dir, output_dir, cats, cb)
                _print_dict_summary(
                    "分类预标注完成", result,
                    keys=["total", "category_counts", "uncertain_count"],
                )
            elif task == "detection":
                targets = cfg.get("task", {}).get("detection", {}).get("targets", [])
                if not targets:
                    console.print("[red]未提供检测目标，请在配置中设置或使用 --categories[/red]")
                    raise typer.Exit(code=2)
                task_id = progress.add_task("[green]检测预标注", total=100)
                cb = _make_progress_callback(progress, task_id)
                result = pipeline.run_detection(image_dir, output_dir, targets, cb)
                _print_dict_summary(
                    "检测预标注完成", result,
                    keys=["total", "total_detections", "label_counts"],
                )
            else:
                console.print(f"[red]未知任务类型: {task}[/red]")
                raise typer.Exit(code=2)
    finally:
        engine.unload()


# ==================== 命令：qualitycheck ====================

@app.command("qualitycheck")
def cmd_qualitycheck(
    config: str = typer.Option(..., "--config", "-c", help="主引擎 profile 配置"),
    image_dir: str = typer.Option(..., "--image-dir", "-i", help="图片目录"),
    annotation_dir: str = typer.Option(..., "--annotation-dir", "-a",
                                         help="标注目录（分类=类别文件夹或CSV；检测=VOC XML 目录）"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="输出目录"),
    task: str = typer.Option("classification", "--task", "-t",
                              help="任务类型: classification | detection"),
    vlm_config: Optional[str] = typer.Option(
        None, "--vlm-config",
        help="可选：VLM 辅助引擎 profile，开启低置信度自动升级"
    ),
    categories: Optional[str] = typer.Option(
        None, "--categories",
        help="覆盖配置中的类别列表，逗号分隔"
    ),
):
    """质检：对比人工标注与引擎判断，产出 HTML/CSV 报告。"""
    cfg = load_config(config)

    if categories:
        cats_list = [c.strip() for c in categories.split(",") if c.strip()]
        if task == "classification":
            cfg.setdefault("task", {}).setdefault("classification", {})["categories"] = cats_list
        else:
            cfg.setdefault("task", {}).setdefault("detection", {})["targets"] = cats_list

    from src.pipeline.qualitycheck import QualityCheckPipeline
    from src.report.generator import ReportGenerator

    primary = _build_engine(cfg)
    vlm = _build_vlm_engine_from_profile(vlm_config) if vlm_config else None

    try:
        qc_cfg = {
            **cfg.get("quality_check", {}),
            "ui_update_interval": cfg.get("runtime", {}).get("ui_update_interval", 10),
            "num_io_workers": cfg.get("runtime", {}).get("num_io_workers", 4),
        }
        pipeline = QualityCheckPipeline(primary, qc_cfg, vlm_engine=vlm)

        os.makedirs(output_dir, exist_ok=True)

        with _progress_bar() as progress:
            if task == "classification":
                cats = cfg.get("task", {}).get("classification", {}).get("categories", [])
                if not cats:
                    console.print("[red]未提供分类类别[/red]")
                    raise typer.Exit(code=2)
                task_id = progress.add_task("[green]分类质检", total=100)
                cb = _make_progress_callback(progress, task_id)
                result = pipeline.run_classification_qc(
                    image_dir, annotation_dir, output_dir, cats, cb,
                )
                _print_dict_summary(
                    "分类质检完成", result,
                    keys=["total_checked", "pass_count", "review_count",
                          "review_ratio", "escalated_count"],
                )
            elif task == "detection":
                targets = cfg.get("task", {}).get("detection", {}).get("targets", [])
                if not targets:
                    console.print("[red]未提供检测目标[/red]")
                    raise typer.Exit(code=2)
                iou = cfg.get("quality_check", {}).get("iou_threshold", 0.5)
                task_id = progress.add_task("[green]检测质检", total=100)
                cb = _make_progress_callback(progress, task_id)
                result = pipeline.run_detection_qc(
                    image_dir, annotation_dir, output_dir, targets,
                    iou_threshold=iou, progress_callback=cb,
                )
                _print_dict_summary(
                    "检测质检完成", result,
                    keys=["total_checked", "total_gt_boxes", "total_pred_boxes",
                          "matched_count", "miss_count", "extra_count",
                          "label_mismatch_count"],
                )
            else:
                console.print(f"[red]未知任务类型: {task}[/red]")
                raise typer.Exit(code=2)

        # 报告生成（分类）
        if task == "classification":
            reporter = ReportGenerator()
            report_fmt = cfg.get("output", {}).get("report_format", "both")
            if report_fmt in ("html", "both"):
                reporter.generate_qc_html_report(
                    result, os.path.join(output_dir, "qc_report.html")
                )
            if report_fmt in ("csv", "both"):
                reporter.generate_qc_csv_report(
                    result, os.path.join(output_dir, "qc_report.csv")
                )
            if cfg.get("output", {}).get("export_error_cases", False):
                reporter.export_error_cases(
                    result, os.path.join(output_dir, "error_cases"),
                    num_workers=cfg.get("runtime", {}).get("num_io_workers", 4),
                )

    finally:
        primary.unload()
        if vlm is not None:
            vlm.unload()


# ==================== 命令：video-classify ====================

@app.command("video-classify")
def cmd_video_classify(
    config: str = typer.Option(..., "--config", "-c", help="引擎 profile 配置"),
    video_dir: str = typer.Option(..., "--video-dir", "-v",
                                    help="视频目录或单个视频文件"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="输出目录"),
    strategy: Optional[str] = typer.Option(
        None, "--strategy", "-s",
        help="覆盖策略: vote | temporal_smooth | vlm_multiframe"
    ),
    categories: Optional[str] = typer.Option(
        None, "--categories", help="覆盖类别列表，逗号分隔"
    ),
):
    """视频片段分类：滑动窗口 → 推理 → 时序平滑 → 输出 CSV/JSON。"""
    cfg = load_config(config)

    # 参数覆盖
    video_cfg = cfg.setdefault("task", {}).setdefault("video", {})
    if strategy:
        video_cfg["strategy"] = strategy
    if categories:
        video_cfg["categories"] = [c.strip() for c in categories.split(",") if c.strip()]

    if not video_cfg.get("categories"):
        console.print("[red]未提供视频类别，请在配置或 --categories 中指定[/red]")
        raise typer.Exit(code=2)

    from src.pipeline.video_classify import VideoClassifyPipeline
    from src.report.generator import ReportGenerator

    engine = _build_engine(cfg)

    try:
        pipeline_cfg = {
            **video_cfg,
            "temp_dir": cfg.get("runtime", {}).get("temp_dir"),
            "ui_update_interval": cfg.get("runtime", {}).get("ui_update_interval", 1),
        }
        pipeline = VideoClassifyPipeline(engine, pipeline_cfg)

        with _progress_bar() as progress:
            task_id = progress.add_task("[green]视频分类", total=100)
            cb = _make_progress_callback(progress, task_id)
            results = pipeline.run_batch(video_dir, output_dir, cb)

        # 生成每个视频的 HTML 报告
        reporter = ReportGenerator()
        for r in results:
            if not r.clips:
                continue
            video_name = os.path.splitext(os.path.basename(r.video_path))[0]
            per_output = (
                output_dir if os.path.isfile(video_dir)
                else os.path.join(output_dir, video_name)
            )
            os.makedirs(per_output, exist_ok=True)
            html_path = os.path.join(per_output, f"{video_name}_report.html")
            reporter.generate_video_report(r, html_path)

        console.rule("[bold green]视频分类完成[/bold green]")
        for r in results:
            console.print(
                f"  [cyan]{os.path.basename(r.video_path)}[/cyan]: "
                f"{len(r.clips)} 个片段, 统计={r.statistics}"
            )

    finally:
        engine.unload()


# ==================== 命令：benchmark ====================

@app.command("benchmark")
def cmd_benchmark(
    config: str = typer.Option(..., "--config", "-c", help="引擎 profile 配置"),
    image_dir: str = typer.Option(..., "--image-dir", "-i", help="图片目录"),
    gt_dir: str = typer.Option(..., "--gt-dir", "-g", help="真值标注目录"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="输出目录"),
    task: str = typer.Option("classification", "--task", "-t",
                              help="任务类型: classification | detection"),
    categories: Optional[str] = typer.Option(
        None, "--categories", help="覆盖类别列表，逗号分隔"
    ),
):
    """基准评估：跑引擎推理并与真值对比，输出 accuracy/mAP 等指标。"""
    cfg = load_config(config)

    if categories:
        cats_list = [c.strip() for c in categories.split(",") if c.strip()]
        if task == "classification":
            cfg.setdefault("task", {}).setdefault("classification", {})["categories"] = cats_list
        else:
            cfg.setdefault("task", {}).setdefault("detection", {})["targets"] = cats_list

    from src.io.image_loader import scan_images
    from src.io.classification_io import read_classification_folders, read_classification_csv
    from src.io.voc_xml import read_voc_annotations
    from src.utils.metrics import Evaluator

    engine = _build_engine(cfg)
    os.makedirs(output_dir, exist_ok=True)

    try:
        images = scan_images(image_dir)
        if not images:
            console.print(f"[red]未找到图片: {image_dir}[/red]")
            raise typer.Exit(code=2)

        if task == "classification":
            cats = cfg.get("task", {}).get("classification", {}).get("categories", [])
            if not cats:
                console.print("[red]未提供分类类别[/red]")
                raise typer.Exit(code=2)

            if os.path.isfile(gt_dir) and gt_dir.endswith(".csv"):
                gt_map = read_classification_csv(gt_dir)
            else:
                gt_map = read_classification_folders(gt_dir)

            y_true, y_pred = [], []
            with _progress_bar() as progress:
                task_id = progress.add_task("[green]分类评估", total=len(images))
                for img_path in images:
                    fn = os.path.basename(img_path)
                    if fn not in gt_map:
                        progress.advance(task_id)
                        continue
                    pred = engine.classify(img_path, cats)
                    y_true.append(gt_map[fn])
                    y_pred.append(pred.predicted_class)
                    progress.advance(task_id)

            if not y_true:
                console.print("[red]未匹配到任何 GT 样本[/red]")
                raise typer.Exit(code=2)

            report = Evaluator.classification_report(y_true, y_pred, cats)

            # 输出
            with open(os.path.join(output_dir, "benchmark.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            Evaluator.plot_confusion_matrix(
                report["confusion_matrix"], report["class_names"],
                output_path=os.path.join(output_dir, "confusion_matrix.png"),
            )
            _print_dict_summary(
                "分类评估结果", report,
                keys=["accuracy", "macro_f1", "weighted_f1"],
            )

        elif task == "detection":
            if not engine.supports(Capability.DETECT):
                console.print("[red]当前引擎不支持检测能力[/red]")
                raise typer.Exit(code=2)
            targets = cfg.get("task", {}).get("detection", {}).get("targets", [])
            if not targets:
                console.print("[red]未提供检测目标[/red]")
                raise typer.Exit(code=2)

            gt_annotations = read_voc_annotations(gt_dir)
            pred_all, gt_all = [], []
            with _progress_bar() as progress:
                task_id = progress.add_task("[green]检测评估", total=len(images))
                for img_path in images:
                    fn = os.path.basename(img_path)
                    stem = os.path.splitext(fn)[0]
                    gt = gt_annotations.get(fn) or gt_annotations.get(stem)
                    if gt is None:
                        progress.advance(task_id)
                        continue
                    det = engine.detect(img_path, targets)
                    pred_all.append(det.detections)
                    gt_all.append(gt["objects"])
                    progress.advance(task_id)

            if not gt_all:
                console.print("[red]未匹配到任何 GT[/red]")
                raise typer.Exit(code=2)

            report = Evaluator.detection_report(pred_all, gt_all, iou_thresholds=[0.5])
            with open(os.path.join(output_dir, "benchmark.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            _print_dict_summary(
                "检测评估结果", report,
                keys=["mAP", "total_gt", "total_pred"],
            )
        else:
            console.print(f"[red]未知任务类型: {task}[/red]")
            raise typer.Exit(code=2)

        console.print(f"[cyan]报告已输出:[/cyan] {output_dir}")

    finally:
        engine.unload()


if __name__ == "__main__":
    app()

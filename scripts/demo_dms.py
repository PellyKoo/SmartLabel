"""
SmartLabel DMS 驾驶员行为监测 — 演示脚本

展示三个典型场景：
  1. 大批量图片 VLM 预标注
  2. 外包标注质检（含低置信度 VLM 升级）
  3. 视频片段自动分类

使用方式：
  # dry-run（Mock 引擎，无需 GPU，验证流程）
  python scripts/demo_dms.py --dry-run

  # 真实 VLM（需提前下载模型）
  python scripts/demo_dms.py --model-path models/Qwen2-VL-7B-Instruct --output-dir demo_output

  # 只跑其中一个场景
  python scripts/demo_dms.py --dry-run --scenario preannotate
"""
import argparse
import os
import sys
import time
import shutil
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from src.engine.base import (
    BaseEngine, Capability, ClassificationResult, VideoClipResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==================== 颜色输出 ====================
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN  = "\033[36m"
RED   = "\033[31m"
DIM   = "\033[2m"


def c(text, color): return f"{color}{text}{RESET}"
def header(text):   print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
def section(text):  print(f"\n{BOLD}{GREEN}▶ {text}{RESET}")
def ok(text):       print(f"  {GREEN}✅ {text}{RESET}")
def warn(text):     print(f"  {YELLOW}⚠️  {text}{RESET}")
def info(text):     print(f"  {DIM}{text}{RESET}")


def banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║       SmartLabel — DMS 驾驶员行为监测演示               ║
║       AI 辅助标注与质检平台  Phase 7 Demo                ║
╚══════════════════════════════════════════════════════════╝{RESET}""")


# ==================== Mock 引擎 ====================

class DmsMockEngine(BaseEngine):
    """
    仿真引擎：按预设规则返回标签，模拟真实模型行为。
    用于 dry-run / CI 测试，无需 GPU。
    """
    LABELS = ["normal", "fatigue", "distracted", "phone", "smoke"]
    # 仿真分布：接近真实场景（大量 normal，少量异常）
    WEIGHTS = [0.60, 0.15, 0.12, 0.08, 0.05]

    def __init__(self, delay: float = 0.05):
        import random
        self._rng = random.Random(42)
        self._delay = delay   # 模拟推理耗时
        self._call = 0

    @property
    def capabilities(self):
        return {Capability.CLASSIFY, Capability.VIDEO_MULTIFRAME}

    def load(self): pass

    def unload(self): pass

    @property
    def is_loaded(self): return True

    def get_engine_info(self):
        return {"type": "mock_dms", "model": "DmsMockEngine"}

    def classify(self, image_path: str, categories: list) -> ClassificationResult:
        time.sleep(self._delay)
        self._call += 1
        label = self._rng.choices(self.LABELS, weights=self.WEIGHTS)[0]
        if label not in categories:
            label = categories[0]
        conf = round(self._rng.uniform(0.55, 0.99), 2)
        return ClassificationResult(
            image_path=image_path,
            predicted_class=label,
            confidence=conf,
            is_uncertain=(conf < 0.75),
            raw_output=f'{{"observation":"mock","label":"{label}"}}',
        )

    def classify_video_frames(self, frame_paths, categories):
        result = self.classify(frame_paths[0], categories)
        return result


# ==================== 测试数据生成 ====================

DMS_CATEGORIES = ["normal", "fatigue", "distracted", "phone", "smoke"]


def make_fake_image(path: str):
    """写一个 1x1 白色 JPEG（cv2 可读）"""
    import numpy as np
    import cv2
    img = (255 * np.ones((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(path, img)


def setup_demo_data(base_dir: str, n_images: int = 30) -> dict:
    """创建临时演示数据集"""
    import random
    rng = random.Random(42)

    image_dir = os.path.join(base_dir, "images")
    ann_dir   = os.path.join(base_dir, "annotations")
    video_dir = os.path.join(base_dir, "videos")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # 创建标注目录（每类一个子目录）
    label_pool = ["normal"] * 18 + ["fatigue"] * 4 + ["distracted"] * 3 + \
                 ["phone"] * 3 + ["smoke"] * 2
    rng.shuffle(label_pool)

    for cls in DMS_CATEGORIES:
        os.makedirs(os.path.join(ann_dir, cls), exist_ok=True)

    image_paths = []
    for i in range(n_images):
        fname = f"dms_{i:04d}.jpg"
        img_path = os.path.join(image_dir, fname)
        make_fake_image(img_path)
        image_paths.append(img_path)

        # 标注：大部分正确，10% 故意标错（模拟外包质量问题）
        true_label = label_pool[i % len(label_pool)]
        ann_label = true_label
        if rng.random() < 0.10:
            ann_label = rng.choice([l for l in DMS_CATEGORIES if l != true_label])
        shutil.copy(img_path, os.path.join(ann_dir, ann_label, fname))

    # 生成一段假视频（复制一帧多次作为视频）
    try:
        import cv2
        import numpy as np
        video_path = os.path.join(video_dir, "dms_sample.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(video_path, fourcc, 15.0, (320, 240))
        for _ in range(150):  # 10s @ 15fps
            frame = (rng.randint(0, 255) * np.ones((240, 320, 3), dtype=np.uint8))
            vw.write(frame.astype("uint8"))
        vw.release()
        has_video = True
    except Exception:
        has_video = False
        video_path = None

    return {
        "image_dir": image_dir,
        "annotation_dir": ann_dir,
        "video_dir": video_dir,
        "video_path": video_path,
        "has_video": has_video,
        "n_images": n_images,
    }


# ==================== 场景 1：预标注 ====================

def demo_preannotate(engine, data: dict, output_dir: str):
    section("场景 1 — 大批量图片 VLM 预标注")
    info(f"输入: {data['image_dir']} ({data['n_images']} 张)")
    info(f"类别: {DMS_CATEGORIES}")

    from src.pipeline.preannotate import PreAnnotatePipeline

    cfg = {"output_format": "both", "file_operation": "copy",
           "num_io_workers": 4, "ui_update_interval": 5}
    pipe = PreAnnotatePipeline(engine, cfg)

    out = os.path.join(output_dir, "preannotate")
    t0 = time.perf_counter()

    counts = Counter()
    uncertain = 0
    total = data["n_images"]

    def progress(cur, tot, latest):
        nonlocal uncertain
        if latest and isinstance(latest, ClassificationResult):
            counts[latest.predicted_class] += 1
            if latest.is_uncertain:
                uncertain += 1
        bar_len = 30
        filled = int(bar_len * cur / max(tot, 1))
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {cur}/{tot}", end="", flush=True)

    result = pipe.run_classification(
        data["image_dir"], out, DMS_CATEGORIES, progress_callback=progress
    )
    elapsed = time.perf_counter() - t0
    print()  # 换行

    # 汇总
    total_done = result.get("total", total)
    ok(f"完成 {total_done} 张，耗时 {elapsed:.1f}s（{total_done/elapsed:.1f} 张/秒）")
    ok(f"各类别分布: {dict(result.get('category_counts', counts))}")
    warn(f"不确定样本: {result.get('uncertain_count', uncertain)} 张（建议人工复核）")
    info(f"输出: {out}/")

    # 生成预标注汇总报告
    from src.report.generator import ReportGenerator
    reporter = ReportGenerator()
    html_path = os.path.join(out, "preannotate_report.html")
    reporter.generate_preannotate_summary(result, html_path)
    ok(f"报告: {html_path}")

    return result


# ==================== 场景 2：质检 ====================

def demo_qualitycheck(engine, data: dict, output_dir: str):
    section("场景 2 — 外包标注质检（低置信度 VLM 升级）")
    info(f"图片: {data['image_dir']}")
    info(f"标注: {data['annotation_dir']} (含约 10% 故意标错模拟外包质量)")
    info(f"策略: 自有模型初筛 → 低置信度自动升级到 VLM 深度复核")

    from src.pipeline.qualitycheck import QualityCheckPipeline

    qc_cfg = {
        "escalation_enabled": True,
        "escalation_threshold": 0.80,
        "ui_update_interval": 5,
    }
    # 演示中主/辅均用同一引擎（生产中主为自有模型，辅为 VLM）
    pipe = QualityCheckPipeline(engine, qc_cfg, vlm_engine=engine)

    out = os.path.join(output_dir, "qualitycheck")
    t0 = time.perf_counter()

    reviewed = [0]
    escalated = [0]

    def progress(cur, tot, latest):
        if latest and isinstance(latest, dict):
            if latest.get("escalated"):
                escalated[0] += 1
            if not latest.get("is_consistent"):
                reviewed[0] += 1
        bar_len = 30
        filled = int(bar_len * cur / max(tot, 1))
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {cur}/{tot}  ⚠️ 异议:{reviewed[0]}  🔺 升级:{escalated[0]}", end="", flush=True)

    result = pipe.run_classification_qc(
        data["image_dir"], data["annotation_dir"], out,
        DMS_CATEGORIES, progress_callback=progress
    )
    elapsed = time.perf_counter() - t0
    print()

    ok(f"检查 {result.get('total_checked', 0)} 张，耗时 {elapsed:.1f}s")
    ok(f"通过率: {(1 - result.get('review_ratio', 0)) * 100:.1f}%  "
       f"异议率: {result.get('review_ratio', 0) * 100:.1f}%")
    ok(f"升级到 VLM 复核: {result.get('escalated_count', 0)} 张")

    # 错误模式
    error_cases = result.get("error_cases", [])
    if error_cases:
        warn("高频错误模式（标注规范问题）:")
        for ec in error_cases[:5]:
            if ec.get("samples"):
                s = ec["samples"][0]
                print(f"    {s['human_label']} → {s['engine_label']}: "
                      f"{ec['count']} 次")

    # 生成报告
    from src.report.generator import ReportGenerator
    reporter = ReportGenerator()
    html_path = os.path.join(out, "qc_report.html")
    csv_path  = os.path.join(out, "qc_report.csv")
    reporter.generate_qc_html_report(result, html_path)
    reporter.generate_qc_csv_report(result, csv_path)
    if result.get("review_samples"):
        reporter.export_error_cases(result, os.path.join(out, "error_cases"))
    ok(f"HTML 报告: {html_path}")
    ok(f"CSV 报告:  {csv_path}")
    info(f"可用浏览器打开 HTML 报告查看待复核样本")

    return result


# ==================== 场景 3：视频分类 ====================

def demo_video(engine, data: dict, output_dir: str):
    section("场景 3 — 驾驶视频自动片段分类")

    if not data.get("has_video") or not data.get("video_path"):
        warn("未生成视频文件（cv2 不可用），跳过视频演示")
        return None

    video_path = data["video_path"]
    info(f"视频: {video_path}")
    info(f"策略: 滑动窗口 5s/步长 2.5s + 时序平滑 → 片段检测")

    from src.pipeline.video_classify import VideoClassifyPipeline

    vcfg = {
        "categories": DMS_CATEGORIES,
        "window_sec": 3.0,
        "stride_sec": 1.5,
        "sample_fps": 1.0,
        "strategy": "temporal_smooth",
        "smooth_window": 3,
        "min_segment_sec": 1.0,
        "output_format": "both",
        "ui_update_interval": 1,
    }
    pipe = VideoClassifyPipeline(engine, vcfg)
    out = os.path.join(output_dir, "video")
    t0 = time.perf_counter()

    win_count = [0]
    def progress(cur, tot, info_dict):
        win_count[0] = cur
        print(f"\r  窗口 {cur}/{tot}  最新: {info_dict.get('label','?')}", end="", flush=True)

    result = pipe.run(video_path, out, progress_callback=progress)
    elapsed = time.perf_counter() - t0
    print()

    ok(f"共 {len(result.clips)} 个片段，耗时 {elapsed:.1f}s")
    for clip in result.clips:
        label_color = GREEN if clip["label"] == "normal" else YELLOW
        print(f"    {DIM}[{clip['start_sec']:.1f}s - {clip['end_sec']:.1f}s]{RESET}"
              f"  {label_color}{clip['label']}{RESET}"
              f"  {DIM}({clip.get('duration_sec', 0):.1f}s){RESET}")

    ok(f"各类时长: {result.statistics}")

    # 生成 HTML 报告
    from src.report.generator import ReportGenerator
    html_path = os.path.join(out, "video_report.html")
    ReportGenerator().generate_video_report(result, html_path)
    ok(f"时间轴报告: {html_path}")

    return result


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="SmartLabel DMS 演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/demo_dms.py --dry-run                    # Mock 引擎，无需 GPU
  python scripts/demo_dms.py --dry-run --scenario qc      # 只演示质检场景
  python scripts/demo_dms.py --model-path models/Qwen2-VL-7B-Instruct \\
      --quantization bnb4 --output-dir demo_output        # 真实 VLM
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="使用 Mock 引擎运行（无需 GPU 和模型文件）")
    parser.add_argument("--model-path", default="models/Qwen2-VL-7B-Instruct",
                        help="VLM 模型路径（真实运行时必填）")
    parser.add_argument("--quantization", default="bnb4",
                        choices=["bnb4", "bnb8", "none", "awq"],
                        help="VLM 量化方式（默认 bnb4）")
    parser.add_argument("--output-dir", default="demo_output",
                        help="演示结果输出目录（默认 demo_output）")
    parser.add_argument("--n-images", type=int, default=30,
                        help="演示用图片数量（默认 30，dry-run 时自动生成）")
    parser.add_argument("--scenario", choices=["all", "preannotate", "qc", "video"],
                        default="all", help="运行哪个场景（默认 all）")
    args = parser.parse_args()

    banner()
    print(f"\n  模式:   {'dry-run (Mock 引擎)' if args.dry_run else '真实 VLM'}")
    print(f"  输出:   {args.output_dir}")
    print(f"  场景:   {args.scenario}")

    # ---- 引擎初始化 ----
    section("引擎初始化")
    if args.dry_run:
        engine = DmsMockEngine(delay=0.02)
        ok("Mock 引擎已就绪（无需 GPU）")
    else:
        from src.engine.vlm_engine import VLMEngine
        vlm_cfg = {
            "model_path": args.model_path,
            "quantization": args.quantization,
            "device": "cuda:0",
            "torch_dtype": "float16",
            "max_new_tokens": 128,
            "multi_sample": True,
            "sample_count": 3,
            "temperature": 0.6,
        }
        engine = VLMEngine(vlm_cfg)
        print(f"  加载模型: {args.model_path}")
        t0 = time.perf_counter()
        engine.load()
        ok(f"VLM 引擎加载完成，耗时 {time.perf_counter()-t0:.1f}s")
        info(f"引擎信息: {engine.get_engine_info()}")

    # ---- 生成演示数据 ----
    section("准备演示数据集")
    with tempfile.TemporaryDirectory(prefix="smartlabel_demo_") as tmpdir:
        info(f"临时数据目录: {tmpdir}")
        data = setup_demo_data(tmpdir, n_images=args.n_images)
        ok(f"已生成 {data['n_images']} 张仿真 DMS 图片")
        if data["has_video"]:
            ok("已生成 10s 仿真驾驶视频")
        else:
            warn("cv2 VideoWriter 不可用，跳过视频数据生成")

        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # ---- 场景执行 ----
        t_total = time.perf_counter()

        if args.scenario in ("all", "preannotate"):
            results["preannotate"] = demo_preannotate(engine, data, output_dir)

        if args.scenario in ("all", "qc"):
            results["qc"] = demo_qualitycheck(engine, data, output_dir)

        if args.scenario in ("all", "video"):
            results["video"] = demo_video(engine, data, output_dir)

        total_elapsed = time.perf_counter() - t_total

        # ---- 汇总 ----
        print(f"\n{BOLD}{CYAN}{'─'*60}")
        print(f"  演示完成  总耗时: {total_elapsed:.1f}s")
        print(f"  输出目录: {output_dir}")
        if args.scenario in ("all", "preannotate"):
            print(f"  预标注报告: {output_dir}/preannotate/preannotate_report.html")
        if args.scenario in ("all", "qc"):
            print(f"  质检报告:   {output_dir}/qualitycheck/qc_report.html")
        if args.scenario in ("all", "video") and results.get("video"):
            print(f"  视频报告:   {output_dir}/video/video_report.html")
        print(f"{'─'*60}{RESET}\n")

    if not args.dry_run:
        engine.unload()


if __name__ == "__main__":
    main()

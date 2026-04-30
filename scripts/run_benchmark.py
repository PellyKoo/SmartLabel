"""
SmartLabel 基准测试脚本

对已标注数据集运行引擎推理，计算 accuracy/F1/混淆矩阵等指标。

用法：
  # dry-run（Mock 引擎）
  python scripts/run_benchmark.py --dry-run \\
      --image-dir /data/images --gt-dir /data/annotations

  # 真实 VLM
  python scripts/run_benchmark.py \\
      --config configs/profiles/dms_preannotate.yaml \\
      --image-dir /data/benchmark/images \\
      --gt-dir /data/benchmark/annotations \\
      --output-dir /data/benchmark/results
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def ok(t):   print(f"  {GREEN}✅ {t}{RESET}")
def warn(t): print(f"  {YELLOW}⚠️  {t}{RESET}")
def err(t):  print(f"  {RED}✗ {t}{RESET}")
def info(t): print(f"  {DIM}{t}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="SmartLabel 基准测试")
    parser.add_argument("--config", help="引擎 profile YAML（与 default.yaml 合并）")
    parser.add_argument("--image-dir", required=True, help="图片目录")
    parser.add_argument("--gt-dir", required=True,
                        help="真值标注目录（文件夹结构或 CSV）")
    parser.add_argument("--output-dir", default="benchmark_output",
                        help="输出目录（默认 benchmark_output）")
    parser.add_argument("--categories", help="类别列表，逗号分隔（覆盖配置）")
    parser.add_argument("--dry-run", action="store_true",
                        help="使用 Mock 引擎（无需 GPU）")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最多评估 N 张，0=全部（调试用）")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}SmartLabel 基准测试{RESET}")
    print(f"  图片目录: {args.image_dir}")
    print(f"  真值目录: {args.gt_dir}")
    print(f"  输出目录: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 类别 ----
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.config:
        from src.cli.commands import load_config
        cfg = load_config(args.config)
        categories = cfg.get("task", {}).get("classification", {}).get("categories", [])
    else:
        categories = []

    if not categories:
        categories = ["normal", "fatigue", "distracted", "phone", "smoke"]
        warn(f"未指定类别，使用 DMS 默认类别: {categories}")

    # ---- 引擎 ----
    if args.dry_run:
        # 复用 demo_dms.py 的 MockEngine
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from demo_dms import DmsMockEngine
        engine = DmsMockEngine(delay=0.01)
        ok("Mock 引擎已就绪（dry-run）")
    else:
        if not args.config:
            err("真实运行需要 --config 指定引擎配置")
            sys.exit(1)
        from src.cli.commands import load_config
        from src.engine.engine_factory import EngineFactory
        cfg = load_config(args.config)
        engine = EngineFactory.create(cfg["engine"])
        print("  加载引擎...")
        t0 = time.perf_counter()
        engine.load()
        ok(f"引擎已加载，耗时 {time.perf_counter()-t0:.1f}s")

    # ---- 加载真值 ----
    from src.io.classification_io import read_classification_folders, read_classification_csv
    from src.io.image_loader import scan_images

    if os.path.isfile(args.gt_dir) and args.gt_dir.endswith(".csv"):
        gt_map = read_classification_csv(args.gt_dir)
    elif os.path.isdir(args.gt_dir):
        gt_map = read_classification_folders(args.gt_dir)
    else:
        err(f"真值路径无效: {args.gt_dir}")
        sys.exit(1)
    info(f"加载 {len(gt_map)} 条真值标注")

    images = scan_images(args.image_dir)
    if args.max_images > 0:
        images = images[:args.max_images]
        warn(f"仅评估前 {len(images)} 张（--max-images）")

    # ---- 推理 ----
    print(f"\n  开始推理（{len(images)} 张）...")
    y_true, y_pred = [], []
    t_total = time.perf_counter()

    for i, img_path in enumerate(images):
        fn = os.path.basename(img_path)
        if fn not in gt_map:
            continue
        result = engine.classify(img_path, categories)
        y_true.append(gt_map[fn])
        y_pred.append(result.predicted_class)

        bar_len = 35
        filled = int(bar_len * (i + 1) / len(images))
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {i+1}/{len(images)}", end="", flush=True)

    elapsed = time.perf_counter() - t_total
    print()

    if not y_true:
        err("未匹配到任何 GT 样本，检查目录结构")
        sys.exit(1)

    # ---- 计算指标 ----
    from src.utils.metrics import Evaluator
    report = Evaluator.classification_report(y_true, y_pred, categories)

    # ---- 输出结果 ----
    print(f"\n{BOLD}{'─'*55}")
    print(f"  评估结果  ({len(y_true)} 张，耗时 {elapsed:.1f}s，{len(y_true)/elapsed:.1f} 张/秒)")
    print(f"{'─'*55}{RESET}")
    print(f"  Accuracy:    {BOLD}{report['accuracy']*100:.2f}%{RESET}")
    print(f"  Macro F1:    {BOLD}{report['macro_f1']:.4f}{RESET}")
    print(f"  Weighted F1: {BOLD}{report['weighted_f1']:.4f}{RESET}")

    print(f"\n  {DIM}{'类别':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}{RESET}")
    for cls in categories:
        m = report["per_class"].get(cls, {})
        if not m:
            continue
        f1_color = GREEN if m["f1"] >= 0.85 else (YELLOW if m["f1"] >= 0.70 else RED)
        print(f"  {cls:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{f1_color}{m['f1']:>10.4f}{RESET} {m['support']:>8}")

    # 混淆矩阵
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    try:
        Evaluator.plot_confusion_matrix(
            report["confusion_matrix"], report["class_names"], output_path=cm_path
        )
        ok(f"混淆矩阵: {cm_path}")
    except Exception as e:
        warn(f"混淆矩阵生成失败: {e}")

    # 保存 JSON
    report_path = os.path.join(args.output_dir, "benchmark_report.json")
    report["meta"] = {
        "n_evaluated": len(y_true),
        "n_total_images": len(images),
        "elapsed_sec": round(elapsed, 2),
        "throughput_per_sec": round(len(y_true) / elapsed, 2),
        "categories": categories,
        "image_dir": args.image_dir,
        "gt_dir": args.gt_dir,
        "dry_run": args.dry_run,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ok(f"完整报告: {report_path}")

    if not args.dry_run:
        engine.unload()

    # 退出码：Accuracy < 0.7 返回 1（方便 CI 检查）
    if report["accuracy"] < 0.70:
        warn(f"Accuracy {report['accuracy']:.2f} 低于 0.70 阈值")
        sys.exit(1)
    ok(f"Benchmark 完成，Accuracy={report['accuracy']:.2f}")


if __name__ == "__main__":
    main()

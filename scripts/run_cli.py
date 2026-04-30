"""
SmartLabel CLI 入口。

使用方式：
    python scripts/run_cli.py --help
    python scripts/run_cli.py preannotate --config configs/profiles/xxx.yaml ...
    python scripts/run_cli.py qualitycheck --config ... --vlm-config ...
    python scripts/run_cli.py video-classify --config ... --video-dir ...
    python scripts/run_cli.py benchmark --config ... --image-dir ... --gt-dir ...
"""
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 下强制 UTF-8 输出，避免 GBK 无法编码中文/表情
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from src.cli import app

if __name__ == "__main__":
    app()

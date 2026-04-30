"""
SmartLabel Web 服务入口（FastAPI + uvicorn）。

使用方式：
    # 本地开发（仅本机访问）
    python scripts/run_web.py

    # 内网访问
    python scripts/run_web.py --host 0.0.0.0 --port 8000

    # 自动重载（开发时）
    python scripts/run_web.py --reload
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def main():
    parser = argparse.ArgumentParser(description="SmartLabel Web 服务")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址（默认 127.0.0.1，内网用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--reload", action="store_true",
                        help="开发模式：代码变更自动重载")
    parser.add_argument("--workers", type=int, default=1,
                        help="worker 数量（GPU 单卡推荐 1）")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

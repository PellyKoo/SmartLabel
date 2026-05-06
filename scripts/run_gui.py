"""
SmartLabel GUI 入口（PyQt5）。

使用方式：
    python scripts/run_gui.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PyInstaller 打包后资源根目录
_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

# Windows 下强制 UTF-8
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# HiDPI 支持（必须在 QApplication 创建前设置）
from PyQt5.QtCore import Qt, QCoreApplication

QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from PyQt5.QtWidgets import QApplication  # noqa: E402
from PyQt5.QtGui import QIcon  # noqa: E402

from src.gui import MainWindow  # noqa: E402

ICON_PATH = _BUNDLE_DIR / "src" / "icon" / "smartlabel_icon.ico"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SmartLabel")
    app.setOrganizationName("SmartLabel")

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

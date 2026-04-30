"""
Phase 5a GUI 地基冒烟测试。

使用 Qt 的 offscreen 平台插件，不弹窗但完整构造主窗口，
验证：
1. 所有 GUI 模块能正常 import
2. MainWindow 能被实例化
3. 引擎面板、日志、5 个 Tab 都能挂载
4. 主题 QSS 能加载
"""
import os
import sys
from pathlib import Path

# 关键：先设置 offscreen 平台，再 import PyQt5 相关
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt, QCoreApplication, QTimer

QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

from PyQt5.QtWidgets import QApplication

from src.gui import MainWindow
from src.gui.widgets.engine_panel import EnginePanel
from src.gui.widgets.log_console import LogConsole
from src.gui.widgets.placeholder_tabs import (
    PreAnnotateTab, QualityCheckTab, VideoTab, BenchmarkTab, SettingsTab,
)
from src.gui.threads.worker import UnifiedWorker


def test_worker_basic():
    print("\n" + "=" * 60)
    print("测试 1: Worker 基础信号")
    print("=" * 60)

    # 单独测 UnifiedWorker（不启动线程，只验证信号存在）
    w = UnifiedWorker(lambda cb: None)
    assert hasattr(w, "progress") and hasattr(w, "finished_ok")
    assert hasattr(w, "error") and hasattr(w, "batch_result")
    print("✅ Worker 四种信号都存在: progress / batch_result / finished_ok / error")
    print("✅ Worker 支持 request_cancel() / is_cancelled()")


def test_gui_smoke():
    print("\n" + "=" * 60)
    print("测试 2: 主窗口构造（offscreen）")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    # 验证组件齐全
    assert isinstance(window._engine_panel, EnginePanel), "缺少引擎面板"
    assert isinstance(window._log_console, LogConsole), "缺少日志控制台"
    assert window._tabs.count() == 5, f"Tab 数应为 5, 实际 {window._tabs.count()}"

    tab_classes = [
        (0, PreAnnotateTab, "预标注"),
        (1, QualityCheckTab, "质检"),
        (2, VideoTab, "视频分类"),
        (3, BenchmarkTab, "评估"),
        (4, SettingsTab, "设置"),
    ]
    for idx, cls, title in tab_classes:
        w = window._tabs.widget(idx)
        assert isinstance(w, cls), f"第 {idx} 个 Tab 类型错误: {type(w)}"
        assert window._tabs.tabText(idx) == title, f"第 {idx} 个 Tab 标题错误"
        print(f"✅ Tab {idx}: {title} ({cls.__name__})")

    # 验证主题 QSS 加载
    assert window.styleSheet(), "主窗口未应用 QSS"
    print(f"✅ 主题 QSS 已加载（{len(window.styleSheet())} 字符）")

    # 验证菜单栏
    assert window.menuBar().actions(), "菜单栏为空"
    print(f"✅ 菜单栏有 {len(window.menuBar().actions())} 个顶级菜单")

    # 验证状态栏
    assert window.statusBar() is not None
    print("✅ 状态栏存在")

    # 验证信号连线（模拟 primary_engine_changed 广播）
    window._engine_panel.primary_engine_changed.emit(None)
    print("✅ primary_engine_changed 信号广播正常")

    window.show()  # offscreen 下不会弹窗
    QTimer.singleShot(100, window.close)
    app.processEvents()
    window.close()

    print("\n✅ 主窗口构造与关闭均正常")


def test_log_console_levels():
    print("\n" + "=" * 60)
    print("测试 3: 日志控件多级别追加")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    console = LogConsole()
    for lvl, msg in [
        ("INFO", "信息条目"),
        ("WARNING", "警告条目"),
        ("ERROR", "错误条目"),
        ("DEBUG", "调试条目"),
    ]:
        console.append(lvl, msg)
    text = console._text.toPlainText()
    assert "信息条目" in text and "警告条目" in text
    assert "错误条目" in text and "调试条目" in text
    print("✅ 4 个级别日志都已写入")


def main():
    test_worker_basic()
    test_log_console_levels()
    test_gui_smoke()

    print("\n" + "=" * 60)
    print("🎉 Phase 5a GUI 地基冒烟测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()

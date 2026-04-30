"""
类别输入控件验证：
1. parse_categories_txt 解析规则（空行、注释、顺序）
2. make_category_input 回填逻辑
3. 各 Tab 的 _categories_row / _targets_row 属性可用
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt, QCoreApplication
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
from PyQt5.QtWidgets import QApplication

from src.gui.widgets._category_input import parse_categories_txt, make_category_input


def test_parse_categories_txt():
    print("\n" + "=" * 60)
    print("测试 1: parse_categories_txt")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("""# 这是注释
normal
fatigue

# 空行和注释都应被跳过
  distracted
yawning
""")
        path = f.name

    try:
        cats = parse_categories_txt(path)
        assert cats == ["normal", "fatigue", "distracted", "yawning"], f"解析结果: {cats}"
        print(f"✅ 解析结果顺序正确: {cats}")
        print("✅ 跳过空行、# 注释、去首尾空白")
    finally:
        os.unlink(path)

    # 不存在文件
    try:
        parse_categories_txt("/nonexistent/path.txt")
    except FileNotFoundError:
        print("✅ 不存在文件抛 FileNotFoundError")


def test_make_category_input():
    print("\n" + "=" * 60)
    print("测试 2: make_category_input 构造")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)

    edit, container = make_category_input(placeholder="test-ph")
    assert edit.placeholderText() == "test-ph"
    print("✅ placeholder 应用正确")

    # container 含 QLineEdit 和 QPushButton 两个子控件
    assert edit.parent() is not None
    print("✅ 返回 (edit, container) 元组")

    # 设置文本、读取
    edit.setText("a,b,c")
    assert edit.text() == "a,b,c"
    print("✅ 文本读写正常")

    # 禁用容器后，按钮应一并禁用（通过 parent 继承）
    container.setEnabled(False)
    assert not edit.isEnabled()
    print("✅ 禁用容器后 edit 也禁用（视觉一致）")


def test_tabs_use_rows():
    print("\n" + "=" * 60)
    print("测试 3: 4 个 Tab 的 _categories_row / _targets_row 属性")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)

    from src.gui.widgets import (
        PreAnnotateTab, QualityCheckTab, VideoTab, BenchmarkTab,
    )

    pa = PreAnnotateTab()
    assert hasattr(pa, "_categories_row") and hasattr(pa, "_targets_row")
    print("✅ PreAnnotateTab 有 _categories_row / _targets_row")

    qc = QualityCheckTab()
    assert hasattr(qc, "_categories_row") and hasattr(qc, "_targets_row")
    print("✅ QualityCheckTab 有 _categories_row / _targets_row")

    vt = VideoTab()
    assert hasattr(vt, "_categories_row")
    print("✅ VideoTab 有 _categories_row（无 targets）")

    bm = BenchmarkTab()
    assert hasattr(bm, "_categories_row") and hasattr(bm, "_targets_row")
    print("✅ BenchmarkTab 有 _categories_row / _targets_row")


def test_task_switch_disables_correct_row():
    print("\n" + "=" * 60)
    print("测试 4: 任务切换时控件互斥禁用")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)

    from src.gui.widgets import PreAnnotateTab, QualityCheckTab, BenchmarkTab

    for cls in (PreAnnotateTab, QualityCheckTab, BenchmarkTab):
        tab = cls()
        tab._task_cb.setCurrentText("classification")
        assert tab._categories_row.isEnabled()
        assert not tab._targets_row.isEnabled()

        tab._task_cb.setCurrentText("detection")
        assert not tab._categories_row.isEnabled()
        assert tab._targets_row.isEnabled()

        print(f"✅ {cls.__name__}: classification/detection 切换时 row 互斥禁用正确")


def test_main_window_boots():
    print("\n" + "=" * 60)
    print("测试 5: 主窗口启动 + 广播引擎")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui import MainWindow
    window = MainWindow()

    class _FakeEngine:
        def supports(self, cap): return True

    window._engine_panel.primary_engine_changed.emit(_FakeEngine())
    app.processEvents()

    # 预标注和质检 Tab 的开始按钮应启用
    assert window._preannotate_tab._start_btn.isEnabled()
    assert window._qualitycheck_tab._start_btn.isEnabled()
    print("✅ 主窗口启动正常，引擎广播后 Tab 状态正确")

    window.close()


def main():
    test_parse_categories_txt()
    test_make_category_input()
    test_tabs_use_rows()
    test_task_switch_disables_correct_row()
    test_main_window_boots()

    print("\n" + "=" * 60)
    print("🎉 类别 txt 载入功能验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()

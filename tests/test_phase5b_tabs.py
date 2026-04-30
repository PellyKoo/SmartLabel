"""
Phase 5b 冒烟测试：预标注 Tab + 质检 Tab + 图片查看器 + 结果浏览器。

使用 offscreen 平台，不弹窗，验证：
1. 新 Tab 能构造、能接受 set_primary_engine/set_vlm_engine
2. ResultBrowser 能加载 items 并切换选择不崩
3. ImageViewer 清空 / 加载图片路径不存在时不崩
4. MainWindow 中的 5 个 Tab 类型正确（PreAnnotate/QualityCheck 为真实类）
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt, QCoreApplication, QTimer
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
from PyQt5.QtWidgets import QApplication

from src.gui import MainWindow
from src.gui.widgets import (
    EnginePanel, LogConsole, ImageViewer, ResultBrowser,
    PreAnnotateTab, QualityCheckTab, VideoTab, BenchmarkTab, SettingsTab,
)


def test_image_viewer():
    print("\n" + "=" * 60)
    print("测试 1: ImageViewer")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.clear()
    print("✅ clear() 无异常")

    # 加载不存在的图片
    viewer.load_image("/nonexistent/path.jpg")
    print("✅ 加载不存在的图片不崩（显示占位文本）")

    # 加载真实图片（若有）
    test_img = None
    data_dir = PROJECT_ROOT / "tests" / "data"
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".png"):
                test_img = str(f)
                break
    if test_img:
        viewer.load_image(test_img)
        print(f"✅ 加载真实图片 {os.path.basename(test_img)} 成功")

        # 带检测框
        viewer.load_image(test_img, detections=[
            {"label": "person", "bbox": [10, 10, 100, 100], "confidence": 0.95},
        ])
        print("✅ 绘制检测框不崩")
    else:
        print("⚠️  tests/data 无图片，跳过真实加载测试")


def test_result_browser():
    print("\n" + "=" * 60)
    print("测试 2: ResultBrowser")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    browser = ResultBrowser()
    browser.clear()

    # 构造假数据（image_path 不存在也应正常入列，缩略图失败降级）
    items = [
        {
            "image_path": "/nonexistent/img1.jpg",
            "title": "img1.jpg",
            "subtitle": "→ normal (0.95)",
            "meta": {"predicted_class": "normal", "confidence": 0.95},
        },
        {
            "image_path": "/nonexistent/img2.jpg",
            "title": "img2.jpg",
            "subtitle": "→ fatigue (0.72)",
            "meta": {"predicted_class": "fatigue", "confidence": 0.72},
        },
    ]
    browser.set_items(items)
    assert browser._list.count() == 2, "列表项数错误"
    print("✅ 设置 2 条 items 后列表显示正确")

    # 选择切换
    browser._list.setCurrentRow(1)
    app.processEvents()
    print("✅ 切换选中项不崩")

    browser.clear()
    assert browser._list.count() == 0
    print("✅ clear() 清空列表")


def test_preannotate_tab():
    print("\n" + "=" * 60)
    print("测试 3: PreAnnotateTab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    tab = PreAnnotateTab()

    # 未加载引擎时开始按钮应禁用
    assert not tab._start_btn.isEnabled(), "未加载引擎时开始按钮应禁用"
    print("✅ 未加载引擎时开始按钮禁用")

    # 模拟加载引擎（传 None 代表释放，传任意对象代表加载）
    class _FakeEngine:
        def supports(self, cap): return True
    tab.set_primary_engine(_FakeEngine())
    assert tab._start_btn.isEnabled(), "加载后开始按钮应启用"
    print("✅ 加载引擎后开始按钮启用")

    tab.set_primary_engine(None)
    assert not tab._start_btn.isEnabled()
    print("✅ 释放引擎后开始按钮禁用")

    # 切换任务类型
    tab._task_cb.setCurrentText("detection")
    assert tab._targets_edit.isEnabled(), "检测任务时目标输入应启用"
    assert not tab._categories_edit.isEnabled(), "检测任务时类别应禁用"
    print("✅ 切换到检测任务时控件互斥禁用正确")


def test_qualitycheck_tab():
    print("\n" + "=" * 60)
    print("测试 4: QualityCheckTab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    tab = QualityCheckTab()

    assert not tab._start_btn.isEnabled(), "未加载引擎时开始按钮应禁用"
    print("✅ 未加载主引擎时开始按钮禁用")

    class _FakeEngine:
        def supports(self, cap): return True

    tab.set_primary_engine(_FakeEngine())
    assert tab._start_btn.isEnabled()
    print("✅ 加载主引擎后开始按钮启用")

    tab.set_vlm_engine(_FakeEngine())
    assert "已加载" in tab._engine_status_lbl.text()
    print(f"✅ VLM 辅助引擎状态更新: {tab._engine_status_lbl.text()}")

    # 切到检测
    tab._task_cb.setCurrentText("detection")
    assert tab._iou_spin.isEnabled()
    assert not tab._categories_edit.isEnabled()
    print("✅ 检测任务时 IoU 输入启用、类别禁用")


def test_main_window_with_real_tabs():
    print("\n" + "=" * 60)
    print("测试 5: 主窗口使用真实 Tab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    assert window._tabs.count() == 5
    expected = [
        ("预标注", PreAnnotateTab),
        ("质检", QualityCheckTab),
        ("视频分类", VideoTab),
        ("评估", BenchmarkTab),
        ("设置", SettingsTab),
    ]
    for i, (title, cls) in enumerate(expected):
        assert window._tabs.tabText(i) == title
        w = window._tabs.widget(i)
        assert isinstance(w, cls), f"Tab {i} 类型错误: {type(w).__name__}"
        print(f"✅ Tab {i}: {title} = {cls.__name__}")

    # 广播引擎状态
    class _FakeEngine:
        def supports(self, cap): return True
    window._engine_panel.primary_engine_changed.emit(_FakeEngine())
    app.processEvents()
    # 预标注和质检 Tab 的开始按钮应都已启用
    assert window._preannotate_tab._start_btn.isEnabled()
    assert window._qualitycheck_tab._start_btn.isEnabled()
    print("✅ 广播主引擎后，预标注/质检 Tab 开始按钮都启用")

    window.close()


def main():
    test_image_viewer()
    test_result_browser()
    test_preannotate_tab()
    test_qualitycheck_tab()
    test_main_window_with_real_tabs()

    print("\n" + "=" * 60)
    print("🎉 Phase 5b 冒烟测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()

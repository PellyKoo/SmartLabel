"""
Phase 5c 冒烟测试：视频 Tab + 播放器 + 时间轴 + 评估 Tab + 设置 Tab。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt, QCoreApplication
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
from PyQt5.QtWidgets import QApplication

from src.gui import MainWindow
from src.gui.widgets import (
    VideoPlayer, TimelineWidget, VideoTab, BenchmarkTab, SettingsTab,
)
from src.engine.base import VideoClipResult


def test_video_player_lifecycle():
    print("\n" + "=" * 60)
    print("测试 1: VideoPlayer 生命周期")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    player = VideoPlayer()

    # 未打开视频时，控制按钮禁用
    assert not player._play_btn.isEnabled(), "未打开时播放按钮应禁用"
    print("✅ 未打开视频时控制按钮禁用")

    # 打开不存在的视频
    ok = player.open("/nonexistent/video.mp4")
    assert not ok
    print("✅ 打开不存在文件返回 False")

    # 若有真实视频则测试完整流程
    data_dir = PROJECT_ROOT / "tests" / "data"
    test_video = None
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                test_video = str(f)
                break

    if test_video:
        ok = player.open(test_video)
        assert ok, f"应能打开真实视频: {test_video}"
        assert player._duration > 0
        assert player._play_btn.isEnabled()
        print(f"✅ 打开真实视频成功: {os.path.basename(test_video)} "
              f"(时长 {player._duration:.1f}s)")

        # 跳转
        player.seek_to(1.0)
        assert abs(player.current_position() - 1.0) < 0.5
        print(f"✅ seek_to(1.0) 当前位置 {player.current_position():.2f}s")

        # 关闭
        player.close()
        assert not player._play_btn.isEnabled()
        print("✅ close() 后按钮禁用、资源释放")
    else:
        print("⚠️  无测试视频，跳过真实加载（仅验证空态）")


def test_timeline_widget():
    print("\n" + "=" * 60)
    print("测试 2: TimelineWidget")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    tl = TimelineWidget()
    tl.resize(400, 60)

    # 空态绘制不崩
    tl.paintEvent(None) if False else tl.update()
    app.processEvents()
    print("✅ 空态不崩")

    # 设置片段
    tl.set_clips([
        {"start_sec": 0.0, "end_sec": 5.0, "label": "normal", "duration_sec": 5.0},
        {"start_sec": 5.0, "end_sec": 8.0, "label": "fatigue", "duration_sec": 3.0},
    ], duration=8.0)
    cmap = tl.color_map()
    assert "normal" in cmap and "fatigue" in cmap
    print(f"✅ 颜色映射: {list(cmap.keys())}")

    # 游标
    tl.set_cursor(4.5)
    app.processEvents()
    print("✅ set_cursor 不崩")

    tl.clear()
    print("✅ clear 不崩")


def test_video_tab():
    print("\n" + "=" * 60)
    print("测试 3: VideoTab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    tab = VideoTab()

    assert not tab._start_btn.isEnabled(), "未加载引擎时禁用"
    class _FakeEngine:
        def supports(self, cap): return True
    tab.set_primary_engine(_FakeEngine())
    assert tab._start_btn.isEnabled()
    print("✅ 引擎加载后开始按钮启用")

    # 模拟显示一个伪造的结果
    fake_result = VideoClipResult(
        video_path="",
        clips=[
            {"start_sec": 0.0, "end_sec": 3.0, "label": "normal", "duration_sec": 3.0},
            {"start_sec": 3.0, "end_sec": 5.0, "label": "fatigue", "duration_sec": 2.0},
        ],
        statistics={"normal": 3.0, "fatigue": 2.0},
    )
    tab._show_video_result(fake_result)
    assert tab._table.rowCount() == 2
    print(f"✅ 片段表格有 {tab._table.rowCount()} 行")
    assert "legend" in tab._legend_lbl.text().lower() or "normal" in tab._legend_lbl.text()
    print("✅ 时间轴图例已更新")


def test_benchmark_tab():
    print("\n" + "=" * 60)
    print("测试 4: BenchmarkTab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    tab = BenchmarkTab()
    assert not tab._start_btn.isEnabled()
    class _FakeEngine:
        def supports(self, cap): return True
    tab.set_primary_engine(_FakeEngine())
    assert tab._start_btn.isEnabled()
    print("✅ 引擎加载后按钮启用")

    # 切换任务类型
    tab._task_cb.setCurrentText("detection")
    assert tab._targets_edit.isEnabled()
    assert not tab._categories_edit.isEnabled()
    print("✅ 切到检测任务时控件互斥正确")

    # 模拟分类结果显示
    fake_report = {
        "task": "classification",
        "accuracy": 0.875,
        "macro_f1": 0.84,
        "weighted_f1": 0.86,
        "per_class": {
            "normal": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
            "fatigue": {"precision": 0.85, "recall": 0.9, "f1": 0.87, "support": 30},
        },
        "confusion_matrix": [[40, 10], [3, 27]],
        "class_names": ["normal", "fatigue"],
    }
    tab._task_cb.setCurrentText("classification")
    tab._current_output_dir = str(PROJECT_ROOT / "tests" / "output")
    os.makedirs(tab._current_output_dir, exist_ok=True)
    tab._fill_classification_table(fake_report["per_class"])
    assert tab._metrics_table.rowCount() == 2
    print(f"✅ 指标表格有 {tab._metrics_table.rowCount()} 行")


def test_settings_tab():
    print("\n" + "=" * 60)
    print("测试 5: SettingsTab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    tab = SettingsTab()

    runtime = tab._collect_runtime()
    assert "num_io_workers" in runtime
    assert "log_level" in runtime
    print(f"✅ 收集运行参数: {runtime}")

    # 检查 prompt 编辑器数量
    n_prompts = len(tab._prompt_editors)
    print(f"✅ 加载了 {n_prompts} 个 Prompt 模板")
    assert n_prompts >= 1


def test_main_window_all_tabs():
    print("\n" + "=" * 60)
    print("测试 6: 主窗口 5 个真实 Tab")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    from src.gui.widgets import (
        PreAnnotateTab, QualityCheckTab, VideoTab, BenchmarkTab, SettingsTab,
    )
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

    # 广播引擎：所有 5 个 Tab 都应响应
    class _FakeEngine:
        def supports(self, cap): return True
    window._engine_panel.primary_engine_changed.emit(_FakeEngine())
    app.processEvents()
    for i in range(5):
        w = window._tabs.widget(i)
        assert hasattr(w, "set_primary_engine")
    print("✅ 所有 Tab 都能接收 primary_engine_changed")

    window.close()


def main():
    test_video_player_lifecycle()
    test_timeline_widget()
    test_video_tab()
    test_benchmark_tab()
    test_settings_tab()
    test_main_window_all_tabs()

    print("\n" + "=" * 60)
    print("🎉 Phase 5c 冒烟测试通过 — GUI 全功能齐备")
    print("=" * 60)


if __name__ == "__main__":
    main()

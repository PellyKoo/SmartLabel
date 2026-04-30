"""
Phase 3 视频分类功能验证脚本。

使用方法：
    1. 将测试视频放到 tests/data/ 目录下（任意 .mp4/.avi 文件）
    2. 在项目根目录运行：
        python tests/test_phase3_video.py
    3. 结果输出到 tests/output/ 目录

本脚本使用 MockEngine 模拟推理（无需真实 VLM 模型），
用于验证滑动窗口抽帧、时序平滑、片段检测、结果输出的完整流程。
"""
import os
import sys
import random

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 注意：直接从子模块导入，避免 src/*/__init__.py 级联拉入未用到的依赖（lxml 等）
from src.engine.base import BaseEngine, Capability, ClassificationResult
from src.io.video_io import scan_videos, VideoReader
from src.postprocess.temporal import temporal_smooth, detect_segments
from src.pipeline.video_classify import VideoClassifyPipeline


class MockEngine(BaseEngine):
    """
    模拟推理引擎。
    按固定规则返回标签，便于验证平滑和片段检测逻辑。

    规则：窗口索引 < 3 -> normal, 3 <= idx < 5 -> fatigue,
         idx == 5 -> normal（孤立噪声点，应被平滑掉）,
         idx >= 6 -> fatigue
    """

    def __init__(self):
        self._counter = 0
        self._seed = 42
        random.seed(self._seed)

    @property
    def capabilities(self) -> set:
        return {Capability.CLASSIFY, Capability.VIDEO_MULTIFRAME}

    def load(self):
        pass

    def unload(self):
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def get_engine_info(self) -> dict:
        return {"type": "mock", "model": "MockEngine v1"}

    def classify(self, image_path: str, categories: list) -> ClassificationResult:
        """按 counter 规则返回"""
        idx = self._counter
        self._counter += 1

        if idx < 3:
            label = "normal"
        elif idx < 5:
            label = "fatigue"
        elif idx == 5:
            label = "normal"  # 孤立噪声，应被平滑
        else:
            label = "fatigue"

        if label not in categories:
            label = categories[0]

        return ClassificationResult(
            image_path=image_path,
            predicted_class=label,
            confidence=0.9,
            is_uncertain=False,
            raw_output=f"mock:{label}",
        )

    def classify_video_frames(self, frame_paths: list, categories: list) -> ClassificationResult:
        """每个窗口返回一次结果（不是逐帧）"""
        idx = self._counter
        self._counter += 1

        if idx < 3:
            label = "normal"
        elif idx < 5:
            label = "fatigue"
        elif idx == 5:
            label = "normal"
        else:
            label = "fatigue"

        if label not in categories:
            label = categories[0]

        return ClassificationResult(
            image_path=frame_paths[0],
            predicted_class=label,
            confidence=0.9,
            is_uncertain=False,
            raw_output=f"mock_video:{label}",
        )


def test_temporal_functions():
    """单元测试：时序平滑 + 片段检测"""
    print("\n" + "=" * 60)
    print("测试 1: 时序平滑函数")
    print("=" * 60)

    labels = ["normal", "normal", "fatigue", "normal", "normal", "fatigue", "fatigue", "fatigue"]
    smoothed = temporal_smooth(labels, window=3)
    print(f"原始:  {labels}")
    print(f"平滑后: {smoothed}")
    assert smoothed[2] == "normal", f"第 2 位的孤立 fatigue 应被平滑为 normal，实际={smoothed[2]}"
    print("✅ 时序平滑：孤立噪声点被正确消除")

    print("\n" + "=" * 60)
    print("测试 2: 片段检测函数")
    print("=" * 60)

    labels = ["normal", "normal", "normal", "fatigue", "fatigue", "fatigue"]
    timestamps = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
    clips = detect_segments(labels, timestamps, min_duration_sec=2.0)

    print(f"输入标签:    {labels}")
    print(f"输入时间戳:  {timestamps}")
    print(f"检测到片段: {len(clips)} 个")
    for c in clips:
        print(f"  [{c['start_sec']}s - {c['end_sec']}s] {c['label']} (时长 {c['duration_sec']}s)")
    assert len(clips) == 2, f"期望 2 个片段，实际 {len(clips)}"
    print("✅ 片段检测：正确识别 2 个连续片段")


def test_video_reader(video_path: str):
    """测试视频读取 + 滑动窗口"""
    print("\n" + "=" * 60)
    print(f"测试 3: 视频读取 + 滑动窗口抽帧")
    print(f"视频: {video_path}")
    print("=" * 60)

    with VideoReader(video_path) as reader:
        meta = reader.get_metadata()
        print(f"元数据: {meta}")

        windows = reader.sliding_window(
            window_sec=5.0, stride_sec=2.5, sample_fps=1.0,
        )
        print(f"\n共抽取 {len(windows)} 个窗口")
        for w in windows[:3]:
            print(f"  窗口 {w['window_idx']}: "
                  f"[{w['start_sec']}s - {w['end_sec']}s], "
                  f"{len(w['frame_paths'])} 帧")
        if len(windows) > 3:
            print(f"  ... 还有 {len(windows) - 3} 个窗口")

        # 验证帧文件确实生成
        if windows:
            first_frame = windows[0]["frame_paths"][0]
            assert os.path.exists(first_frame), f"帧文件不存在: {first_frame}"
            size_kb = os.path.getsize(first_frame) / 1024
            print(f"\n✅ 帧文件验证: {first_frame} (大小 {size_kb:.1f} KB)")

        # 清理
        reader.cleanup()
        print("✅ 临时帧文件已清理")


def test_full_pipeline(video_path: str, output_dir: str):
    """端到端测试：完整视频分类流水线"""
    print("\n" + "=" * 60)
    print(f"测试 4: 完整视频分类流水线")
    print(f"视频: {video_path}")
    print(f"输出: {output_dir}")
    print("=" * 60)

    engine = MockEngine()
    config = {
        "categories": ["normal", "fatigue", "distracted"],
        "window_sec": 5.0,
        "stride_sec": 2.5,
        "sample_fps": 1.0,
        "strategy": "temporal_smooth",
        "smooth_window": 3,
        "min_segment_sec": 2.0,
        "output_format": "both",
        "ui_update_interval": 1,
    }

    pipeline = VideoClassifyPipeline(engine, config)

    def progress_cb(current, total, info):
        print(f"  [{current}/{total}] 窗口 {info.get('window_idx')}: "
              f"{info.get('start_sec')}s -> {info.get('label')}")

    result = pipeline.run(video_path, output_dir, progress_callback=progress_cb)

    print(f"\n分类结果：")
    print(f"  视频: {result.video_path}")
    print(f"  片段数: {len(result.clips)}")
    for c in result.clips:
        print(f"    [{c['start_sec']}s - {c['end_sec']}s] {c['label']} (时长 {c['duration_sec']}s)")
    print(f"  各类时长统计: {result.statistics}")

    # 验证输出文件
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{video_name}_clips.csv")
    json_path = os.path.join(output_dir, f"{video_name}_result.json")

    assert os.path.exists(csv_path), f"CSV 未生成: {csv_path}"
    assert os.path.exists(json_path), f"JSON 未生成: {json_path}"

    print(f"\n✅ CSV 输出: {csv_path}")
    print(f"✅ JSON 输出: {json_path}")


def main():
    # 先跑不依赖视频的单元测试
    test_temporal_functions()

    # 查找测试视频
    data_dir = os.path.join(PROJECT_ROOT, "tests", "data")
    output_dir = os.path.join(PROJECT_ROOT, "tests", "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    videos = scan_videos(data_dir) if os.path.isdir(data_dir) else []

    if not videos:
        print("\n" + "=" * 60)
        print("⚠️  tests/data/ 目录下没有视频文件")
        print("=" * 60)
        print("\n请放置一个视频文件（.mp4/.avi/.mkv 等）到：")
        print(f"  {data_dir}")
        print("\n然后重新运行本脚本。")
        print("\n✅ 单元测试（时序函数）已通过，可独立验证核心逻辑。")
        return

    video_path = videos[0]
    print(f"\n找到测试视频: {video_path}")

    test_video_reader(video_path)
    test_full_pipeline(video_path, output_dir)

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()

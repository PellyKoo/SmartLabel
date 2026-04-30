import os
from typing import Optional, Callable

from src.engine.base import BaseEngine, Capability, VideoClipResult
from src.io.video_io import (
    VideoReader, scan_videos,
    save_video_clips_csv, save_video_clips_json,
)
from src.postprocess.temporal import temporal_smooth, detect_segments, compute_segment_statistics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoClassifyPipeline:
    """
    视频片段分类流水线。

    完整流程：
    1. 滑动窗口抽帧
    2. 逐窗口推理（支持多种策略）
    3. 时序后处理（平滑 + 片段检测）
    4. 输出结果（CSV + JSON）
    """

    def __init__(self, engine: BaseEngine, config: dict):
        """
        Args:
            engine: 推理引擎实例（已加载）
            config: 任务配置
                categories: list[str]          # 分类类别
                window_sec: float              # 窗口时长（默认 5.0）
                stride_sec: float              # 滑动步长（默认 2.5）
                sample_fps: float              # 窗口内采样帧率（默认 1.0）
                strategy: str                  # "vote" | "temporal_smooth" | "vlm_multiframe"
                smooth_window: int             # 时序平滑窗口（默认 3）
                min_segment_sec: float         # 最短片段时长（默认 2.0）
                output_format: str             # "csv" | "json" | "both"
                temp_dir: str                  # 临时文件目录
                ui_update_interval: int        # 进度回调间隔
        """
        self.engine = engine
        self.config = config

    def run(self, video_path: str, output_dir: str,
            progress_callback: Optional[Callable] = None) -> VideoClipResult:
        """
        对单个视频执行片段分类。

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            progress_callback: fn(current, total, latest_result)

        Returns:
            VideoClipResult
        """
        categories = self.config.get("categories", [])
        if not categories:
            raise ValueError("必须提供分类类别列表 (categories)")

        strategy = self.config.get("strategy", "temporal_smooth")
        temp_dir = self.config.get("temp_dir", None)

        logger.info(
            f"开始视频分类: {video_path}, 策略={strategy}, "
            f"类别={categories}"
        )

        # Step 1: 打开视频 + 滑动窗口抽帧
        reader = VideoReader(video_path, temp_dir=temp_dir)
        try:
            windows = reader.sliding_window(
                window_sec=self.config.get("window_sec", 5.0),
                stride_sec=self.config.get("stride_sec", 2.5),
                sample_fps=self.config.get("sample_fps", 1.0),
            )

            if not windows:
                logger.warning(f"视频抽帧结果为空: {video_path}")
                return VideoClipResult(
                    video_path=video_path, clips=[], statistics={}
                )

            # Step 2: 逐窗口推理
            window_labels = []
            window_timestamps = []
            use_vlm_multiframe = (
                strategy == "vlm_multiframe"
                and self.engine.supports(Capability.VIDEO_MULTIFRAME)
            )

            for i, win in enumerate(windows):
                try:
                    if use_vlm_multiframe:
                        result = self.engine.classify_video_frames(
                            win["frame_paths"], categories
                        )
                    else:
                        # 默认使用引擎基类的逐帧投票
                        result = self.engine.classify_video_frames(
                            win["frame_paths"], categories
                        )

                    window_labels.append(result.predicted_class)

                except Exception as e:
                    logger.error(
                        f"窗口 {i} 推理失败 "
                        f"({win['start_sec']:.1f}s-{win['end_sec']:.1f}s): {e}"
                    )
                    window_labels.append("error")

                window_timestamps.append(win["start_sec"])

                if progress_callback:
                    interval = self.config.get("ui_update_interval", 1)
                    if (i + 1) % interval == 0 or (i + 1) == len(windows):
                        progress_callback(i + 1, len(windows), {
                            "window_idx": i,
                            "start_sec": win["start_sec"],
                            "label": window_labels[-1],
                        })

            # Step 3: 时序后处理
            if strategy in ("temporal_smooth", "vlm_multiframe"):
                smoothed = temporal_smooth(
                    window_labels,
                    window=self.config.get("smooth_window", 3),
                )
                clips = detect_segments(
                    smoothed, window_timestamps,
                    min_duration_sec=self.config.get("min_segment_sec", 2.0),
                )
            else:
                # "vote" 策略：不做平滑，直接合并
                clips = detect_segments(
                    window_labels, window_timestamps,
                    min_duration_sec=self.config.get("min_segment_sec", 0),
                )

            # 计算统计
            stats = compute_segment_statistics(clips)

            video_result = VideoClipResult(
                video_path=video_path,
                clips=clips,
                statistics=stats,
            )

            # Step 4: 保存结果
            self._save_results(video_result, output_dir)

            logger.info(
                f"视频分类完成: {video_path}, "
                f"{len(clips)} 个片段, 统计={stats}"
            )
            return video_result

        finally:
            # 清理临时帧文件
            reader.cleanup()
            reader.release()

    def run_batch(self, video_dir: str, output_dir: str,
                   progress_callback: Optional[Callable] = None) -> list[VideoClipResult]:
        """
        批量处理视频目录。

        Args:
            video_dir: 视频目录或单个视频路径
            output_dir: 输出目录
            progress_callback: fn(current, total, latest_result)

        Returns:
            VideoClipResult 列表
        """
        if os.path.isfile(video_dir):
            result = self.run(video_dir, output_dir, progress_callback)
            return [result]

        videos = scan_videos(video_dir)
        if not videos:
            logger.warning(f"未找到视频: {video_dir}")
            return []

        logger.info(f"开始批量视频分类: {len(videos)} 个视频")

        results = []
        for vi, video_path in enumerate(videos):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, video_name)

            def _video_progress(current, total, info):
                if progress_callback:
                    progress_callback(
                        vi * 100 + int(current / total * 100),
                        len(videos) * 100,
                        {
                            "video_idx": vi,
                            "video_path": video_path,
                            **(info if isinstance(info, dict) else {}),
                        },
                    )

            try:
                result = self.run(video_path, video_output_dir, _video_progress)
                results.append(result)
            except Exception as e:
                logger.error(f"视频处理失败: {video_path}, 错误: {e}")
                results.append(VideoClipResult(
                    video_path=video_path, clips=[], statistics={}
                ))

        logger.info(f"批量视频分类完成: {len(results)}/{len(videos)} 个成功")
        return results

    def _save_results(self, video_result: VideoClipResult, output_dir: str):
        """保存视频分类结果"""
        os.makedirs(output_dir, exist_ok=True)
        output_format = self.config.get("output_format", "both")

        video_name = os.path.splitext(
            os.path.basename(video_result.video_path)
        )[0]

        if output_format in ("csv", "both"):
            csv_path = os.path.join(output_dir, f"{video_name}_clips.csv")
            save_video_clips_csv(video_result.clips, csv_path)

        if output_format in ("json", "both"):
            json_path = os.path.join(output_dir, f"{video_name}_result.json")
            save_video_clips_json(video_result, json_path)

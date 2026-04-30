import os
import math
import shutil
from typing import Optional, Callable

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 支持的视频扩展名
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


def scan_videos(video_dir: str, recursive: bool = False) -> list[str]:
    """
    扫描目录中的所有视频文件，按文件名排序。

    Args:
        video_dir: 视频目录路径
        recursive: 是否递归扫描子目录

    Returns:
        视频文件绝对路径列表
    """
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    videos = []
    if recursive:
        for root, _, files in os.walk(video_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                    videos.append(os.path.join(root, f))
    else:
        for f in os.listdir(video_dir):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(video_dir, f))

    videos.sort()
    logger.info(f"扫描到 {len(videos)} 个视频: {video_dir}")
    return videos


class VideoReader:
    """
    视频读取器，支持滑动窗口抽帧。

    使用 OpenCV VideoCapture 读取视频，按时间窗口提取帧并保存为临时图片文件。
    """

    def __init__(self, video_path: str, temp_dir: str = None):
        """
        Args:
            video_path: 视频文件路径
            temp_dir: 临时帧文件目录，默认在视频同目录下创建
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0

        # 临时目录
        if temp_dir is None:
            temp_dir = os.path.join(
                os.path.dirname(video_path), ".smartlabel_temp"
            )
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        self._temp_dir = os.path.join(temp_dir, video_stem)
        os.makedirs(self._temp_dir, exist_ok=True)

        logger.info(
            f"视频已打开: {video_path}, "
            f"FPS={self._fps:.1f}, 时长={self._duration:.1f}s, "
            f"分辨率={self._width}x{self._height}, 总帧数={self._frame_count}"
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_metadata(self) -> dict:
        """获取视频元数据"""
        return {
            "video_path": self.video_path,
            "fps": self._fps,
            "duration_sec": round(self._duration, 2),
            "frame_count": self._frame_count,
            "width": self._width,
            "height": self._height,
        }

    def extract_frame_at_time(self, timestamp_sec: float) -> Optional[np.ndarray]:
        """
        提取指定时间点的帧。

        Args:
            timestamp_sec: 时间戳（秒）

        Returns:
            BGR 格式的 numpy 数组，失败返回 None
        """
        frame_idx = int(timestamp_sec * self._fps)
        if frame_idx < 0 or frame_idx >= self._frame_count:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        return frame if ret else None

    def extract_frame_to_file(self, timestamp_sec: float,
                               output_path: str = None) -> Optional[str]:
        """
        提取帧并保存为图片文件。

        Args:
            timestamp_sec: 时间戳（秒）
            output_path: 输出路径，默认保存到临时目录

        Returns:
            保存的文件路径，失败返回 None
        """
        frame = self.extract_frame_at_time(timestamp_sec)
        if frame is None:
            return None

        if output_path is None:
            output_path = os.path.join(
                self._temp_dir, f"frame_{timestamp_sec:.3f}s.jpg"
            )

        cv2.imwrite(output_path, frame)
        return output_path

    def sliding_window(self, window_sec: float = 5.0,
                        stride_sec: float = 2.5,
                        sample_fps: float = 1.0) -> list[dict]:
        """
        滑动窗口抽帧。

        将视频按 stride_sec 步长划分窗口，每个窗口内按 sample_fps 均匀抽帧，
        帧保存为临时图片文件。

        Args:
            window_sec: 窗口时长（秒）
            stride_sec: 窗口滑动步长（秒）
            sample_fps: 窗口内采样帧率（帧/秒）

        Returns:
            窗口列表：[
                {
                    "window_idx": int,
                    "start_sec": float,
                    "end_sec": float,
                    "frame_paths": list[str],
                    "timestamps": list[float],
                },
                ...
            ]
        """
        if self._duration <= 0:
            logger.warning(f"视频时长为0: {self.video_path}")
            return []

        windows = []
        start = 0.0
        window_idx = 0

        while start < self._duration:
            end = min(start + window_sec, self._duration)

            # 计算窗口内的采样时间点
            num_frames = max(1, int((end - start) * sample_fps))
            if num_frames == 1:
                timestamps = [start + (end - start) / 2]
            else:
                step = (end - start) / num_frames
                timestamps = [start + step * i + step / 2 for i in range(num_frames)]

            # 提取并保存帧
            frame_paths = []
            valid_timestamps = []
            for ts in timestamps:
                if ts >= self._duration:
                    continue
                path = self.extract_frame_to_file(ts)
                if path is not None:
                    frame_paths.append(path)
                    valid_timestamps.append(round(ts, 3))

            if frame_paths:
                windows.append({
                    "window_idx": window_idx,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "frame_paths": frame_paths,
                    "timestamps": valid_timestamps,
                })
                window_idx += 1

            start += stride_sec

        logger.info(
            f"滑动窗口抽帧完成: {len(windows)} 个窗口, "
            f"窗口={window_sec}s, 步长={stride_sec}s, 采样FPS={sample_fps}"
        )
        return windows

    def cleanup(self):
        """清理临时帧文件"""
        if os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            logger.info(f"已清理临时文件: {self._temp_dir}")

    def release(self):
        """释放视频资源"""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def save_video_clips_csv(clips: list[dict], output_path: str):
    """
    保存视频片段结果为 CSV。

    Args:
        clips: 片段列表 [{"start_sec", "end_sec", "label", "duration_sec"}, ...]
        output_path: CSV 输出路径
    """
    import csv

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "label", "duration_sec"])
        for clip in clips:
            writer.writerow([
                clip["start_sec"],
                clip["end_sec"],
                clip["label"],
                clip.get("duration_sec", round(clip["end_sec"] - clip["start_sec"], 2)),
            ])

    logger.info(f"视频片段 CSV 已保存: {output_path}, 共 {len(clips)} 条")


def save_video_clips_json(video_result, output_path: str):
    """
    保存视频分类结果为 JSON。

    Args:
        video_result: VideoClipResult 实例
        output_path: JSON 输出路径
    """
    import json

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = {
        "video_path": video_result.video_path,
        "clips": video_result.clips,
        "statistics": video_result.statistics,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"视频分类 JSON 已保存: {output_path}")

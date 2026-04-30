"""
OpenCV 手动解码 + QLabel 贴图的视频播放器。

跨平台稳定（不依赖系统解码器），支持：
- 打开 / 关闭视频
- 播放 / 暂停 / 单帧步进
- 跳转到任意秒
- 位置变化信号（供时间轴同步高亮）
- 播放结束信号
"""
import os
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QSizePolicy,
)


def _cv_bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """OpenCV BGR ndarray → QPixmap"""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # QImage 需要连续内存且 stride 对齐
    rgb = np.ascontiguousarray(rgb)
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())  # copy 避免底层 buffer 被回收


def _fmt_time(sec: float) -> str:
    """秒 → mm:ss.s"""
    if sec < 0 or not np.isfinite(sec):
        return "00:00.0"
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


class VideoPlayer(QWidget):
    """
    视频播放器。

    Signals:
        position_changed(float sec)   # 当前播放位置变更
        finished()                    # 播放到结尾
        opened(float duration_sec)    # 视频打开成功
    """

    position_changed = pyqtSignal(float)
    finished = pyqtSignal()
    opened = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._duration: float = 0.0
        self._current_frame: int = 0
        self._playing: bool = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._update_controls_enabled(False)

    # -------- public API --------

    def open(self, video_path: str) -> bool:
        """打开视频文件，成功返回 True"""
        self.close()
        if not os.path.isfile(video_path):
            self._frame_lbl.setText(f"视频不存在: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._frame_lbl.setText(f"无法打开视频: {video_path}")
            return False

        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps if self._fps > 0 else 0.0
        self._current_frame = 0

        self._slider.setMaximum(max(self._total_frames - 1, 0))
        self._slider.setValue(0)
        self._total_time_lbl.setText(_fmt_time(self._duration))
        self._update_controls_enabled(True)
        self.opened.emit(self._duration)
        self._show_frame_at(0)
        return True

    def close(self):
        """释放视频资源"""
        self._timer.stop()
        self._playing = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._fps = 0.0
        self._total_frames = 0
        self._duration = 0.0
        self._current_frame = 0
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._frame_lbl.setPixmap(QPixmap())
        self._frame_lbl.setText("未打开视频")
        self._update_controls_enabled(False)

    def seek_to(self, timestamp_sec: float):
        """跳转到指定秒"""
        if self._cap is None or self._fps <= 0:
            return
        frame_idx = int(timestamp_sec * self._fps)
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self._show_frame_at(frame_idx)

    def play(self):
        if self._cap is None or self._playing:
            return
        # 若在末尾点击播放，回到开头
        if self._current_frame >= self._total_frames - 1:
            self._show_frame_at(0)

        self._playing = True
        self._play_btn.setText("⏸ 暂停")
        interval_ms = max(1, int(1000.0 / max(self._fps, 1.0)))
        self._timer.start(interval_ms)

    def pause(self):
        if not self._playing:
            return
        self._playing = False
        self._play_btn.setText("▶ 播放")
        self._timer.stop()

    def is_playing(self) -> bool:
        return self._playing

    def current_position(self) -> float:
        if self._fps <= 0:
            return 0.0
        return self._current_frame / self._fps

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._frame_lbl = QLabel("未打开视频")
        self._frame_lbl.setAlignment(Qt.AlignCenter)
        self._frame_lbl.setMinimumSize(320, 240)
        self._frame_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame_lbl.setStyleSheet(
            "background: #000; color: #666; border: 1px solid #3C3C3C;"
        )
        layout.addWidget(self._frame_lbl, stretch=1)

        # 进度条 + 时间显示
        slider_row = QHBoxLayout()
        self._current_time_lbl = QLabel("00:00.0")
        self._current_time_lbl.setFixedWidth(60)
        self._total_time_lbl = QLabel("00:00.0")
        self._total_time_lbl.setFixedWidth(60)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._was_playing_before_seek = False

        slider_row.addWidget(self._current_time_lbl)
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._total_time_lbl)
        layout.addLayout(slider_row)

        # 控制按钮
        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("▶ 播放")
        self._play_btn.clicked.connect(self._toggle_play)
        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.clicked.connect(self._on_stop)
        self._prev_frame_btn = QPushButton("⏮ -1f")
        self._prev_frame_btn.clicked.connect(lambda: self._step_frame(-1))
        self._next_frame_btn = QPushButton("+1f ⏭")
        self._next_frame_btn.clicked.connect(lambda: self._step_frame(+1))

        btn_row.addWidget(self._prev_frame_btn)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(self._next_frame_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _update_controls_enabled(self, enabled: bool):
        for w in (self._slider, self._play_btn, self._stop_btn,
                   self._prev_frame_btn, self._next_frame_btn):
            w.setEnabled(enabled)

    # -------- internal --------

    def _show_frame_at(self, frame_idx: int):
        """设置当前帧（绝对索引）并显示"""
        if self._cap is None:
            return
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return

        self._current_frame = frame_idx
        pix = _cv_bgr_to_qpixmap(frame)

        # 等比缩放到 label
        target = self._frame_lbl.size()
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._frame_lbl.setPixmap(scaled)

        pos = self.current_position()
        self._current_time_lbl.setText(_fmt_time(pos))
        if not self._slider.isSliderDown():
            self._slider.blockSignals(True)
            self._slider.setValue(frame_idx)
            self._slider.blockSignals(False)
        self.position_changed.emit(pos)

    @pyqtSlot()
    def _on_tick(self):
        """播放计时器 tick，读下一帧"""
        if self._cap is None:
            return
        if self._current_frame >= self._total_frames - 1:
            self.pause()
            self.finished.emit()
            return
        self._show_frame_at(self._current_frame + 1)

    def _toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def _on_stop(self):
        self.pause()
        self._show_frame_at(0)

    def _step_frame(self, delta: int):
        if self._playing:
            self.pause()
        self._show_frame_at(self._current_frame + delta)

    def _on_slider_pressed(self):
        self._was_playing_before_seek = self._playing
        if self._playing:
            self.pause()

    def _on_slider_moved(self, val: int):
        # 轻量：只改当前时间标签，松开时才真正跳转
        if self._fps > 0:
            self._current_time_lbl.setText(_fmt_time(val / self._fps))

    def _on_slider_released(self):
        self._show_frame_at(self._slider.value())
        if self._was_playing_before_seek:
            self.play()
        self._was_playing_before_seek = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 重绘当前帧以适应新尺寸
        if self._cap is not None and self._current_frame < self._total_frames:
            self._show_frame_at(self._current_frame)

    def closeEvent(self, event):
        self.close()
        super().closeEvent(event)

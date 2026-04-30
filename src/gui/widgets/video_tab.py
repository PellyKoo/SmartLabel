"""
视频片段分类 Tab。

左侧：输入配置 + 控制 + 片段列表
右侧：视频播放器 + 时间轴（与播放位置同步）
"""
import os
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QFontMetrics
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QComboBox, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QSpinBox, QDoubleSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from src.engine.base import VideoClipResult
from src.gui.threads.worker import UnifiedWorker
from src.gui.widgets._category_input import make_category_input
from src.gui.widgets.video_player import VideoPlayer
from src.utils.logger import get_logger

logger = get_logger(__name__)


_CLIP_COLORS = [
    "#4CAF50", "#FF9800", "#EF5350", "#2196F3", "#9C27B0",
    "#00BCD4", "#FFEB3B", "#795548", "#E91E63", "#3F51B5",
]


class TimelineWidget(QWidget):
    """
    彩色片段时间轴。

    Signals:
        seek_requested(float sec)   # 用户点击时间轴，请求跳转
    """

    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration: float = 0.0
        self._clips: list[dict] = []
        self._color_map: dict[str, QColor] = {}
        self._cursor_sec: float = 0.0
        self.setMinimumHeight(50)
        self.setMouseTracking(False)

    def set_clips(self, clips: list[dict], duration: float):
        self._clips = clips or []
        self._duration = max(0.01, duration)
        labels = sorted({c["label"] for c in self._clips})
        self._color_map = {
            l: QColor(_CLIP_COLORS[i % len(_CLIP_COLORS)])
            for i, l in enumerate(labels)
        }
        self.update()

    def set_cursor(self, sec: float):
        self._cursor_sec = max(0.0, min(sec, self._duration))
        self.update()

    def clear(self):
        self._clips = []
        self._color_map = {}
        self._cursor_sec = 0.0
        self._duration = 0.0
        self.update()

    def color_map(self) -> dict[str, QColor]:
        return dict(self._color_map)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_y = h // 3
        bar_h = h // 3

        # 背景
        painter.fillRect(0, bar_y, w, bar_h, QColor("#252526"))
        painter.setPen(QPen(QColor("#3C3C3C"), 1))
        painter.drawRect(0, bar_y, w - 1, bar_h - 1)

        if self._duration <= 0 or not self._clips:
            painter.setPen(QColor("#666"))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "暂无片段数据")
            return

        # 片段
        for clip in self._clips:
            x = clip["start_sec"] / self._duration * w
            cw = max(1.0, (clip["end_sec"] - clip["start_sec"]) / self._duration * w)
            color = self._color_map.get(clip["label"], QColor("#888"))
            painter.fillRect(QRectF(x, bar_y, cw, bar_h), color)

        # 游标（当前播放位置）
        cx = self._cursor_sec / self._duration * w
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawLine(int(cx), bar_y - 4, int(cx), bar_y + bar_h + 4)

        # 时间刻度
        painter.setPen(QColor("#888"))
        painter.drawText(QRectF(0, bar_y + bar_h + 4, 60, 16),
                          Qt.AlignLeft, "0.0s")
        painter.drawText(QRectF(w - 60, bar_y + bar_h + 4, 60, 16),
                          Qt.AlignRight, f"{self._duration:.1f}s")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._duration > 0:
            sec = event.x() / self.width() * self._duration
            self.seek_requested.emit(sec)


class VideoTab(QWidget):
    """视频片段分类 Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._primary_engine = None
        self._worker: Optional[UnifiedWorker] = None
        self._last_result: Optional[VideoClipResult] = None
        self._current_video_path: Optional[str] = None
        self._build_ui()

    # -------- public --------

    def set_primary_engine(self, engine):
        self._primary_engine = engine
        ok = engine is not None
        self._start_btn.setEnabled(ok)
        self._engine_status_lbl.setText("引擎: 已加载" if ok else "引擎: 未加载")
        self._engine_status_lbl.setStyleSheet(
            "color: #4CAF50;" if ok else "color: #888;"
        )

    def set_vlm_engine(self, engine):
        """本 Tab 不使用 VLM 辅助"""

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # 左：配置 + 片段列表
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.addWidget(self._build_config_group())
        left_v.addLayout(self._build_control_row())
        left_v.addWidget(self._build_clip_table(), stretch=1)
        splitter.addWidget(left)

        # 右：播放器 + 时间轴
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)

        self._player = VideoPlayer()
        self._player.position_changed.connect(self._on_player_position)
        right_v.addWidget(self._player, stretch=1)

        timeline_gb = QGroupBox("时间轴")
        timeline_v = QVBoxLayout(timeline_gb)
        self._timeline = TimelineWidget()
        self._timeline.seek_requested.connect(self._on_timeline_seek)
        timeline_v.addWidget(self._timeline)
        self._legend_lbl = QLabel("")
        self._legend_lbl.setWordWrap(True)
        self._legend_lbl.setTextFormat(Qt.RichText)
        timeline_v.addWidget(self._legend_lbl)
        right_v.addWidget(timeline_gb)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([480, 720])
        layout.addWidget(splitter, stretch=1)

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("配置")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._video_edit, v_row = self._file_or_dir_picker(
            "选择视频文件或目录"
        )
        form.addRow("视频路径:", v_row)

        self._output_dir_edit, out_row = self._path_picker(
            "选择输出目录", is_dir=True
        )
        form.addRow("输出目录:", out_row)

        self._categories_edit, self._categories_row = make_category_input(
            placeholder="从 txt 载入类别，或手动逗号分隔",
            file_dialog_caption="选择视频类别 txt（每行一个）",
            parent=self,
        )
        form.addRow("类别:", self._categories_row)

        self._strategy_cb = QComboBox()
        self._strategy_cb.addItems(["temporal_smooth", "vote", "vlm_multiframe"])
        form.addRow("策略:", self._strategy_cb)

        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(0.5, 60.0)
        self._window_spin.setSingleStep(0.5)
        self._window_spin.setValue(5.0)
        self._window_spin.setSuffix(" s")
        form.addRow("窗口时长:", self._window_spin)

        self._stride_spin = QDoubleSpinBox()
        self._stride_spin.setRange(0.1, 60.0)
        self._stride_spin.setSingleStep(0.5)
        self._stride_spin.setValue(2.5)
        self._stride_spin.setSuffix(" s")
        form.addRow("滑动步长:", self._stride_spin)

        self._sample_fps_spin = QDoubleSpinBox()
        self._sample_fps_spin.setRange(0.1, 30.0)
        self._sample_fps_spin.setSingleStep(0.5)
        self._sample_fps_spin.setValue(1.0)
        form.addRow("采样 FPS:", self._sample_fps_spin)

        self._smooth_window_spin = QSpinBox()
        self._smooth_window_spin.setRange(1, 15)
        self._smooth_window_spin.setValue(3)
        form.addRow("平滑窗口:", self._smooth_window_spin)

        self._min_seg_spin = QDoubleSpinBox()
        self._min_seg_spin.setRange(0.0, 30.0)
        self._min_seg_spin.setSingleStep(0.5)
        self._min_seg_spin.setValue(2.0)
        self._min_seg_spin.setSuffix(" s")
        form.addRow("最短片段:", self._min_seg_spin)

        return gb

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始分类")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)

        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)

        self._progress = QProgressBar()
        self._progress.setFormat("%v / %m")
        row.addWidget(self._progress, stretch=1)

        self._engine_status_lbl = QLabel("引擎: 未加载")
        self._engine_status_lbl.setStyleSheet("color: #888;")
        row.addWidget(self._engine_status_lbl)

        return row

    def _build_clip_table(self) -> QGroupBox:
        gb = QGroupBox("片段列表")
        v = QVBoxLayout(gb)

        self._summary_lbl = QLabel("未开始")
        self._summary_lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
        v.addWidget(self._summary_lbl)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["开始(s)", "结束(s)", "标签", "时长(s)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.cellClicked.connect(self._on_clip_clicked)
        v.addWidget(self._table)

        return gb

    def _path_picker(self, caption: str, is_dir: bool = True):
        edit = QLineEdit()
        btn = QPushButton("...")
        btn.setFixedWidth(32)

        def _browse():
            if is_dir:
                path = QFileDialog.getExistingDirectory(self, caption, edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, caption, edit.text())
            if path:
                edit.setText(path)

        btn.clicked.connect(_browse)
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        row.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setLayout(row)
        return edit, container

    def _file_or_dir_picker(self, caption: str):
        """视频路径可以是文件或目录"""
        edit = QLineEdit()
        file_btn = QPushButton("文件")
        dir_btn = QPushButton("目录")
        file_btn.setFixedWidth(50)
        dir_btn.setFixedWidth(50)

        def _pick_file():
            path, _ = QFileDialog.getOpenFileName(
                self, caption, edit.text(),
                "视频文件 (*.mp4 *.avi *.mkv *.mov *.webm);;所有文件 (*.*)"
            )
            if path:
                edit.setText(path)
                self._try_open_video(path)

        def _pick_dir():
            path = QFileDialog.getExistingDirectory(self, caption, edit.text())
            if path:
                edit.setText(path)

        file_btn.clicked.connect(_pick_file)
        dir_btn.clicked.connect(_pick_dir)
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(file_btn)
        row.addWidget(dir_btn)
        row.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setLayout(row)
        return edit, container

    # -------- actions --------

    def _try_open_video(self, path: str):
        """选择到文件时自动预览"""
        if os.path.isfile(path):
            ok = self._player.open(path)
            if ok:
                self._current_video_path = path
                self._timeline.clear()
                self._legend_lbl.setText("")

    def _validate(self) -> Optional[dict]:
        if self._primary_engine is None:
            QMessageBox.warning(self, "提示", "请先加载主引擎")
            return None
        video_path = self._video_edit.text().strip()
        output_dir = self._output_dir_edit.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "提示", "视频路径无效")
            return None
        if not output_dir:
            QMessageBox.warning(self, "提示", "请指定输出目录")
            return None
        cats = [c.strip() for c in self._categories_edit.text().split(",") if c.strip()]
        if not cats:
            QMessageBox.warning(self, "提示", "请输入类别")
            return None

        return {
            "video_path": video_path,
            "output_dir": output_dir,
            "categories": cats,
            "strategy": self._strategy_cb.currentText(),
            "window_sec": self._window_spin.value(),
            "stride_sec": self._stride_spin.value(),
            "sample_fps": self._sample_fps_spin.value(),
            "smooth_window": self._smooth_window_spin.value(),
            "min_segment_sec": self._min_seg_spin.value(),
        }

    def _on_start(self):
        args = self._validate()
        if args is None:
            return

        from src.pipeline.video_classify import VideoClassifyPipeline

        pipeline_cfg = {
            "categories": args["categories"],
            "window_sec": args["window_sec"],
            "stride_sec": args["stride_sec"],
            "sample_fps": args["sample_fps"],
            "strategy": args["strategy"],
            "smooth_window": args["smooth_window"],
            "min_segment_sec": args["min_segment_sec"],
            "output_format": "both",
            "ui_update_interval": 1,
        }
        pipeline = VideoClassifyPipeline(self._primary_engine, pipeline_cfg)

        if os.path.isfile(args["video_path"]):
            def task_fn(cb):
                return pipeline.run(args["video_path"], args["output_dir"], cb)
        else:
            def task_fn(cb):
                return pipeline.run_batch(args["video_path"], args["output_dir"], cb)

        self._worker = UnifiedWorker(task_fn, ui_update_interval=1)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._set_running(True)
        self._summary_lbl.setText("运行中…")
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self._summary_lbl.setText("停止中…")

    def _on_progress(self, current: int, total: int):
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(current)

    def _on_finished(self, result):
        self._set_running(False)

        # run() 返回 VideoClipResult，run_batch() 返回 list
        if isinstance(result, list):
            if not result:
                self._summary_lbl.setText("完成: 无结果")
                return
            # 只展示第一个
            first = result[0]
            self._summary_lbl.setText(
                f"批量完成: {len(result)} 个视频 | 展示第一个: "
                f"{os.path.basename(first.video_path)} - "
                f"{len(first.clips)} 个片段"
            )
            self._show_video_result(first)
        else:
            self._last_result = result
            self._summary_lbl.setText(
                f"完成: {len(result.clips)} 个片段 | 统计: {result.statistics}"
            )
            self._show_video_result(result)

    def _on_error(self, msg: str):
        self._set_running(False)
        self._summary_lbl.setText(f"失败: {msg}")
        QMessageBox.critical(self, "视频分类失败", msg)

    def _set_running(self, running: bool):
        self._start_btn.setEnabled(not running and self._primary_engine is not None)
        self._stop_btn.setEnabled(running)
        for w in (self._video_edit, self._output_dir_edit, self._categories_row,
                   self._strategy_cb, self._window_spin, self._stride_spin,
                   self._sample_fps_spin, self._smooth_window_spin, self._min_seg_spin):
            w.setEnabled(not running)

    def _show_video_result(self, result: VideoClipResult):
        self._last_result = result
        # 打开视频（若尚未打开或路径不同）
        if result.video_path and result.video_path != self._current_video_path:
            if self._player.open(result.video_path):
                self._current_video_path = result.video_path

        # 时间轴
        duration = max((c["end_sec"] for c in result.clips), default=0.0)
        if duration <= 0 and self._player._duration > 0:
            duration = self._player._duration
        self._timeline.set_clips(result.clips, duration)

        # 图例
        cmap = self._timeline.color_map()
        total = sum(result.statistics.values()) if result.statistics else 0
        legend_parts = []
        for lbl, color in cmap.items():
            dur = result.statistics.get(lbl, 0)
            pct = dur / total * 100 if total > 0 else 0
            legend_parts.append(
                f'<span style="color:{color.name()}">■</span> '
                f'{lbl}: {dur:.1f}s ({pct:.1f}%)'
            )
        self._legend_lbl.setText(" &nbsp; ".join(legend_parts))

        # 片段表格
        self._table.setRowCount(0)
        for clip in result.clips:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(f"{clip['start_sec']:.2f}"))
            self._table.setItem(row, 1, QTableWidgetItem(f"{clip['end_sec']:.2f}"))
            self._table.setItem(row, 2, QTableWidgetItem(clip["label"]))
            dur = clip.get("duration_sec", clip["end_sec"] - clip["start_sec"])
            self._table.setItem(row, 3, QTableWidgetItem(f"{dur:.2f}"))

    def _on_clip_clicked(self, row: int, col: int):
        if self._last_result is None or row >= len(self._last_result.clips):
            return
        clip = self._last_result.clips[row]
        self._player.seek_to(clip["start_sec"])

    def _on_timeline_seek(self, sec: float):
        self._player.seek_to(sec)

    def _on_player_position(self, sec: float):
        self._timeline.set_cursor(sec)

"""
图片查看器。

功能：
- QLabel 贴图，自适应窗口缩放（保持宽高比）
- 可选绘制检测框（label + bbox）
- 空态提示
"""
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy


# 检测框颜色循环表
_BOX_COLORS = [
    "#4CAF50", "#FF9800", "#EF5350", "#2196F3",
    "#9C27B0", "#00BCD4", "#FFEB3B", "#795548",
]


class ImageViewer(QWidget):
    """自适应尺寸的图片查看器"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._detections: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("未选择图片")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(200, 200)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label.setStyleSheet(
            "background: #1E1E1E; color: #666; border: 1px solid #3C3C3C;"
        )
        layout.addWidget(self._label)

    # -------- public --------

    def load_image(self, image_path: str, detections: Optional[list[dict]] = None):
        """
        加载图片并可选绘制检测框。

        Args:
            image_path: 图片路径
            detections: [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float|None}]
        """
        if not image_path:
            self.clear()
            return

        pix = QPixmap(image_path)
        if pix.isNull():
            self._pixmap = None
            self._label.setText(f"无法加载图片:\n{image_path}")
            return

        self._pixmap = pix
        self._detections = detections or []
        self._refresh()

    def clear(self):
        self._pixmap = None
        self._detections = []
        self._label.setText("未选择图片")
        self._label.setPixmap(QPixmap())

    # -------- internal --------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap is None or self._pixmap.isNull():
            return

        target = self._label.size()
        scaled = self._pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if self._detections:
            scaled = self._draw_detections(
                scaled,
                src_w=self._pixmap.width(),
                src_h=self._pixmap.height(),
            )

        self._label.setPixmap(scaled)

    def _draw_detections(self, pix: QPixmap, src_w: int, src_h: int) -> QPixmap:
        """在已缩放的 pixmap 上绘制检测框（坐标从原图映射）"""
        canvas = QPixmap(pix)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        scale_x = canvas.width() / src_w
        scale_y = canvas.height() / src_h

        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)

        for i, det in enumerate(self._detections):
            bbox = det.get("bbox")
            label = det.get("label", "?")
            conf = det.get("confidence")
            if bbox is None or len(bbox) < 4:
                continue

            color = QColor(_BOX_COLORS[i % len(_BOX_COLORS)])
            pen = QPen(color, 2)
            painter.setPen(pen)

            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # 标签背景 + 文字
            text = f"{label} {conf:.2f}" if conf is not None else label
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(text) + 6
            th = metrics.height()
            painter.fillRect(x1, max(0, y1 - th), tw, th, color)
            painter.setPen(QColor("#000"))
            painter.drawText(x1 + 3, max(th - 3, y1 - 3), text)

        painter.end()
        return canvas

"""
底部日志控件。

功能：
- 显示来自 src.utils.logger 的日志（通过 Qt 信号跨线程安全投递）
- 按等级着色（INFO/WARNING/ERROR/DEBUG）
- 最多保留 N 行，超出则丢弃最旧的
- 支持手动清空 / 自动滚动到底部开关
"""
import logging
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor, QFont
from PyQt5.QtWidgets import (
    QWidget, QPlainTextEdit, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QLabel,
)

_LEVEL_COLORS = {
    "DEBUG": "#888888",
    "INFO": "#DDDDDD",
    "WARNING": "#FFA726",
    "ERROR": "#EF5350",
    "CRITICAL": "#EF5350",
}


class _QtLogSignal(QObject):
    """桥接 logging.Handler 到 Qt 信号（跨线程安全）"""
    log_arrived = pyqtSignal(str, str, str)  # level, time_str, message


class _QtLogHandler(logging.Handler):
    """将 logging 记录转发到 Qt 信号"""

    def __init__(self, signal_obj: _QtLogSignal):
        super().__init__()
        self._signal = signal_obj

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            self._signal.log_arrived.emit(record.levelname, time_str, msg)
        except Exception:
            pass


class LogConsole(QWidget):
    """底部日志控件"""

    def __init__(self, parent=None, max_lines: int = 2000):
        super().__init__(parent)
        self._max_lines = max_lines
        self._signal = _QtLogSignal()
        self._handler = _QtLogHandler(self._signal)
        self._handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

        self._build_ui()
        self._signal.log_arrived.connect(self._on_log)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("日志"))
        toolbar.addStretch()

        self._autoscroll_cb = QCheckBox("自动滚动")
        self._autoscroll_cb.setChecked(True)
        toolbar.addWidget(self._autoscroll_cb)

        self._clear_btn = QPushButton("清空")
        self._clear_btn.setFixedWidth(60)
        self._clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(self._clear_btn)

        layout.addLayout(toolbar)

        # 文本区
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self._max_lines)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self._text.setFont(mono)
        layout.addWidget(self._text)

    def attach_to_logger(self, logger_name: str = "smartlabel",
                          level: int = logging.INFO):
        """将本控件挂载到指定 logger"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # 去重：避免多次 attach
        for h in logger.handlers:
            if isinstance(h, _QtLogHandler):
                return
        logger.addHandler(self._handler)

    def detach(self, logger_name: str = "smartlabel"):
        logging.getLogger(logger_name).removeHandler(self._handler)

    def clear(self):
        self._text.clear()

    def append(self, level: str, message: str):
        """手动追加日志"""
        time_str = datetime.now().strftime("%H:%M:%S")
        self._on_log(level, time_str, message)

    def _on_log(self, level: str, time_str: str, message: str):
        color = _LEVEL_COLORS.get(level, "#DDDDDD")
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#777"))
        cursor.insertText(f"[{time_str}] ", fmt)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(f"{level:<7} ", fmt)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#DDDDDD"))
        cursor.insertText(f"{message}\n", fmt)

        if self._autoscroll_cb.isChecked():
            self._text.moveCursor(QTextCursor.End)

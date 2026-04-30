"""
统一 Worker 线程（内置 UI 节流）。

用法：
    worker = UnifiedWorker(lambda cb: pipeline.run(..., progress_callback=cb))
    worker.progress.connect(on_progress)
    worker.batch_result.connect(on_batch)
    worker.finished_ok.connect(on_done)
    worker.error.connect(on_error)
    worker.start()

    # 取消：
    worker.request_cancel()
"""
import threading
import time
import traceback
from typing import Any, Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal


class CancelledError(RuntimeError):
    """用户请求取消"""


class UnifiedWorker(QThread):
    """
    通用后台任务 Worker。

    Signals:
        progress(int current, int total)           # 进度更新
        batch_result(list)                         # 累积 N 条结果一次性发出
        finished_ok(object)                        # 任务成功，附 result
        error(str)                                 # 任务异常
        log(str, str)                              # (level, message) 给 UI 日志
    """

    progress = pyqtSignal(int, int)
    batch_result = pyqtSignal(list)
    finished_ok = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str, str)

    def __init__(self, task_fn: Callable[[Callable], Any],
                 ui_update_interval: int = 10,
                 progress_min_interval_ms: int = 50,
                 parent=None):
        """
        Args:
            task_fn: 执行函数，签名 task_fn(progress_cb) -> result
                     progress_cb 签名为 (current, total, latest_item)
            ui_update_interval: 每 N 条 latest_item 批量发射一次 batch_result
            progress_min_interval_ms: progress 信号最小发射间隔（毫秒），
                避免高频任务（如 video classify interval=1）刷爆 UI 事件队列
            parent: QObject parent
        """
        super().__init__(parent)
        self._task_fn = task_fn
        self._ui_update_interval = max(1, ui_update_interval)
        self._cancel_event = threading.Event()
        self._buffer = []
        self._buffer_lock = threading.Lock()
        # 节流：记录上次 progress 发射时间（毫秒）
        self._progress_min_interval = max(0, progress_min_interval_ms) / 1000.0
        self._last_progress_emit = 0.0

    def request_cancel(self):
        """请求取消。task_fn 需主动检查 is_cancelled()。"""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _progress_callback(self, current: int, total: int, latest: Any = None):
        """传给 pipeline 的进度回调（节流 progress + 批量发射 batch_result）"""
        if self._cancel_event.is_set():
            raise CancelledError("用户已取消")

        # 节流 progress 信号：距上次发射不足 min_interval 则跳过
        # 但末尾（current == total）和首次必须发射
        now = time.monotonic()
        is_terminal = (current >= total) or (current == 1)
        if is_terminal or (now - self._last_progress_emit) >= self._progress_min_interval:
            self.progress.emit(int(current), int(total))
            self._last_progress_emit = now

        if latest is not None:
            with self._buffer_lock:
                self._buffer.append(latest)
                if len(self._buffer) >= self._ui_update_interval:
                    batch = self._buffer
                    self._buffer = []
                    self.batch_result.emit(batch)

    def _flush_buffer(self):
        with self._buffer_lock:
            if self._buffer:
                batch = self._buffer
                self._buffer = []
                self.batch_result.emit(batch)

    def run(self):
        try:
            result = self._task_fn(self._progress_callback)
            self._flush_buffer()
            if self._cancel_event.is_set():
                self.error.emit("任务已取消")
                return
            self.finished_ok.emit(result)
        except CancelledError:
            self._flush_buffer()
            self.error.emit("任务已取消")
        except Exception as e:
            self._flush_buffer()
            tb = traceback.format_exc()
            self.log.emit("ERROR", tb)
            self.error.emit(f"{type(e).__name__}: {e}")

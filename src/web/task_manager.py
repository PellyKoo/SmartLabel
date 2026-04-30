"""
GPU 资源管理：EnginePool + TaskManager。

核心原则：
- GPU 推理串行（gpu_lock 互斥）
- 任务排队，不并发抢显存
- 引擎实例复用（同 key 不重复 load）
"""
import threading
import queue
import uuid
import traceback
from datetime import datetime, timedelta
from typing import Optional, Callable

from src.engine.base import BaseEngine
from src.engine.engine_factory import EngineFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 任务历史保留上限和 TTL（终态任务超过此时长会被回收）
MAX_TASKS = 200
TASK_TTL = timedelta(hours=24)
_TERMINAL_STATES = {"completed", "failed", "cancelled"}


# ==================== EnginePool ====================

class EnginePool:
    """引擎单例池，避免重复 load/unload。"""

    def __init__(self):
        self._engines: dict[str, BaseEngine] = {}
        self._lock = threading.Lock()
        # key -> Event：表示该 key 正在被某个线程加载，其他并发请求等它完成
        self._loading: dict[str, threading.Event] = {}

    def get_or_create(self, engine_key: str, config: dict) -> BaseEngine:
        """
        加载引擎（或返回已加载的）。

        关键：load() 操作（VLM 30-60s）在 self._lock **外**执行，
        否则会阻塞所有 is_loaded() / list_keys() / engine_info() 查询，
        前端轮询会卡死。

        并发场景：多个线程同时请求同 key 时，只有一个真正加载，
        其他线程等 Event 唤醒后取已加载的实例。
        """
        # 步骤 1：快速检查 + 决定谁来加载（持锁极短时间）
        with self._lock:
            if engine_key in self._engines:
                return self._engines[engine_key]

            event = self._loading.get(engine_key)
            if event is not None:
                # 别人已经在加载，自己等
                wait_event = event
                am_loader = False
            else:
                # 由当前线程负责加载
                wait_event = threading.Event()
                self._loading[engine_key] = wait_event
                am_loader = True

        # 步骤 2：非加载线程 → 等加载完成
        if not am_loader:
            wait_event.wait()
            with self._lock:
                engine = self._engines.get(engine_key)
            if engine is None:
                raise RuntimeError(f"引擎加载失败: {engine_key}")
            return engine

        # 步骤 3：加载线程 → 在锁外执行 load（其他查询不被阻塞）
        try:
            engine = EngineFactory.create(config)
            engine.load()
            with self._lock:
                self._engines[engine_key] = engine
            logger.info(f"引擎已加载: key={engine_key}")
            return engine
        finally:
            # 无论成功失败都唤醒等待者并清理 _loading 标记
            with self._lock:
                self._loading.pop(engine_key, None)
            wait_event.set()

    def get(self, engine_key: str) -> Optional[BaseEngine]:
        with self._lock:
            return self._engines.get(engine_key)

    def is_loaded(self, engine_key: str) -> bool:
        with self._lock:
            return engine_key in self._engines

    def release(self, engine_key: str):
        with self._lock:
            engine = self._engines.pop(engine_key, None)
            if engine:
                engine.unload()
                logger.info(f"引擎已释放: key={engine_key}")

    def release_all(self):
        with self._lock:
            for engine in self._engines.values():
                try:
                    engine.unload()
                except Exception:
                    pass
            self._engines.clear()

    def list_keys(self) -> list[str]:
        with self._lock:
            return list(self._engines.keys())

    def engine_info(self, engine_key: str) -> dict:
        with self._lock:
            engine = self._engines.get(engine_key)
            if engine:
                return engine.get_engine_info()
            return {}


# ==================== TaskManager ====================

class TaskManager:
    """
    后台任务管理器。

    - GPU 推理串行（_gpu_lock）
    - 任务排队等待
    - 引擎实例由 EnginePool 复用
    """

    def __init__(self):
        self.engine_pool = EnginePool()
        self._gpu_lock = threading.Lock()
        self._task_queue: queue.Queue = queue.Queue()
        self._tasks: dict[str, dict] = {}
        self._tasks_lock = threading.Lock()
        self._stop_events: dict[str, threading.Event] = {}

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, task_type: str, config: dict) -> str:
        """提交任务到队列，返回 task_id。"""
        # 顺便做一次任务历史回收（O(N)，N 至多 200 量级，不影响性能）
        self._cleanup_old_tasks()

        task_id = uuid.uuid4().hex[:8]
        stop_event = threading.Event()
        with self._tasks_lock:
            self._tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "status": "queued",
                "progress": [0, 0],
                "created_at": datetime.now().isoformat(),
                "result": None,
                "error": None,
            }
            self._stop_events[task_id] = stop_event
        self._task_queue.put((task_id, task_type, config, stop_event))
        logger.info(f"任务已提交: id={task_id}, type={task_type}")
        return task_id

    def _cleanup_old_tasks(self):
        """
        回收任务历史：
        1. 删除终态（completed/failed/cancelled）且超过 TASK_TTL 的任务
        2. 若总数仍超 MAX_TASKS，按 created_at 删最旧的终态任务
        进行中的任务（queued/running）从不删。
        """
        now = datetime.now()
        with self._tasks_lock:
            # TTL 清理
            for tid in list(self._tasks.keys()):
                t = self._tasks[tid]
                if t["status"] not in _TERMINAL_STATES:
                    continue
                try:
                    created = datetime.fromisoformat(t["created_at"])
                    if (now - created) > TASK_TTL:
                        del self._tasks[tid]
                        self._stop_events.pop(tid, None)
                except (ValueError, KeyError):
                    pass

            # 上限保护：超过 MAX_TASKS 时按时间删最旧终态任务
            if len(self._tasks) > MAX_TASKS:
                terminal_tids = sorted(
                    [tid for tid, t in self._tasks.items()
                     if t["status"] in _TERMINAL_STATES],
                    key=lambda tid: self._tasks[tid].get("created_at", ""),
                )
                while len(self._tasks) > MAX_TASKS and terminal_tids:
                    old = terminal_tids.pop(0)
                    del self._tasks[old]
                    self._stop_events.pop(old, None)

    def stop(self, task_id: str):
        """请求停止任务（协作式取消）。"""
        with self._tasks_lock:
            ev = self._stop_events.get(task_id)
        if ev:
            ev.set()

    def get_status(self, task_id: str) -> dict:
        with self._tasks_lock:
            task = self._tasks.get(task_id)
        if task is None:
            return {"error": "task not found"}
        return dict(task)

    def list_tasks(self) -> list[dict]:
        with self._tasks_lock:
            return [dict(t) for t in self._tasks.values()]

    def get_result(self, task_id: str):
        with self._tasks_lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None
        return task.get("result")

    def _set_status(self, task_id: str, **kwargs):
        with self._tasks_lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)

    def _worker_loop(self):
        while True:
            task_id, task_type, config, stop_event = self._task_queue.get()
            self._set_status(task_id, status="running")
            try:
                with self._gpu_lock:
                    if stop_event.is_set():
                        self._set_status(task_id, status="cancelled")
                        continue
                    result = self._execute(task_id, task_type, config, stop_event)
                if stop_event.is_set():
                    self._set_status(task_id, status="cancelled")
                else:
                    self._set_status(task_id, status="completed", result=result)
                    logger.info(f"任务完成: id={task_id}")
            except InterruptedError:
                # 协作式取消（progress_cb 检测到 stop_event 抛出）
                self._set_status(task_id, status="cancelled")
                logger.info(f"任务已取消: id={task_id}")
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"任务失败: id={task_id}\n{tb}")
                self._set_status(task_id, status="failed", error=str(e))

    def _make_progress_cb(self, task_id: str, stop_event: threading.Event) -> Callable:
        def _cb(current: int, total: int, latest=None):
            if stop_event.is_set():
                raise InterruptedError("任务已取消")
            self._set_status(task_id, progress=[current, total])
        return _cb

    def _execute(self, task_id: str, task_type: str, config: dict,
                 stop_event: threading.Event):
        cb = self._make_progress_cb(task_id, stop_event)
        engine_key = config["engine_key"]
        engine_cfg = config["engine_config"]
        engine = self.engine_pool.get_or_create(engine_key, engine_cfg)

        if task_type == "preannotate_cls":
            from src.pipeline.preannotate import PreAnnotatePipeline
            pipe = PreAnnotatePipeline(engine, config)
            return pipe.run_classification(
                config["image_dir"], config["output_dir"],
                config["categories"], cb,
            )

        elif task_type == "preannotate_det":
            from src.pipeline.preannotate import PreAnnotatePipeline
            pipe = PreAnnotatePipeline(engine, config)
            return pipe.run_detection(
                config["image_dir"], config["output_dir"],
                config["targets"], cb,
            )

        elif task_type == "qualitycheck_cls":
            from src.pipeline.qualitycheck import QualityCheckPipeline
            vlm_key = config.get("vlm_engine_key")
            vlm_cfg = config.get("vlm_engine_config")
            vlm = None
            if vlm_key:
                # 校验 vlm_cfg 有效（必须是 dict 且能拿到 model_path）
                if not isinstance(vlm_cfg, dict) or not vlm_cfg:
                    logger.warning(
                        f"vlm_engine_key='{vlm_key}' 但 vlm_engine_config 为空，"
                        f"跳过 VLM 升级策略"
                    )
                else:
                    vlm_model_path = (vlm_cfg.get("vlm") or {}).get("model_path") \
                                      or vlm_cfg.get("model_path")
                    if not vlm_model_path:
                        logger.warning(
                            f"vlm_engine_config 缺少 model_path，跳过 VLM 升级策略"
                        )
                    else:
                        vlm = self.engine_pool.get_or_create(vlm_key, vlm_cfg)
            pipe = QualityCheckPipeline(engine, config, vlm_engine=vlm)
            return pipe.run_classification_qc(
                config["image_dir"], config["annotation_dir"],
                config["output_dir"], config["categories"], cb,
            )

        elif task_type == "qualitycheck_det":
            from src.pipeline.qualitycheck import QualityCheckPipeline
            pipe = QualityCheckPipeline(engine, config)
            return pipe.run_detection_qc(
                config["image_dir"], config["annotation_dir"],
                config["output_dir"], config["targets"],
                iou_threshold=config.get("iou_threshold", 0.5),
                progress_callback=cb,
            )

        elif task_type == "video_classify":
            from src.pipeline.video_classify import VideoClassifyPipeline
            pipe = VideoClassifyPipeline(engine, config)
            return pipe.run_batch(config["video_dir"], config["output_dir"], cb)

        else:
            raise ValueError(f"未知任务类型: {task_type}")


# 全局单例
_manager: Optional[TaskManager] = None
_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TaskManager()
    return _manager

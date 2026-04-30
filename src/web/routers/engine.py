"""引擎管理 API：加载（异步）、释放、查询状态"""
import threading
from fastapi import APIRouter, HTTPException
from src.web.schemas import EngineLoadRequest, EngineStatusResponse
from src.web.task_manager import get_task_manager

router = APIRouter(prefix="/api/engine", tags=["engine"])

# 加载任务状态：{engine_key: {"status": loading|ready|failed, "error": str}}
_load_states: dict[str, dict] = {}
_load_lock = threading.Lock()

# 状态字典最大保留条数，超出时删除最旧的 failed 条目
_MAX_LOAD_STATES = 50


def _trim_load_states_locked():
    """
    在 _load_lock 持有时调用：超出上限时删除最旧的 failed 条目。
    保留 loading（正在进行）和 ready（实际由 engine_pool 管理但状态字典里也记着）。
    """
    if len(_load_states) <= _MAX_LOAD_STATES:
        return
    # dict 保持插入顺序，从最早的开始找 failed 删
    for k in list(_load_states.keys()):
        if len(_load_states) <= _MAX_LOAD_STATES:
            break
        if _load_states[k].get("status") == "failed":
            del _load_states[k]


def _do_load(engine_key: str, engine_config: dict):
    """后台线程执行引擎加载"""
    with _load_lock:
        _load_states[engine_key] = {"status": "loading", "error": ""}
    try:
        mgr = get_task_manager()
        mgr.engine_pool.get_or_create(engine_key, engine_config)
        with _load_lock:
            _load_states[engine_key] = {"status": "ready", "error": ""}
    except Exception as e:
        with _load_lock:
            _load_states[engine_key] = {"status": "failed", "error": str(e)}
            _trim_load_states_locked()


@router.post("/load")
def load_engine(req: EngineLoadRequest):
    """
    提交引擎加载任务（立即返回 202）。

    加载在后台线程进行，前端轮询 /api/engine/load-status/{engine_key}。
    若已加载则直接返回 ready。
    """
    mgr = get_task_manager()
    if mgr.engine_pool.is_loaded(req.engine_key):
        return {"engine_key": req.engine_key, "status": "ready"}

    with _load_lock:
        cur = _load_states.get(req.engine_key, {})
        if cur.get("status") == "loading":
            return {"engine_key": req.engine_key, "status": "loading"}
        # 重试或新加载：清掉旧的 failed 状态，让 _do_load 写入新的 loading
        _load_states.pop(req.engine_key, None)

    t = threading.Thread(
        target=_do_load, args=(req.engine_key, req.engine_config), daemon=True
    )
    t.start()
    return {"engine_key": req.engine_key, "status": "loading"}


@router.get("/load-status/{engine_key}")
def load_status(engine_key: str):
    """查询引擎加载进度（前端轮询用）"""
    mgr = get_task_manager()
    if mgr.engine_pool.is_loaded(engine_key):
        info = mgr.engine_pool.engine_info(engine_key)
        return {"engine_key": engine_key, "status": "ready", "info": info}
    with _load_lock:
        state = _load_states.get(engine_key, {"status": "not_started", "error": ""})
    return {"engine_key": engine_key, **state}


@router.post("/unload/{engine_key}")
def unload_engine(engine_key: str):
    """释放引擎"""
    mgr = get_task_manager()
    if not mgr.engine_pool.is_loaded(engine_key):
        raise HTTPException(status_code=404, detail="引擎未加载")
    mgr.engine_pool.release(engine_key)
    with _load_lock:
        _load_states.pop(engine_key, None)
    return {"message": f"引擎已释放: {engine_key}"}


@router.get("/status")
def engine_status():
    """所有已加载引擎的状态"""
    mgr = get_task_manager()
    keys = mgr.engine_pool.list_keys()
    return {
        "loaded_engines": [
            {"engine_key": k, **mgr.engine_pool.engine_info(k)}
            for k in keys
        ]
    }


@router.get("/status/{engine_key}", response_model=EngineStatusResponse)
def engine_status_single(engine_key: str):
    mgr = get_task_manager()
    is_loaded = mgr.engine_pool.is_loaded(engine_key)
    info = mgr.engine_pool.engine_info(engine_key) if is_loaded else {}
    return EngineStatusResponse(engine_key=engine_key, is_loaded=is_loaded, info=info)

"""任务管理 API：列表、状态、结果、停止"""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from src.web.schemas import TaskStatusResponse
from src.web.task_manager import get_task_manager

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("", summary="所有任务列表")
def list_tasks():
    mgr = get_task_manager()
    return mgr.list_tasks()


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
def task_status(task_id: str):
    mgr = get_task_manager()
    task = mgr.get_status(task_id)
    if "error" in task and len(task) == 1:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.get("/{task_id}/result")
def task_result(task_id: str):
    mgr = get_task_manager()
    task = mgr.get_status(task_id)
    if "error" in task and len(task) == 1:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务未完成: status={task['status']}")
    result = mgr.get_result(task_id)
    if result is None:
        return {}
    # 将 dataclass / object 序列化为 dict
    if hasattr(result, '__dict__'):
        return vars(result)
    if isinstance(result, list):
        return [vars(r) if hasattr(r, '__dict__') else r for r in result]
    return result


@router.post("/{task_id}/stop")
def stop_task(task_id: str):
    mgr = get_task_manager()
    task = mgr.get_status(task_id)
    if "error" in task and len(task) == 1:
        raise HTTPException(status_code=404, detail="任务不存在")
    mgr.stop(task_id)
    return {"message": "停止请求已发送"}


@router.get("/{task_id}/review-samples")
def review_samples(task_id: str):
    """质检任务的待复核样本"""
    mgr = get_task_manager()
    result = mgr.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务未完成或不存在")
    if isinstance(result, dict):
        return result.get("review_samples", [])
    raise HTTPException(status_code=400, detail="非质检任务")


@router.get("/{task_id}/timeline")
def timeline(task_id: str):
    """视频分类任务的时间轴数据"""
    mgr = get_task_manager()
    result = mgr.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务未完成或不存在")
    # result 是 list[VideoClipResult] 或 VideoClipResult
    if isinstance(result, list):
        return [
            {"video_path": r.video_path, "clips": r.clips, "statistics": r.statistics}
            for r in result
        ]
    if hasattr(result, "clips"):
        return {"video_path": result.video_path, "clips": result.clips,
                "statistics": result.statistics}
    raise HTTPException(status_code=400, detail="非视频分类任务")


@router.get("/{task_id}/report")
def download_report(task_id: str):
    """下载质检 HTML 报告"""
    mgr = get_task_manager()
    task = mgr.get_status(task_id)
    if "error" in task and len(task) == 1:
        raise HTTPException(status_code=404, detail="任务不存在")
    result = mgr.get_result(task_id)
    if not result or not isinstance(result, dict):
        raise HTTPException(status_code=400, detail="无可下载报告")
    # output_dir 在任务 config 中，这里从 result 旁路取
    output_dir = task.get("_output_dir", "")
    if not output_dir:
        raise HTTPException(status_code=404, detail="未记录输出目录")
    report_path = os.path.join(output_dir, "qc_report.html")
    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="报告文件不存在")
    return FileResponse(report_path, media_type="text/html",
                        filename="qc_report.html")

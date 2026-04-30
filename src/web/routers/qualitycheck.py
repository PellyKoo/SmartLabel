"""质检 API"""
from fastapi import APIRouter
from src.web.schemas import (
    QualityCheckClsRequest, QualityCheckDetRequest, TaskStartResponse,
)
from src.web.task_manager import get_task_manager

router = APIRouter(prefix="/api/qualitycheck", tags=["qualitycheck"])


@router.post("/classification/start", response_model=TaskStartResponse)
def start_cls(req: QualityCheckClsRequest):
    """启动分类质检任务"""
    mgr = get_task_manager()
    task_id = mgr.submit("qualitycheck_cls", req.model_dump())
    return TaskStartResponse(task_id=task_id)


@router.post("/detection/start", response_model=TaskStartResponse)
def start_det(req: QualityCheckDetRequest):
    """启动检测质检任务"""
    mgr = get_task_manager()
    task_id = mgr.submit("qualitycheck_det", req.model_dump())
    return TaskStartResponse(task_id=task_id)

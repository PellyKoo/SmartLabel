"""预标注 API"""
from fastapi import APIRouter
from src.web.schemas import (
    PreAnnotateClsRequest, PreAnnotateDetRequest, TaskStartResponse,
)
from src.web.task_manager import get_task_manager

router = APIRouter(prefix="/api/preannotate", tags=["preannotate"])


@router.post("/classification/start", response_model=TaskStartResponse)
def start_cls(req: PreAnnotateClsRequest):
    """启动分类预标注任务"""
    mgr = get_task_manager()
    task_id = mgr.submit("preannotate_cls", req.model_dump())
    return TaskStartResponse(task_id=task_id)


@router.post("/detection/start", response_model=TaskStartResponse)
def start_det(req: PreAnnotateDetRequest):
    """启动检测预标注任务"""
    mgr = get_task_manager()
    task_id = mgr.submit("preannotate_det", req.model_dump())
    return TaskStartResponse(task_id=task_id)

"""视频分类 API"""
from fastapi import APIRouter
from src.web.schemas import VideoClassifyRequest, TaskStartResponse
from src.web.task_manager import get_task_manager

router = APIRouter(prefix="/api/video", tags=["video"])


@router.post("/classify/start", response_model=TaskStartResponse)
def start_classify(req: VideoClassifyRequest):
    """启动视频片段分类任务"""
    mgr = get_task_manager()
    task_id = mgr.submit("video_classify", req.model_dump())
    return TaskStartResponse(task_id=task_id)

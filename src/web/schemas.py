"""Pydantic 请求/响应模型"""
from typing import Any, Optional
from pydantic import BaseModel, Field


# ==================== 引擎 ====================

class EngineLoadRequest(BaseModel):
    engine_key: str = Field(..., description="引擎唯一标识，如 'vlm_7b' 或 'dms_onnx'")
    engine_config: dict = Field(..., description="EngineFactory.create() 接受的配置字典")


class EngineStatusResponse(BaseModel):
    engine_key: str
    is_loaded: bool
    info: dict = {}


# ==================== 预标注 ====================

class PreAnnotateClsRequest(BaseModel):
    engine_key: str
    engine_config: dict
    image_dir: str
    output_dir: str
    categories: list[str]
    output_format: str = "both"
    file_operation: str = "copy"
    num_io_workers: int = 4
    ui_update_interval: int = 10


class PreAnnotateDetRequest(BaseModel):
    engine_key: str
    engine_config: dict
    image_dir: str
    output_dir: str
    targets: list[str]
    num_io_workers: int = 4
    ui_update_interval: int = 10


# ==================== 质检 ====================

class QualityCheckClsRequest(BaseModel):
    engine_key: str
    engine_config: dict
    image_dir: str
    annotation_dir: str
    output_dir: str
    categories: list[str]
    vlm_engine_key: Optional[str] = None
    vlm_engine_config: Optional[dict] = None
    escalation_enabled: bool = True
    escalation_threshold: float = 0.8
    num_io_workers: int = 4
    ui_update_interval: int = 10


class QualityCheckDetRequest(BaseModel):
    engine_key: str
    engine_config: dict
    image_dir: str
    annotation_dir: str
    output_dir: str
    targets: list[str]
    iou_threshold: float = 0.5
    num_io_workers: int = 4
    ui_update_interval: int = 10


# ==================== 视频 ====================

class VideoClassifyRequest(BaseModel):
    engine_key: str
    engine_config: dict
    video_dir: str
    output_dir: str
    categories: list[str]
    strategy: str = "temporal_smooth"
    window_sec: float = 5.0
    stride_sec: float = 2.5
    sample_fps: float = 1.0
    smooth_window: int = 3
    min_segment_sec: float = 2.0
    output_format: str = "both"
    ui_update_interval: int = 1


# ==================== 任务 ====================

class TaskStatusResponse(BaseModel):
    id: str
    type: str
    status: str        # queued | running | completed | failed | cancelled
    progress: list[int] = [0, 0]
    created_at: str
    error: Optional[str] = None


class TaskStartResponse(BaseModel):
    task_id: str
    message: str = "任务已提交"

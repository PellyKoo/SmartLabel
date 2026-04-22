from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ==================== 能力声明 ====================

class Capability(Enum):
    """引擎能力枚举"""
    CLASSIFY = "classify"
    DETECT = "detect"
    QC_WITH_REASON = "qc_with_reason"
    VIDEO_MULTIFRAME = "video_multiframe"


# ==================== 统一数据结构 ====================

@dataclass
class ClassificationResult:
    image_path: str
    predicted_class: str
    confidence: Optional[float]     # 自有模型=softmax概率; VLM=None或采样一致比例
    is_uncertain: bool
    raw_output: str                 # 原始输出（VLM 文本 / logits 字符串）


@dataclass
class DetectionResult:
    image_path: str
    detections: list[dict] = field(default_factory=list)
    # [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]


@dataclass
class QCResult:
    image_path: str
    human_label: str
    engine_label: str
    is_consistent: bool
    confidence: Optional[float]
    reason: str                     # VLM 给出的理由；自有模型为空


@dataclass
class VideoClipResult:
    video_path: str
    clips: list[dict] = field(default_factory=list)
    # [{"start_sec", "end_sec", "label", "confidence"}]
    statistics: dict = field(default_factory=dict)
    # {"normal": 45.0, "fatigue": 12.5} 各类时长(秒)


# ==================== 引擎基类 ====================

class BaseEngine(ABC):
    """推理引擎抽象基类"""

    @property
    @abstractmethod
    def capabilities(self) -> set[Capability]:
        """声明引擎支持的能力集合"""

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    @abstractmethod
    def load(self):
        """加载模型到 GPU"""

    @abstractmethod
    def unload(self):
        """释放显存"""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @abstractmethod
    def get_engine_info(self) -> dict:
        """引擎信息：类型、模型路径、显存占用等"""

    # ---- 分类（所有引擎都应实现） ----
    @abstractmethod
    def classify(self, image_path: str, categories: list[str]) -> ClassificationResult:
        ...

    # ---- 检测（仅声明了 DETECT 能力的引擎实现） ----
    def detect(self, image_path: str, targets: list[str]) -> DetectionResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持 detect，"
            f"当前能力: {self.capabilities}"
        )

    # ---- 质检（默认实现：分类后对比。VLM 可覆盖为带理由版本） ----
    def classify_for_qc(self, image_path: str, human_label: str,
                         categories: list[str]) -> QCResult:
        result = self.classify(image_path, categories)
        return QCResult(
            image_path=image_path,
            human_label=human_label,
            engine_label=result.predicted_class,
            is_consistent=(human_label == result.predicted_class),
            confidence=result.confidence,
            reason=""
        )

    # ---- 视频多帧（默认实现：逐帧分类+投票。VLM 可覆盖） ----
    def classify_video_frames(self, frame_paths: list[str],
                               categories: list[str]) -> ClassificationResult:
        from collections import Counter
        votes = []
        for fp in frame_paths:
            r = self.classify(fp, categories)
            votes.append(r.predicted_class)
        winner = Counter(votes).most_common(1)[0][0]
        agreement = votes.count(winner) / len(votes)
        return ClassificationResult(
            image_path=frame_paths[0],
            predicted_class=winner,
            confidence=agreement,
            is_uncertain=(agreement < 0.6),
            raw_output=str(dict(Counter(votes)))
        )

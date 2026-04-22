from src.engine.base import BaseEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EngineFactory:
    """引擎工厂，根据配置创建对应的引擎实例。"""

    @staticmethod
    def create(config: dict) -> BaseEngine:
        """
        根据配置创建引擎实例。

        Args:
            config: 引擎配置字典，需包含 "type" 字段。
                type="vlm"       -> VLMEngine
                type="pytorch"   -> CustomModelEngine(model_format="pytorch")
                type="onnx"      -> CustomModelEngine(model_format="onnx")
                type="tensorrt"  -> CustomModelEngine(model_format="tensorrt")

        Returns:
            BaseEngine 实例（未加载模型，需手动调用 load()）
        """
        engine_type = config.get("type", "vlm")
        logger.info(f"创建引擎: type={engine_type}")

        if engine_type == "vlm":
            from src.engine.vlm_engine import VLMEngine
            vlm_config = config.get("vlm", {})
            # 将顶层 prompts_dir 传递给 VLM 引擎
            if "prompts_dir" not in vlm_config and "prompts_dir" in config:
                vlm_config["prompts_dir"] = config["prompts_dir"]
            return VLMEngine(vlm_config)

        elif engine_type in ("pytorch", "onnx", "tensorrt"):
            from src.engine.custom_engine import CustomModelEngine
            cfg = config.get("custom", {}).copy()
            cfg["model_format"] = engine_type
            return CustomModelEngine(cfg)

        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")

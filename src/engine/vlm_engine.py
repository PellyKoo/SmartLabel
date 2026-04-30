import os
from collections import Counter
from typing import Optional

import torch

from src.engine.base import BaseEngine, Capability, ClassificationResult, QCResult
from src.postprocess.vlm_parser import VLMOutputParser
from src.prompts.manager import PromptManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VLMEngine(BaseEngine):
    """Qwen2-VL 视觉语言模型引擎"""

    def __init__(self, config: dict):
        """
        config:
            model_path: str                 # 模型路径
            quantization: "awq" | "gptq" | "bnb4" | "bnb8" | "none"
                - awq/gptq: 加载已量化权重（transformers 自动识别）
                - bnb4: 运行时 4-bit NF4 量化（bitsandbytes，Windows 兼容）
                - bnb8: 运行时 8-bit 量化（bitsandbytes）
                - none: fp16/bf16 原始权重
            device: "cuda:0"
            torch_dtype: "float16" | "bfloat16"
            max_new_tokens: 256
            multi_sample: bool              # 是否开启多次采样（默认 False）
            sample_count: int               # 采样次数（默认 3）
            temperature: float              # 采样温度（默认 0.7，仅 multi_sample 时用）
            prompts_dir: str                # Prompt 模板目录
        """
        self._config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._parser = VLMOutputParser()
        self._prompt_manager = PromptManager(
            config.get("prompts_dir", "configs/prompts")
        )

    @property
    def capabilities(self) -> set[Capability]:
        return {
            Capability.CLASSIFY,
            Capability.QC_WITH_REASON,
            Capability.VIDEO_MULTIFRAME,
        }

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """
        加载 Qwen2-VL：
        - Qwen2VLForConditionalGeneration.from_pretrained(model_path)
        - AWQ 模型自动识别
        - 2080Ti 强制 torch.float16
        """
        if self.is_loaded:
            logger.warning("VLM 引擎已加载，跳过重复加载")
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_path = self._config["model_path"]

        # ---------- 设备自动解析 ----------
        device = self._config.get("device", "auto")
        cuda_available = torch.cuda.is_available()
        if device == "auto":
            device = "cuda:0" if cuda_available else "cpu"
            logger.info(f"device=auto 解析为: {device} "
                        f"(CUDA {'可用' if cuda_available else '不可用'})")
        is_cpu = device == "cpu" or not cuda_available

        # 把解析后的 device 写回 config，供后续推理 _infer_messages.to(device) 使用
        self._config["device"] = device

        # ---------- dtype 解析 + CPU 降级 ----------
        dtype_str = self._config.get("torch_dtype", "float16")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        if is_cpu:
            # CPU 上 fp16 推理慢且数值不稳，强制 fp32
            if dtype != torch.float32:
                logger.warning(f"CPU 模式下 {dtype} 不稳定，自动切换为 float32")
                dtype = torch.float32
            # 7B 模型在 CPU 上几乎无法跑（OOM 或极慢）
            if any(tag in model_path for tag in ("7B", "7b", "13B", "13b", "32B", "32b")):
                logger.warning(
                    "⚠️ CPU 模式下检测到大模型（7B+），推理会非常慢且可能 OOM。"
                    "强烈建议切换到 2B 模型。"
                )
        elif dtype == torch.bfloat16:
            # 2080Ti / V100 / T4 等 sm < 80 的卡不支持 bfloat16
            gpu_name = torch.cuda.get_device_name(0)
            if "2080" in gpu_name or "1080" in gpu_name or "Tesla T4" in gpu_name:
                logger.warning(f"检测到 {gpu_name}，不支持 bfloat16，强制使用 float16")
                dtype = torch.float16

        # ---------- 量化解析 + CPU 降级 ----------
        quantization = self._config.get("quantization", "none")
        if is_cpu and quantization in ("bnb4", "bnb8", "awq", "gptq"):
            logger.warning(
                f"CPU 模式不支持 {quantization} 量化（需要 CUDA），"
                f"自动切换为 none（fp32）"
            )
            quantization = "none"
            self._config["quantization"] = "none"

        logger.info(f"加载 VLM 模型: {model_path} "
                    f"(device={device}, 量化={quantization}, dtype={dtype})")

        load_kwargs = {"torch_dtype": dtype}

        # bnb4 / bnb8：运行时量化（仅 CUDA 可用，绕开 autoawq）
        if quantization in ("bnb4", "bnb8"):
            from transformers import BitsAndBytesConfig
            if quantization == "bnb4":
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs["device_map"] = "auto"   # bnb 要求
        else:
            # awq/gptq/none：transformers 自动识别 AWQ/GPTQ 元数据
            # CPU 下 device_map="cpu"，GPU 下用具体 device
            load_kwargs["device_map"] = "cpu" if is_cpu else device

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(model_path)

        logger.info(f"VLM 模型加载完成，设备: {device}, 量化: {quantization}")

    def unload(self):
        """释放显存"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM 引擎已释放")

    def get_engine_info(self) -> dict:
        info = {
            "type": "vlm",
            "model_path": self._config.get("model_path", ""),
            "quantization": self._config.get("quantization", "none"),
            "device": self._config.get("device", "cuda:0"),
            "is_loaded": self.is_loaded,
        }
        device = self._config.get("device", "cuda:0")
        if self.is_loaded and torch.cuda.is_available() and device.startswith("cuda"):
            try:
                device_idx = int(device.split(":")[-1])
                allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
                total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                info["gpu_memory_allocated_gb"] = round(allocated, 2)
                info["gpu_memory_total_gb"] = round(total, 2)
                info["gpu_name"] = torch.cuda.get_device_name(device_idx)
            except Exception as e:
                logger.warning(f"获取显存信息失败: {e}")
        return info

    # ==================== 分类 ====================

    def classify(self, image_path: str, categories: list[str]) -> ClassificationResult:
        """单张分类。若开启 multi_sample，执行多次采样并返回一致性结果。"""
        if self._config.get("multi_sample", False):
            return self._multi_sample_classify(image_path, categories)
        return self._single_classify(image_path, categories)

    def _single_classify(self, image_path: str, categories: list[str]) -> ClassificationResult:
        """单次推理"""
        prompt = self._prompt_manager.render("classify_json", categories=categories)
        raw_output = self._infer(image_path, prompt)
        label, is_uncertain = self._parser.parse_classification(raw_output, categories)
        return ClassificationResult(
            image_path=image_path,
            predicted_class=label,
            confidence=None,
            is_uncertain=is_uncertain,
            raw_output=raw_output
        )

    def _multi_sample_classify(self, image_path: str, categories: list[str]) -> ClassificationResult:
        """
        多次采样 + 投票。confidence = 一致投票比例。

        示例：3 次采样结果 [phone_call, phone_call, smoking]
              -> predicted_class = phone_call
              -> confidence = 2/3 = 0.67
              -> is_uncertain = True（< 1.0 表示有分歧）
        """
        n = self._config.get("sample_count", 3)
        temp = self._config.get("temperature", 0.7)
        # temperature<=0 时 _infer 会走贪心解码，N 次结果完全相同，多次采样毫无意义
        if temp <= 0:
            logger.warning(
                f"multi_sample=True 但 temperature={temp} 会退化为贪心解码，"
                f"N 次结果将完全相同。已自动调整为 0.7。"
            )
            temp = 0.7
        prompt = self._prompt_manager.render("classify_json", categories=categories)

        results = []
        for _ in range(n):
            raw = self._infer(image_path, prompt, temperature=temp)
            label, _ = self._parser.parse_classification(raw, categories)
            results.append(label)

        counter = Counter(results)
        winner = counter.most_common(1)[0][0]
        agreement = counter[winner] / n

        return ClassificationResult(
            image_path=image_path,
            predicted_class=winner,
            confidence=round(agreement, 2),
            is_uncertain=(agreement < 1.0),
            raw_output=str(dict(counter))
        )

    # ==================== 质检 ====================

    def classify_for_qc(self, image_path: str, human_label: str,
                         categories: list[str]) -> QCResult:
        """覆盖基类：使用质检专用 Prompt，VLM 返回判断理由。"""
        prompt = self._prompt_manager.render(
            "qc_json", human_label=human_label, categories=categories
        )
        raw_output = self._infer(image_path, prompt)
        parsed = self._parser.parse_qc_json(raw_output)

        return QCResult(
            image_path=image_path,
            human_label=human_label,
            engine_label=parsed.get("suggested_label", "parse_error"),
            is_consistent=parsed.get("correct", None) is True,
            confidence=None,
            reason=parsed.get("reason", raw_output)
        )

    # ==================== 视频多帧 ====================

    def classify_video_frames(self, frame_paths: list[str],
                               categories: list[str]) -> ClassificationResult:
        """
        覆盖基类：利用 Qwen2-VL 原生多图/视频理解。
        - 帧数 <= 8：直接多图输入
        - 帧数 > 8：均匀采样 8 帧
        """
        max_frames = 8
        if len(frame_paths) <= max_frames:
            sampled = frame_paths
        else:
            step = len(frame_paths) / max_frames
            sampled = [frame_paths[int(i * step)] for i in range(max_frames)]

        prompt = self._prompt_manager.render("video_clip", categories=categories)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": [f"file://{f}" for f in sampled], "fps": 1.0},
            {"type": "text", "text": prompt}
        ]}]
        raw_output = self._infer_messages(messages)
        label, is_uncertain = self._parser.parse_classification(raw_output, categories)

        return ClassificationResult(
            image_path=sampled[0],
            predicted_class=label,
            confidence=None,
            is_uncertain=is_uncertain,
            raw_output=raw_output
        )

    # ==================== 底层推理 ====================

    def _infer(self, image_path: str, prompt: str, temperature: float = 0.0) -> str:
        """单图推理底层方法"""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]
        return self._infer_messages(messages, temperature)

    def _infer_messages(self, messages: list, temperature: float = 0.0) -> str:
        """
        通用推理方法：
        1. processor.apply_chat_template()
        2. process_vision_info()
        3. model.generate()
        4. decode
        """
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self._config.get("device", "cuda:0"))

        gen_kwargs = {"max_new_tokens": self._config.get("max_new_tokens", 256)}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        result = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return result.strip()

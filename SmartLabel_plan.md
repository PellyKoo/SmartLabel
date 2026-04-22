# SmartLabel：AI 辅助标注与质检平台 — 技术方案

> 通用图像/视频标注辅助工具，支持 VLM 和自有模型双引擎，覆盖预标注、质检、视频片段分类全流程。


---

## 一、项目概述

### 1.1 背景与痛点

| 项目阶段 | 痛点 | 期望 |
|----------|------|------|
| 成熟项目（如 DMS，已有 30w 数据训练模型） | 外包标注质量不稳定，缺自动化质检 | 自有模型预标注 + VLM 智能质检 |
| 新项目（无现成模型） | 标注从零开始，效率低 | VLM 通用预标注 |
| 视频数据 | 人工逐帧查看标注，极低效 | 自动片段分类 + 人工校验 |
| 跨平台 | Windows 本地 + Linux 服务器，需求不同 | 三种前端统一后端 |

### 1.2 核心架构："双引擎 + 三前端"

```
推理引擎（可插拔，能力声明）：
  ├── VLM 引擎（Qwen2-VL）        → 通用：分类、质检推理、视频多帧理解
  └── 自有模型引擎                  → 专用：高精度分类、检测
       ├── PyTorch (.pt/.pth)
       ├── ONNX (.onnx)
       └── TensorRT (.engine)

前端（共享后端）：
  ├── PyQt5 桌面端     → Windows/Linux，全功能 GUI
  ├── CLI 终端         → Linux 服务器，批量无人值守
  └── Web（FastAPI）   → 服务器远程，浏览器操作
```

### 1.3 典型使用场景

| 场景 | 推理引擎 | 智能策略 | 前端 |
|------|----------|----------|------|
| DMS 预标注 | 自有模型 | 直接高精度预标注 | PyQt / CLI |
| DMS 质检 | 自有模型 + VLM | 自有模型初筛 → 低置信度转 VLM 复核 | PyQt / Web |
| 新项目预标注 | VLM | VLM 通用能力，可选多次采样提升稳定性 | PyQt / CLI |
| 视频片段分类 | VLM（多帧理解） | 滑动窗口 + 时序平滑 + 片段检测 | PyQt / Web |
| 大规模批量 | 自有模型 / VLM | GPU 任务队列串行调度 | CLI |

### 1.4 约束条件

- **数据安全**：全程内网，零外部 API
- **硬件兼容**：2080Ti（11GB）到数据中心 GPU
- **格式兼容**：检测 = Pascal VOC XML，分类 = 文件夹/CSV，视频 = CSV/JSON 时间段
- **跨平台**：Windows（PyQt）+ Linux（CLI + Web）

---

## 二、项目结构

```
smartlabel/
├── configs/
│   ├── default.yaml                 # 默认配置
│   ├── profiles/                    # 项目预设配置
│   │   ├── dms_preannotate.yaml
│   │   ├── dms_qualitycheck.yaml
│   │   └── new_project_vlm.yaml
│   └── prompts/                     # Prompt 模板（文件化）
│       ├── classify_json.txt
│       ├── qc_json.txt
│       ├── video_clip.txt
│       └── detect_grounding.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── engine/                      # ===== 推理引擎（可插拔） =====
│   │   ├── __init__.py
│   │   ├── base.py                  # 抽象基类 + 能力声明
│   │   ├── vlm_engine.py            # VLM 引擎（Qwen2-VL）
│   │   ├── custom_engine.py         # 自有模型引擎（PT/ONNX/TRT）
│   │   └── engine_factory.py        # 引擎工厂
│   │
│   ├── pipeline/                    # ===== 业务流水线 =====
│   │   ├── __init__.py
│   │   ├── preannotate.py           # 预标注（图片分类+检测）
│   │   ├── qualitycheck.py          # 质检（支持双引擎升级策略）
│   │   └── video_classify.py        # 视频片段分类（时序平滑）
│   │
│   ├── io/                          # ===== 数据 I/O =====
│   │   ├── __init__.py
│   │   ├── voc_xml.py               # VOC XML 读写
│   │   ├── classification_io.py     # 分类结果读写
│   │   ├── video_io.py              # 视频读取、抽帧
│   │   └── image_loader.py          # 图片批量加载
│   │
│   ├── postprocess/                 # ===== 输出后处理 =====
│   │   ├── __init__.py
│   │   ├── vlm_parser.py            # VLM 输出解析（JSON优先+多级降级）
│   │   ├── coord_converter.py       # 坐标转换
│   │   └── temporal.py              # 时序平滑 & 片段检测
│   │
│   ├── prompts/                     # ===== Prompt 管理 =====
│   │   ├── __init__.py
│   │   └── manager.py
│   │
│   ├── report/                      # ===== 报告生成 =====
│   │   ├── __init__.py
│   │   └── generator.py             # HTML/CSV 报告（含错误case导出）
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── visualize.py
│   │   └── metrics.py               # 评估指标（增强版）
│   │
│   ├── gui/                         # ===== PyQt5 桌面端 =====
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── engine_panel.py      # 引擎配置（双引擎支持）
│   │   │   ├── preannotate_tab.py
│   │   │   ├── qualitycheck_tab.py
│   │   │   ├── video_tab.py
│   │   │   ├── benchmark_tab.py
│   │   │   ├── settings_tab.py
│   │   │   ├── image_viewer.py
│   │   │   ├── video_player.py
│   │   │   ├── result_browser.py
│   │   │   └── log_console.py
│   │   ├── threads/
│   │   │   ├── __init__.py
│   │   │   └── worker.py            # 统一 Worker（内置 UI 节流）
│   │   └── styles/
│   │       └── theme.qss
│   │
│   ├── web/                         # ===== Web（FastAPI） =====
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI 应用
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py
│   │   │   ├── preannotate.py
│   │   │   ├── qualitycheck.py
│   │   │   ├── video.py
│   │   │   └── tasks.py
│   │   ├── task_manager.py          # 任务队列 + GPU 锁 + 引擎池
│   │   └── static/                  # 前端静态文件
│   │
│   └── cli/                         # ===== CLI =====
│       ├── __init__.py
│       └── commands.py              # Typer 命令
│
├── scripts/
│   ├── run_gui.py
│   ├── run_web.py
│   └── run_cli.py
│
├── tests/
├── requirements.txt
├── README.md
└── Makefile
```

---

## 三、推理引擎层

### 3.1 抽象基类 + 能力声明 (`src/engine/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ==================== 能力声明 ====================

class Capability(Enum):
    """引擎能力枚举"""
    CLASSIFY = "classify"              # 图片分类
    DETECT = "detect"                  # 目标检测
    QC_WITH_REASON = "qc_with_reason"  # 质检并给出理由（VLM 独有）
    VIDEO_MULTIFRAME = "video_multiframe"  # 多帧/视频理解（VLM 独有）


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
    detections: list[dict]          # [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]

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
    clips: list[dict]               # [{"start_sec", "end_sec", "label", "confidence"}]
    statistics: dict                 # {"normal": 45.0, "fatigue": 12.5} 各类时长(秒)


# ==================== 引擎基类 ====================

class BaseEngine(ABC):
    """推理引擎抽象基类"""

    @property
    @abstractmethod
    def capabilities(self) -> set[Capability]:
        """声明引擎支持的能力集合"""

    def supports(self, cap: Capability) -> bool:
        """检查是否支持某能力"""
        return cap in self.capabilities

    @abstractmethod
    def load(self):
        """加载模型到 GPU"""

    @abstractmethod
    def unload(self):
        """释放显存"""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def get_engine_info(self) -> dict:
        """引擎信息：类型、模型路径、显存占用等"""

    # ---- 分类（所有引擎都应实现） ----
    @abstractmethod
    def classify(self, image_path: str, categories: list[str]) -> ClassificationResult: ...

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
```

**Pipeline 调用前必须检查能力**：
```python
# 在 pipeline 中
if not engine.supports(Capability.DETECT):
    logger.warning("当前引擎不支持检测，跳过检测预标注")
    return None
```

---

### 3.2 VLM 引擎 (`src/engine/vlm_engine.py`)

```python
class VLMEngine(BaseEngine):
    """Qwen2-VL 视觉语言模型引擎"""

    def __init__(self, config: dict):
        """
        config:
            model_path: str
            quantization: "awq" | "gptq" | "none"
            device: "cuda:0"
            torch_dtype: "float16" | "bfloat16"
            max_new_tokens: 256
            multi_sample: bool       # 是否开启多次采样（默认 False）
            sample_count: int        # 采样次数（默认 3）
            temperature: float       # 采样温度（默认 0.7，仅 multi_sample 时用）
        """
        self._config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._parser = None  # VLMOutputParser 实例

    @property
    def capabilities(self) -> set[Capability]:
        return {
            Capability.CLASSIFY,
            Capability.QC_WITH_REASON,
            Capability.VIDEO_MULTIFRAME,
            # 注意：不声明 DETECT，VLM 检测不可靠
            # 如确需使用，可在子类或配置中显式开启
        }

    def load(self):
        """
        加载 Qwen2-VL：
        - Qwen2VLForConditionalGeneration.from_pretrained(model_path)
        - AWQ 模型自动识别
        - 2080Ti 强制 torch.float16
        - 初始化 VLMOutputParser
        """

    def classify(self, image_path, categories) -> ClassificationResult:
        """
        单张分类。若开启 multi_sample，执行多次采样并返回一致性结果。

        Prompt 格式（JSON 优先）：
            请观察图片，判断类别。只输出 JSON：
            {"label": "类别名", "reason": "简要原因"}

        解析流程：
            VLMOutputParser.parse_classification_json() → 三级降级
        """
        if self._config.get("multi_sample", False):
            return self._multi_sample_classify(image_path, categories)
        return self._single_classify(image_path, categories)

    def _single_classify(self, image_path, categories) -> ClassificationResult:
        """单次推理"""
        prompt = self._prompt_manager.render("classify_json", categories=categories)
        raw_output = self._infer(image_path, prompt)
        label, is_uncertain = self._parser.parse_classification(raw_output, categories)
        return ClassificationResult(
            image_path=image_path,
            predicted_class=label,
            confidence=None,           # 单次推理无真实 confidence，标记 None
            is_uncertain=is_uncertain,
            raw_output=raw_output
        )

    def _multi_sample_classify(self, image_path, categories) -> ClassificationResult:
        """
        多次采样 + 投票。confidence = 一致投票比例。

        示例：3 次采样结果 [phone_call, phone_call, smoking]
              → predicted_class = phone_call
              → confidence = 2/3 = 0.67
              → is_uncertain = True（< 1.0 表示有分歧）
        """
        n = self._config.get("sample_count", 3)
        temp = self._config.get("temperature", 0.7)
        results = []
        for _ in range(n):
            raw = self._infer(image_path, prompt, temperature=temp)
            label, _ = self._parser.parse_classification(raw, categories)
            results.append(label)

        from collections import Counter
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

    def classify_for_qc(self, image_path, human_label, categories) -> QCResult:
        """
        覆盖基类：使用质检专用 Prompt，VLM 返回判断理由。

        Prompt 要求输出 JSON：
            {"correct": true/false, "suggested_label": "xxx", "reason": "xxx"}
        """
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

    def classify_video_frames(self, frame_paths, categories) -> ClassificationResult:
        """
        覆盖基类：利用 Qwen2-VL 原生多图/视频理解。

        - 帧数 ≤ 8：直接多图输入
        - 帧数 > 8：均匀采样 8 帧
        """
        sampled = frame_paths if len(frame_paths) <= 8 else \
                  [frame_paths[i] for i in range(0, len(frame_paths), len(frame_paths) // 8)][:8]

        messages = [{"role": "user", "content": [
            {"type": "video", "video": [f"file://{f}" for f in sampled], "fps": 1.0},
            {"type": "text", "text": self._prompt_manager.render("video_clip", categories=categories)}
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
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self._config["device"])

        gen_kwargs = {"max_new_tokens": self._config.get("max_new_tokens", 256)}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self._model.generate(**inputs, **gen_kwargs)
        return self._processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
```

---

### 3.3 自有模型引擎 (`src/engine/custom_engine.py`)

```python
class CustomModelEngine(BaseEngine):
    """自有训练模型引擎，支持 PyTorch / ONNX / TensorRT"""

    def __init__(self, config: dict):
        """
        config:
            model_path: str
            model_format: "pytorch" | "onnx" | "tensorrt"
            model_type: "classifier" | "detector" | "both"
            class_names: list[str]        # 有序类别表
            input_size: [H, W]
            confidence_threshold: float   # 检测阈值，默认 0.5
            device: str
            preprocess:
                mean: [float, float, float]
                std: [float, float, float]
                channel_order: "RGB" | "BGR"
                resize_mode: "letterbox" | "direct"
        """

    @property
    def capabilities(self) -> set[Capability]:
        caps = {Capability.CLASSIFY}
        model_type = self._config.get("model_type", "classifier")
        if model_type in ("detector", "both"):
            caps.add(Capability.DETECT)
        return caps

    def load(self):
        """
        按 model_format 分发加载：
        - pytorch: torch.load() + eval() + to(device)
        - onnx: onnxruntime.InferenceSession(providers=["CUDAExecutionProvider"])
        - tensorrt: trt.Runtime + deserialize_cuda_engine
        """

    def classify(self, image_path, categories) -> ClassificationResult:
        """
        1. 预处理：resize → normalize → to_tensor
        2. forward → softmax
        3. argmax → class_names 映射
        4. confidence = softmax 概率值（真实值）
        """

    def detect(self, image_path, targets) -> DetectionResult:
        """
        1. 预处理：letterbox resize → normalize
        2. forward → 解码（YOLO 格式 / SSD 格式，按配置）
        3. NMS
        4. confidence < threshold → 过滤
        5. 坐标映射回原图尺寸
        """

    def _preprocess(self, image_path: str):
        """
        统一预处理流程（根据 config.preprocess 配置）：
        - 读取图片（OpenCV）
        - 通道转换（BGR→RGB 或保持）
        - Resize（letterbox 或 direct）
        - Normalize（mean/std）
        - To tensor
        """
```

---

### 3.4 引擎工厂 (`src/engine/engine_factory.py`)

```python
class EngineFactory:

    @staticmethod
    def create(config: dict) -> BaseEngine:
        engine_type = config.get("type", "vlm")
        if engine_type == "vlm":
            return VLMEngine(config.get("vlm", {}))
        elif engine_type in ("pytorch", "onnx", "tensorrt"):
            cfg = config.get("custom", {})
            cfg["model_format"] = engine_type
            return CustomModelEngine(cfg)
        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")
```

---

## 四、VLM 输出解析（鲁棒性增强）(`src/postprocess/vlm_parser.py`)

**设计原则**：JSON 优先解析，三级降级兜底，绝不伪造数据。

```python
import json
import re
from typing import Optional


class VLMOutputParser:
    """
    VLM 输出文本解析器。

    解析策略（按优先级降级）：
      Level 1: JSON 解析（prompt 要求输出 JSON）
      Level 2: Regex 提取（"label: xxx" 格式）
      Level 3: 包含匹配（输出文本中包含候选类别）
      Level 4: 标记 uncertain
    """

    def __init__(self, categories: list[str]):
        self.categories = categories
        self._label_regex = re.compile(
            r'(?:"label"|label)\s*[:：]\s*["\']?([^"\'}\n,]+)', re.IGNORECASE
        )

    def parse_classification(self, raw_output: str,
                              categories: list[str] = None) -> tuple[str, bool]:
        """
        解析分类输出。

        Returns: (predicted_class, is_uncertain)
        """
        cats = categories or self.categories
        text = raw_output.strip()

        # Level 1: JSON 解析
        label = self._try_json_extract(text, "label")
        if label:
            matched = self._exact_match(label, cats)
            if matched:
                return matched, False

        # Level 2: Regex 提取
        match = self._label_regex.search(text)
        if match:
            extracted = match.group(1).strip()
            matched = self._exact_match(extracted, cats)
            if matched:
                return matched, False

        # Level 3: 精确文本匹配（整个输出就是类别名）
        matched = self._exact_match(text, cats)
        if matched:
            return matched, False

        # Level 4: 包含匹配（输出中包含某个类别名，取最长匹配）
        found = [c for c in cats if c.lower() in text.lower()]
        if found:
            return max(found, key=len), False

        # 全部失败
        return "uncertain", True

    def parse_qc_json(self, raw_output: str) -> dict:
        """
        解析质检 JSON 输出。
        容错：去 markdown 代码块、修复布尔值大小写。
        """
        text = raw_output.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # 尝试提取 JSON 对象子串（应对前后有多余文字）
        json_match = re.search(r'\{[^{}]+\}', text)
        if json_match:
            text = json_match.group(0)

        text = text.replace("True", "true").replace("False", "false")
        text = text.replace("None", "null")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "correct": None,
                "suggested_label": "parse_error",
                "reason": raw_output[:200]
            }

    def parse_detection(self, raw_output: str,
                         img_width: int, img_height: int) -> list[dict]:
        """
        解析 VLM grounding 输出。
        Qwen2-VL 坐标范围 [0, 1000] → 像素坐标。
        """
        detections = []
        # 尝试 JSON 数组解析
        try:
            items = json.loads(raw_output)
            if isinstance(items, list):
                for item in items:
                    if "bbox" in item and "label" in item:
                        bbox = item["bbox"]
                        abs_bbox = [
                            bbox[0] / 1000 * img_width,
                            bbox[1] / 1000 * img_height,
                            bbox[2] / 1000 * img_width,
                            bbox[3] / 1000 * img_height,
                        ]
                        detections.append({
                            "label": item["label"],
                            "bbox": [int(c) for c in abs_bbox],
                            "confidence": None
                        })
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return detections

    # ---- 内部工具方法 ----

    def _try_json_extract(self, text: str, key: str) -> Optional[str]:
        """尝试从 JSON 字符串中提取指定 key"""
        try:
            # 先尝试直接解析
            data = json.loads(text)
            if isinstance(data, dict) and key in data:
                return str(data[key]).strip()
        except json.JSONDecodeError:
            # 尝试提取 JSON 子串
            match = re.search(r'\{[^{}]+\}', text)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if key in data:
                        return str(data[key]).strip()
                except json.JSONDecodeError:
                    pass
        return None

    def _exact_match(self, text: str, categories: list[str]) -> Optional[str]:
        """大小写不敏感的精确匹配"""
        text_lower = text.lower().strip()
        for cat in categories:
            if text_lower == cat.lower():
                return cat
        return None
```

---

## 五、时序处理模块 (`src/postprocess/temporal.py`)

```python
from collections import Counter


def temporal_smooth(labels: list[str], window: int = 3) -> list[str]:
    """
    时序中值滤波平滑。消除孤立噪声点。

    示例：
        输入: [normal, normal, fatigue, normal, normal]
        输出: [normal, normal, normal, normal, normal]
        （中间的孤立 fatigue 被平滑掉）

    Args:
        labels: 时间序列标签列表
        window: 滤波窗口大小（奇数）
    """
    if window < 3:
        return labels
    half = window // 2
    smoothed = list(labels)
    for i in range(half, len(labels) - half):
        window_labels = labels[i - half: i + half + 1]
        counter = Counter(window_labels)
        smoothed[i] = counter.most_common(1)[0][0]
    return smoothed


def detect_segments(labels: list[str], timestamps: list[float],
                     min_duration_sec: float = 2.0) -> list[dict]:
    """
    连续相同标签合并为片段，过滤掉过短的片段。

    Args:
        labels: 时间序列标签
        timestamps: 对应的时间戳（秒）
        min_duration_sec: 最短片段时长，低于此值的片段合并到前一个

    Returns: [{"start_sec", "end_sec", "label", "duration_sec"}]
    """
    if not labels:
        return []

    raw_segments = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            raw_segments.append({
                "start_sec": timestamps[start_idx],
                "end_sec": timestamps[i],
                "label": current_label,
            })
            current_label = labels[i]
            start_idx = i

    # 最后一个片段
    raw_segments.append({
        "start_sec": timestamps[start_idx],
        "end_sec": timestamps[-1],
        "label": current_label,
    })

    # 过滤过短片段：合并到前一个片段
    filtered = []
    for seg in raw_segments:
        seg["duration_sec"] = round(seg["end_sec"] - seg["start_sec"], 2)
        if seg["duration_sec"] < min_duration_sec and filtered:
            filtered[-1]["end_sec"] = seg["end_sec"]
            filtered[-1]["duration_sec"] = round(
                filtered[-1]["end_sec"] - filtered[-1]["start_sec"], 2
            )
        else:
            filtered.append(seg)

    return filtered
```

---

## 六、视频片段分类流水线 (`src/pipeline/video_classify.py`)

```python
class VideoClassifyPipeline:

    def __init__(self, engine: BaseEngine, config: dict):
        """
        config:
            window_sec: 5.0
            stride_sec: 2.5
            sample_fps: 1.0
            categories: [str]
            strategy: "vote" | "temporal_smooth" | "vlm_multiframe"
            smooth_window: 3           # 时序平滑窗口
            min_segment_sec: 2.0       # 最短片段时长
            temp_dir: "/tmp/smartlabel"
        """

    def run(self, video_path: str, output_path: str,
            progress_callback=None) -> VideoClipResult:
        """
        完整流程：

        1. 滑动窗口抽帧
           VideoReader.sliding_window() → 窗口列表

        2. 逐窗口推理
           策略选择：
           - "vlm_multiframe"：engine.classify_video_frames(frames)
             利用 VLM 多帧理解，一次推理整个窗口
           - "vote"：逐帧分类 + 多数投票
             适合自有模型

        3. 时序后处理
           策略 "temporal_smooth"（默认推荐）：
           a) 对窗口级标签序列做时序平滑（消除噪声）
           b) 片段检测（连续同类合并）
           c) 过滤过短片段

           策略 "vote"（简单模式）：
           直接合并相邻同类窗口，不做平滑

        4. 输出结果（CSV + JSON）
        """
        video = VideoReader(video_path)
        windows = video.sliding_window(
            window_sec=self.config["window_sec"],
            stride_sec=self.config["stride_sec"],
            sample_fps=self.config["sample_fps"]
        )

        # 逐窗口推理
        window_labels = []
        window_timestamps = []
        strategy = self.config.get("strategy", "temporal_smooth")

        for i, win in enumerate(windows):
            if strategy == "vlm_multiframe" and \
               self.engine.supports(Capability.VIDEO_MULTIFRAME):
                result = self.engine.classify_video_frames(
                    win["frame_paths"], self.config["categories"]
                )
            else:
                result = self.engine.classify_video_frames(
                    win["frame_paths"], self.config["categories"]
                )
            window_labels.append(result.predicted_class)
            window_timestamps.append(win["start_sec"])

            if progress_callback:
                progress_callback(i + 1, len(windows), result)

        # 时序后处理
        if strategy in ("temporal_smooth", "vlm_multiframe"):
            smoothed = temporal_smooth(
                window_labels,
                window=self.config.get("smooth_window", 3)
            )
            clips = detect_segments(
                smoothed, window_timestamps,
                min_duration_sec=self.config.get("min_segment_sec", 2.0)
            )
        else:
            clips = detect_segments(window_labels, window_timestamps)

        # 统计各类时长
        stats = {}
        for clip in clips:
            stats[clip["label"]] = stats.get(clip["label"], 0) + clip["duration_sec"]

        return VideoClipResult(
            video_path=video_path, clips=clips, statistics=stats
        )
```

---

## 七、质检流水线 — 低置信度智能升级 (`src/pipeline/qualitycheck.py`)

**核心策略**：如果同时加载了自有模型和 VLM 两个引擎，高置信度样本直接由自有模型判定，低置信度样本升级到 VLM 做深度复核。

```python
class QualityCheckPipeline:

    def __init__(self, primary_engine: BaseEngine, config: dict,
                 vlm_engine: BaseEngine = None):
        """
        primary_engine: 主引擎（自有模型或 VLM）
        vlm_engine: 可选的 VLM 引擎，用于低置信度升级
        config:
            escalation_threshold: 0.8   # 低于此置信度 → 升级到 VLM
            escalation_enabled: True    # 是否开启升级策略
        """
        self.primary = primary_engine
        self.vlm = vlm_engine
        self.config = config

    def run_classification_qc(self, image_dir, annotation_dir, output_dir,
                                categories, progress_callback=None) -> dict:
        """
        质检流程：

        对每张图片：
        1. primary_engine 分类 → 得到 engine_label + confidence

        2. 判断升级：
           if escalation_enabled
              and vlm_engine is not None
              and confidence < escalation_threshold:
                → VLM 二次复核（带理由）

        3. 对比人工标注 vs 最终引擎判断
           一致 → PASS
           不一致 → REVIEW（附带 VLM 理由，如果有）

        Returns:
            {
                "total_checked": int,
                "pass_count": int,
                "review_count": int,
                "review_ratio": float,
                "escalated_count": int,            # 升级到 VLM 的数量
                "category_review_stats": {
                    "phone_call": {"total": 100, "review": 5, "ratio": 0.05},
                    ...
                },
                "review_samples": [
                    {
                        "image_path": str,
                        "human_label": str,
                        "engine_label": str,
                        "confidence": float | None,
                        "escalated": bool,          # 是否经过 VLM 复核
                        "vlm_reason": str            # VLM 理由（如有）
                    }, ...
                ],
                "error_cases": [...]                # 典型错误样本（报告用）
            }
        """
        results = []
        annotations = read_classification_folders(annotation_dir)
        images = scan_images(image_dir)
        escalation_threshold = self.config.get("escalation_threshold", 0.8)
        escalation_enabled = self.config.get("escalation_enabled", True) and self.vlm is not None

        for i, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            human_label = annotations.get(filename)
            if human_label is None:
                continue

            # Step 1: 主引擎分类
            primary_result = self.primary.classify(img_path, categories)

            # Step 2: 判断是否升级
            escalated = False
            vlm_reason = ""
            final_label = primary_result.predicted_class
            final_confidence = primary_result.confidence

            if escalation_enabled and \
               primary_result.confidence is not None and \
               primary_result.confidence < escalation_threshold:
                # 低置信度 → VLM 复核
                qc_result = self.vlm.classify_for_qc(img_path, human_label, categories)
                final_label = qc_result.engine_label
                vlm_reason = qc_result.reason
                escalated = True

            # Step 3: 对比
            is_consistent = (human_label == final_label)
            results.append({
                "image_path": img_path,
                "human_label": human_label,
                "engine_label": final_label,
                "confidence": final_confidence,
                "is_consistent": is_consistent,
                "escalated": escalated,
                "vlm_reason": vlm_reason,
            })

            if progress_callback and (i + 1) % self.config.get("ui_update_interval", 10) == 0:
                progress_callback(i + 1, len(images), results[-1])

        # 汇总
        review_samples = [r for r in results if not r["is_consistent"]]
        # ...构建返回 dict...

    def run_detection_qc(self, image_dir, xml_dir, output_dir,
                           targets, iou_threshold=0.5,
                           progress_callback=None) -> dict:
        """
        检测质检：
        - 需要引擎支持 Capability.DETECT
        - 对比人工 VOC XML 与引擎检测结果
        - IoU 匹配 → 标记漏标/多标/类别错误/框偏差
        """
        if not self.primary.supports(Capability.DETECT):
            raise RuntimeError("当前主引擎不支持检测能力，无法执行检测质检")
        # ... 实现 ...
```

---

## 八、GPU 资源管理 (`src/web/task_manager.py`)

```python
import threading
import queue
import uuid
from datetime import datetime


class EnginePool:
    """引擎单例池——避免重复 load/unload"""

    def __init__(self):
        self._engines: dict[str, BaseEngine] = {}
        self._lock = threading.Lock()

    def get_or_create(self, engine_key: str, config: dict) -> BaseEngine:
        """
        获取引擎实例。同一配置只创建一次。
        engine_key: 唯一标识（如 "vlm_awq" 或 "dms_onnx"）
        """
        with self._lock:
            if engine_key not in self._engines:
                engine = EngineFactory.create(config)
                engine.load()
                self._engines[engine_key] = engine
            return self._engines[engine_key]

    def release(self, engine_key: str):
        with self._lock:
            if engine_key in self._engines:
                self._engines[engine_key].unload()
                del self._engines[engine_key]

    def release_all(self):
        with self._lock:
            for engine in self._engines.values():
                engine.unload()
            self._engines.clear()


class TaskManager:
    """
    后台任务管理器。

    核心原则：
    - GPU 推理串行执行（gpu_lock 互斥）
    - 任务排队等待，不并发抢占显存
    - 引擎实例复用（不重复加载）
    """

    def __init__(self):
        self.engine_pool = EnginePool()
        self._gpu_lock = threading.Lock()       # GPU 推理互斥锁
        self._task_queue = queue.Queue()
        self._tasks: dict[str, dict] = {}       # task_id → 状态
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def submit(self, task_type: str, config: dict) -> str:
        """提交任务到队列，返回 task_id"""
        task_id = str(uuid.uuid4())[:8]
        self._tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "queued",
            "progress": [0, 0],
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
        }
        self._task_queue.put((task_id, task_type, config))
        return task_id

    def get_status(self, task_id: str) -> dict:
        return self._tasks.get(task_id, {"error": "task not found"})

    def _worker_loop(self):
        """后台 Worker 循环，串行处理任务"""
        while True:
            task_id, task_type, config = self._task_queue.get()
            self._tasks[task_id]["status"] = "running"
            try:
                with self._gpu_lock:                # GPU 互斥
                    result = self._execute(task_type, config, task_id)
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["result"] = result
            except Exception as e:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = str(e)

    def _execute(self, task_type, config, task_id):
        """根据任务类型执行对应 pipeline"""
        engine = self.engine_pool.get_or_create(
            config["engine_key"], config["engine_config"]
        )

        def progress_cb(current, total, latest):
            self._tasks[task_id]["progress"] = [current, total]

        if task_type == "preannotate_cls":
            pipeline = PreAnnotatePipeline(engine, config)
            return pipeline.run_classification(
                config["image_dir"], config["output_dir"],
                config["categories"], progress_callback=progress_cb
            )
        elif task_type == "qualitycheck_cls":
            vlm = self.engine_pool.get_or_create(
                config.get("vlm_key"), config.get("vlm_config")
            ) if config.get("vlm_key") else None
            pipeline = QualityCheckPipeline(engine, config, vlm_engine=vlm)
            return pipeline.run_classification_qc(
                config["image_dir"], config["annotation_dir"],
                config["output_dir"], config["categories"],
                progress_callback=progress_cb
            )
        # ... 其他任务类型 ...
```

---

## 九、Prompt 模板（JSON 优先格式）

### configs/prompts/classify_json.txt

```
请观察这张图片，判断图中人员的行为状态。

可选类别：{categories}

请严格只输出以下 JSON 格式，不要输出任何其他文字：
{{"label": "类别名称"}}
```

### configs/prompts/qc_json.txt

```
这是一张监控图片。现有人工标注为："{human_label}"

请判断此标注是否正确。可选类别：{categories}

只输出 JSON，不要其他文字：
{{"correct": true或false, "suggested_label": "你认为正确的类别", "reason": "简要理由"}}
```

### configs/prompts/video_clip.txt

```
这是一段监控视频的连续画面。请综合所有画面，判断这段时间内人员的整体行为状态。

可选类别：{categories}

只输出 JSON：
{{"label": "类别名称"}}
```

### configs/prompts/detect_grounding.txt

```
请在这张图片中定位以下目标：{targets}

只输出 JSON 数组：
[{{"label": "目标名称", "bbox": [x1, y1, x2, y2]}}]

bbox 为像素坐标，左上角为原点。未找到的目标不要输出。
```

---

## 十、报告生成（增强版）(`src/report/generator.py`)

```python
class ReportGenerator:

    def generate_qc_html_report(self, qc_results: dict, output_path: str):
        """
        HTML 质检报告内容：
        1. 总体统计看板（通过率、异议率、升级率）
        2. 各类别异议率柱状图
        3. 待复核样本图库（缩略图 + 人工标签 vs 引擎标签 + VLM 理由）
        4. 典型错误案例分析（Top 10 高频错误模式）
           - 如 "yawning 被标为 normal" 出现 8 次
           - 附带典型图片
        5. 升级到 VLM 的样本统计（如开启了升级策略）
        """

    def generate_qc_csv_report(self, qc_results: dict, output_path: str):
        """CSV 报告：每行一条样本，含所有字段"""

    def export_error_cases(self, qc_results: dict, output_dir: str):
        """
        导出错误 case（新增）：
        - 按错误类型分文件夹
        - 每张图附带 metadata JSON
        - 便于 review 和分析标注规范问题
        输出结构：
            error_cases/
            ├── yawning_as_normal/
            │   ├── img001.jpg
            │   ├── img001.json  # {"human": "normal", "engine": "yawning", "reason": "..."}
            │   └── ...
            ├── smoking_as_phone_call/
            │   └── ...
        """

    def generate_video_report(self, video_result: VideoClipResult, output_path: str):
        """视频分类报告：时间轴可视化 + 各类时长统计"""

    def generate_preannotate_summary(self, pa_results: dict, output_path: str):
        """预标注汇总"""
```

---

## 十一、评估指标（增强版）(`src/utils/metrics.py`)

```python
class Evaluator:

    @staticmethod
    def classification_report(y_true, y_pred, class_names) -> dict:
        """
        返回:
            accuracy, per-class precision/recall/F1,
            macro-F1, weighted-F1, confusion_matrix
        """

    @staticmethod
    def detection_report(pred_boxes, gt_boxes,
                          iou_thresholds=[0.5]) -> dict:
        """
        返回:
            mAP@0.5, per-class AP, precision, recall, 平均 IoU
        """

    @staticmethod
    def disagreement_analysis(results: list[dict]) -> dict:
        """
        分歧分析（新增）：
        输入质检结果列表，输出：
        - 高频错误模式 Top N（如 "yawning→normal" 出现 15 次）
        - 各类别被错判为什么的分布
        - 升级到 VLM 后纠正率
        """

    @staticmethod
    def plot_confusion_matrix(cm, class_names, output_path=None):
        """混淆矩阵热力图"""
```

---

## 十二、配置文件 (`configs/default.yaml`)

```yaml
# ========================================
# SmartLabel v3 默认配置
# ========================================

# ---------- 引擎配置 ----------
engine:
  type: "vlm"                      # "vlm" | "pytorch" | "onnx" | "tensorrt"

  vlm:
    model_path: "/data/models/Qwen2-VL-7B-Instruct-AWQ"
    quantization: "awq"
    device: "cuda:0"
    torch_dtype: "float16"
    max_new_tokens: 256
    multi_sample: false             # 多次采样（开启后精度↑速度↓）
    sample_count: 3
    temperature: 0.7

  custom:
    model_path: ""
    model_format: "onnx"
    model_type: "classifier"        # "classifier" | "detector" | "both"
    class_names: []
    input_size: [640, 640]
    confidence_threshold: 0.5
    preprocess:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      channel_order: "RGB"
      resize_mode: "letterbox"

# ---------- 任务配置 ----------
task:
  classification:
    categories: []
    output_format: "both"           # "folder" | "csv" | "both"
    file_operation: "copy"          # "copy" | "symlink"

  detection:
    targets: []
    output_format: "voc_xml"

  video:
    categories: []
    window_sec: 5.0
    stride_sec: 2.5
    sample_fps: 1.0
    strategy: "temporal_smooth"     # "vote" | "temporal_smooth" | "vlm_multiframe"
    smooth_window: 3
    min_segment_sec: 2.0
    output_format: "both"           # "csv" | "json" | "both"

# ---------- 质检配置 ----------
quality_check:
  iou_threshold: 0.5
  sample_mode: "all"
  sample_ratio: 1.0
  escalation_enabled: true          # 低置信度升级到 VLM
  escalation_threshold: 0.8         # 置信度低于此值 → VLM 复核

# ---------- 运行配置 ----------
runtime:
  num_io_workers: 4                 # IO 线程数（非 GPU 线程）
  ui_update_interval: 10            # GUI 每 N 张更新一次
  log_level: "INFO"
  temp_dir: "/tmp/smartlabel"

# ---------- 输出配置 ----------
output:
  report_format: "both"
  save_visualization: true
  export_error_cases: true          # 导出错误 case 文件夹

# ---------- Prompt ----------
prompts:
  dir: "configs/prompts"
  classification: "classify_json.txt"
  qc: "qc_json.txt"
  video: "video_clip.txt"
  detection: "detect_grounding.txt"
```

---

## 十三、三种前端设计

### 13.1 架构关系

```
┌────────────────────────────────────────────────────────────┐
│                    共享后端核心                               │
│  engine/ + pipeline/ + io/ + postprocess/ + report/ + ...  │
└─────────┬──────────────────┬──────────────────┬────────────┘
          │                  │                  │
   ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────────┐
   │  PyQt5 GUI  │   │    CLI      │   │ FastAPI + Web   │
   │ (全功能桌面) │   │ (批量无头)   │   │ (服务器远程)     │
   └─────────────┘   └─────────────┘   └─────────────────┘
```

### 13.2 PyQt5 桌面端

**主窗口**：左侧引擎面板 + 右侧 Tab（预标注/质检/视频/评估/设置） + 底部日志。

**引擎面板**（升级版）：
```
┌─ 引擎配置 ──────────────────┐
│                              │
│  ── 主引擎 ──                │
│  类型: [VLM ▼] / [自有模型 ▼] │
│  模型路径: [选择...]          │
│  [加载] [释放]   ● 已就绪     │
│  显存: 4.8GB / 11.0GB       │
│                              │
│  ── VLM 辅助引擎（可选） ──   │
│  ☐ 启用低置信度 VLM 复核      │
│  模型路径: [选择...]          │
│  升级阈值: [0.8]             │
│  [加载] [释放]   ○ 未加载     │
│                              │
│  ── VLM 采样设置 ──          │
│  ☐ 多次采样      次数: [3]   │
│  温度: [0.7]                 │
└──────────────────────────────┘
```

**视频 Tab**（含时间轴）：
```
┌─────────────────────────────────────────────────────────────────┐
│ 视频片段分类                                                     │
├──────────────────────────────┬──────────────────────────────────┤
│                              │        视频预览                   │
│  视频文件/目录:               │   ┌────────────────────────┐    │
│  [选择文件] [选择目录]        │   │    视频播放器            │    │
│                              │   │    (当前片段高亮)        │    │
│  分类类别:                    │   └────────────────────────┘    │
│  ☑ normal  ☑ fatigue        │   ⏮  ▶  ⏭   00:12 / 01:30     │
│  ☑ distracted  [+ 添加]     │                                  │
│                              │   ── 时间轴 ──                   │
│  策略: [时序平滑 ▼]          │   ┌────────────────────────┐    │
│  窗口: [5.0]s  步长: [2.5]s  │   │██normal██▓fatigue▓█████│    │
│  平滑窗口: [3]               │   └────────────────────────┘    │
│  最短片段: [2.0]s             │                                  │
│                              ├──────────────────────────────────┤
│  ▶ 开始  ⏸ 暂停  ⏹ 停止    │  00:00-00:12  normal   ✅       │
│                              │  00:12-00:25  fatigue  ⚠️       │
│  📊 导出CSV  📄 导出JSON    │  00:25-01:00  normal   ✅       │
└──────────────────────────────┴──────────────────────────────────┘
```

**后台 Worker 节流**：
```python
class UnifiedWorker(QThread):
    progress = pyqtSignal(int, int)
    batch_result = pyqtSignal(list)     # 批量发送，非逐张
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, pipeline_func, ui_update_interval=10):
        """
        ui_update_interval: 每 N 条结果 emit 一次 batch_result
        内部用 buffer 暂存，到 N 条一次性发出
        """
```

### 13.3 CLI

```bash
# 预标注
python scripts/run_cli.py preannotate \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/dms/batch01/images \
    --output-dir /data/dms/batch01/pre_annotations \
    --task classification

# 质检（双引擎升级模式）
python scripts/run_cli.py qualitycheck \
    --config configs/profiles/dms_qualitycheck.yaml \
    --image-dir /data/dms/batch01/images \
    --annotation-dir /data/dms/batch01/annotations \
    --output-dir /data/dms/batch01/qc_report \
    --task classification \
    --vlm-config configs/profiles/new_project_vlm.yaml  # 可选：指定 VLM 引擎配置

# 视频
python scripts/run_cli.py video-classify \
    --config configs/profiles/dms_qualitycheck.yaml \
    --video-dir /data/dms/videos \
    --output-dir /data/dms/video_labels \
    --strategy temporal_smooth

# 评估
python scripts/run_cli.py benchmark \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/benchmark/images \
    --gt-dir /data/benchmark/ground_truth
```

### 13.4 Web（FastAPI）

**API**：

```
POST   /api/engine/load              # 加载引擎
POST   /api/engine/unload            # 释放
GET    /api/engine/status             # 引擎状态 + 显存

POST   /api/preannotate/start        # 启动预标注任务
POST   /api/qualitycheck/start       # 启动质检
POST   /api/video/classify/start     # 启动视频分类

GET    /api/tasks                     # 所有任务列表
GET    /api/tasks/{id}/status         # 进度
GET    /api/tasks/{id}/result         # 结果
POST   /api/tasks/{id}/stop           # 停止

GET    /api/tasks/{id}/review-samples # 待复核样本（质检）
GET    /api/tasks/{id}/timeline       # 时间轴数据（视频）
GET    /api/tasks/{id}/report         # 下载报告

GET    /api/image?path=xxx            # 图片代理（前端展示）
```

**任务管理**：TaskManager（见第八节）确保 GPU 串行、引擎复用。

**前端**：原生 HTML + CSS + JS，功能对齐 PyQt 核心页面。

---

## 十四、环境与依赖

### requirements.txt

```
# ---- GUI ----
PyQt5>=5.15.9

# ---- Web ----
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# ---- CLI ----
typer>=0.9.0
rich>=13.0.0

# ---- VLM ----
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0
accelerate>=0.28.0
autoawq>=0.2.0
qwen-vl-utils>=0.0.2

# ---- 自有模型 ----
onnxruntime-gpu>=1.16.0       # 可选
# tensorrt                    # 可选，按环境安装

# ---- 数据处理 ----
Pillow>=10.0.0
opencv-python>=4.8.0
lxml>=4.9.0
pandas>=2.0.0

# ---- 可视化与报告 ----
matplotlib>=3.7.0
jinja2>=3.1.0

# ---- 评估 ----
scikit-learn>=1.3.0

# ---- 工具 ----
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 十五、开发里程碑

### Phase 1：引擎层 + 分类核心（3 天）

1. `engine/base.py` — 基类 + 能力声明 + 数据结构
2. `engine/vlm_engine.py` — VLM 引擎（分类 + 多次采样）
3. `engine/custom_engine.py` — 自有模型引擎（ONNX 分类）
4. `engine/engine_factory.py` — 工厂
5. `postprocess/vlm_parser.py` — VLM 输出解析（JSON 优先三级降级）
6. `io/classification_io.py` — 分类结果输出（copy 模式）
7. `prompts/manager.py` — Prompt 管理

**验收**：两个引擎都能跑通分类。VLM 多次采样可选开启。

### Phase 2：检测 + 质检 + 升级策略（3 天）

1. 两个引擎增加检测能力（自有模型为主）
2. `io/voc_xml.py` — VOC XML 读写
3. `pipeline/preannotate.py` — 预标注
4. `pipeline/qualitycheck.py` — 质检（含低置信度升级策略）
5. `report/generator.py` — 报告（含错误 case 导出）
6. `utils/metrics.py` — 评估指标

**验收**：预标注→质检→报告全链路。双引擎升级策略可用。

### Phase 3：视频片段分类（2 天）

1. `io/video_io.py` — 视频读取 + 滑动窗口抽帧
2. `postprocess/temporal.py` — 时序平滑 + 片段检测
3. `pipeline/video_classify.py` — 视频分类流水线

**验收**：输入视频 → 输出时间段标签 CSV/JSON，时序平滑生效。

### Phase 4：CLI（1 天）

1. `cli/commands.py` — 全部命令
2. `scripts/run_cli.py` — 入口

**验收**：所有功能命令行可用。

### Phase 5：PyQt5 GUI（4-5 天）

1. 主窗口 + 引擎面板（双引擎）
2. 预标注 / 质检 / 视频 / 评估 / 设置 Tab
3. 视频播放器 + 时间轴
4. 后台 Worker（节流）
5. 深色主题

**验收**：Windows GUI 全功能。

### Phase 6：Web 界面（3-4 天）

1. FastAPI + TaskManager + EnginePool
2. 各 API Router
3. 前端页面

**验收**：Linux 服务器 Web 全功能。

### Phase 7：打磨 + 比赛案例（2 天）

1. Prompt 调优
2. Benchmark 数据收集
3. 比赛案例撰写

---

## 十六、数据安全

| 措施 | 说明 |
|------|------|
| 模型本地 | 内网部署，不依赖外部 |
| 零网络请求 | 可断网运行 |
| 数据不出内网 | 全程内网处理存储 |
| Web 内网绑定 | FastAPI 绑定内网 IP |
| 视频临时文件 | 处理完即清理 |
| 权限控制 | chmod 750 |

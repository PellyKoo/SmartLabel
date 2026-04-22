import os
import numpy as np

from src.engine.base import BaseEngine, Capability, ClassificationResult, DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomModelEngine(BaseEngine):
    """自有训练模型引擎，支持 PyTorch / ONNX / TensorRT"""

    def __init__(self, config: dict):
        """
        config:
            model_path: str
            model_format: "pytorch" | "onnx" | "tensorrt"
            model_type: "classifier" | "detector" | "both"
            class_names: list[str]           # 有序类别表
            input_size: [H, W]
            confidence_threshold: float      # 检测阈值，默认 0.5
            device: str
            preprocess:
                mean: [float, float, float]
                std: [float, float, float]
                channel_order: "RGB" | "BGR"
                resize_mode: "letterbox" | "direct"
        """
        self._config = config
        self._model = None
        self._session = None       # ONNX session
        self._trt_context = None   # TensorRT context

    @property
    def capabilities(self) -> set[Capability]:
        caps = {Capability.CLASSIFY}
        model_type = self._config.get("model_type", "classifier")
        if model_type in ("detector", "both"):
            caps.add(Capability.DETECT)
        return caps

    @property
    def is_loaded(self) -> bool:
        fmt = self._config.get("model_format", "pytorch")
        if fmt == "pytorch":
            return self._model is not None
        elif fmt == "onnx":
            return self._session is not None
        elif fmt == "tensorrt":
            return self._trt_context is not None
        return False

    def load(self):
        """按 model_format 分发加载"""
        if self.is_loaded:
            logger.warning("自有模型引擎已加载，跳过重复加载")
            return

        fmt = self._config.get("model_format", "pytorch")
        model_path = self._config["model_path"]
        logger.info(f"加载自有模型: {model_path} (格式={fmt})")

        if fmt == "pytorch":
            self._load_pytorch(model_path)
        elif fmt == "onnx":
            self._load_onnx(model_path)
        elif fmt == "tensorrt":
            self._load_tensorrt(model_path)
        else:
            raise ValueError(f"不支持的模型格式: {fmt}")

        logger.info(f"自有模型加载完成 (格式={fmt})")

    def _load_pytorch(self, model_path: str):
        import torch
        device = self._config.get("device", "cuda:0")
        self._model = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(self._model, "eval"):
            self._model.eval()
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)

    def _load_onnx(self, model_path: str):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device = self._config.get("device", "cuda:0")
        if "cpu" in device:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)

    def _load_tensorrt(self, model_path: str):
        import tensorrt as trt
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        self._trt_context = engine.create_execution_context()
        self._trt_engine = engine

    def unload(self):
        """释放显存"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._session is not None:
            del self._session
            self._session = None
        if self._trt_context is not None:
            del self._trt_context
            self._trt_context = None
            if hasattr(self, "_trt_engine"):
                del self._trt_engine

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("自有模型引擎已释放")

    def get_engine_info(self) -> dict:
        info = {
            "type": "custom",
            "model_path": self._config.get("model_path", ""),
            "model_format": self._config.get("model_format", "pytorch"),
            "model_type": self._config.get("model_type", "classifier"),
            "class_names": self._config.get("class_names", []),
            "is_loaded": self.is_loaded,
        }
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = int(self._config.get("device", "cuda:0").split(":")[-1])
                allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
                total = torch.cuda.get_device_properties(device_idx).total_mem / 1024**3
                info["gpu_memory_allocated_gb"] = round(allocated, 2)
                info["gpu_memory_total_gb"] = round(total, 2)
        except (ImportError, RuntimeError):
            pass
        return info

    # ==================== 分类 ====================

    def classify(self, image_path: str, categories: list[str]) -> ClassificationResult:
        """
        1. 预处理：resize -> normalize -> to_tensor
        2. forward -> softmax
        3. argmax -> class_names 映射
        4. confidence = softmax 概率值
        """
        input_tensor = self._preprocess(image_path)
        class_names = self._config.get("class_names", categories)

        fmt = self._config.get("model_format", "pytorch")
        if fmt == "pytorch":
            probs = self._forward_pytorch(input_tensor)
        elif fmt == "onnx":
            probs = self._forward_onnx(input_tensor)
        elif fmt == "tensorrt":
            probs = self._forward_tensorrt(input_tensor)
        else:
            raise ValueError(f"不支持的模型格式: {fmt}")

        # softmax（如果输出不是概率分布）
        if probs.ndim == 2:
            probs = probs[0]
        if not (np.all(probs >= 0) and abs(probs.sum() - 1.0) < 0.1):
            probs = _softmax(probs)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        predicted_class = class_names[idx] if idx < len(class_names) else f"class_{idx}"

        return ClassificationResult(
            image_path=image_path,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            is_uncertain=(confidence < self._config.get("confidence_threshold", 0.5)),
            raw_output=str({class_names[i]: round(float(probs[i]), 4)
                           for i in range(min(len(probs), len(class_names)))})
        )

    def _forward_pytorch(self, input_tensor: np.ndarray) -> np.ndarray:
        import torch
        device = self._config.get("device", "cuda:0")
        tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.cpu().numpy()

    def _forward_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        input_name = self._session.get_inputs()[0].name
        input_data = input_tensor[np.newaxis].astype(np.float32)
        outputs = self._session.run(None, {input_name: input_data})
        return outputs[0]

    def _forward_tensorrt(self, input_tensor: np.ndarray) -> np.ndarray:
        import torch
        device = self._config.get("device", "cuda:0")

        input_data = torch.from_numpy(input_tensor[np.newaxis].astype(np.float32)).to(device)

        # 获取输出形状
        output_shape = tuple(self._trt_engine.get_tensor_shape(
            self._trt_engine.get_tensor_name(1)
        ))
        # 动态 batch 维度修正
        if output_shape[0] == -1:
            output_shape = (1,) + output_shape[1:]
        output_data = torch.empty(output_shape, dtype=torch.float32, device=device)

        input_name = self._trt_engine.get_tensor_name(0)
        output_name = self._trt_engine.get_tensor_name(1)
        self._trt_context.set_tensor_address(input_name, input_data.data_ptr())
        self._trt_context.set_tensor_address(output_name, output_data.data_ptr())

        stream = torch.cuda.Stream(device=device)
        self._trt_context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        return output_data.cpu().numpy()

    # ==================== 检测 ====================

    def detect(self, image_path: str, targets: list[str]) -> DetectionResult:
        """
        检测流程：
        1. 预处理：letterbox resize -> normalize
        2. forward -> 解码
        3. NMS
        4. confidence < threshold -> 过滤
        5. 坐标映射回原图尺寸
        """
        if not self.supports(Capability.DETECT):
            raise NotImplementedError(
                f"{self.__class__.__name__} 未声明检测能力，"
                f"model_type={self._config.get('model_type')}"
            )

        import cv2

        # 读取原图尺寸
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        orig_h, orig_w = img.shape[:2]

        # 预处理 + 推理
        input_tensor = self._preprocess(image_path)
        fmt = self._config.get("model_format", "pytorch")

        if fmt == "pytorch":
            raw_output = self._forward_pytorch(input_tensor)
        elif fmt == "onnx":
            raw_output = self._forward_onnx(input_tensor)
        elif fmt == "tensorrt":
            raw_output = self._forward_tensorrt(input_tensor)
        else:
            raise ValueError(f"不支持的模型格式: {fmt}")

        # 解码检测结果并映射回原图
        input_size = self._config.get("input_size", [640, 640])
        threshold = self._config.get("confidence_threshold", 0.5)
        class_names = self._config.get("class_names", targets)

        detections = self._decode_detections(
            raw_output, class_names, threshold,
            input_size, (orig_h, orig_w)
        )

        return DetectionResult(image_path=image_path, detections=detections)

    def _decode_detections(self, raw_output: np.ndarray, class_names: list[str],
                            threshold: float, input_size: list[int],
                            orig_size: tuple[int, int]) -> list[dict]:
        """
        解码检测输出（YOLO 格式: [batch, num_det, 5+num_classes]）。
        包含 NMS 和坐标映射。
        """
        if raw_output.ndim == 3:
            raw_output = raw_output[0]

        # 如果形状为 [num_det, 5+C]（cx, cy, w, h, obj_conf, cls1, cls2, ...）
        # 或 [num_det, 4+C]（cx, cy, w, h, cls1, cls2, ...）
        detections = []
        num_classes = len(class_names)

        for det in raw_output:
            if len(det) == 5 + num_classes:
                cx, cy, w, h, obj_conf = det[:5]
                cls_scores = det[5:]
                scores = obj_conf * cls_scores
            elif len(det) == 4 + num_classes:
                cx, cy, w, h = det[:4]
                scores = det[4:]
            else:
                continue

            cls_idx = int(np.argmax(scores))
            conf = float(scores[cls_idx])
            if conf < threshold:
                continue

            # cx,cy,w,h -> x1,y1,x2,y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # 映射回原图坐标
            input_h, input_w = input_size
            orig_h, orig_w = orig_size

            resize_mode = self._config.get("preprocess", {}).get("resize_mode", "letterbox")
            if resize_mode == "letterbox":
                scale = min(input_w / orig_w, input_h / orig_h)
                pad_w = (input_w - orig_w * scale) / 2
                pad_h = (input_h - orig_h * scale) / 2
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale
            else:
                x1 = x1 / input_w * orig_w
                y1 = y1 / input_h * orig_h
                x2 = x2 / input_w * orig_w
                y2 = y2 / input_h * orig_h

            # 裁剪到图片范围
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            label = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            detections.append({
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 4)
            })

        # 简单 NMS（按类别）
        detections = self._nms(detections, iou_threshold=0.45)
        return detections

    @staticmethod
    def _nms(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
        """按类别分组的 NMS"""
        if not detections:
            return []

        from collections import defaultdict
        by_class = defaultdict(list)
        for d in detections:
            by_class[d["label"]].append(d)

        result = []
        for label, dets in by_class.items():
            dets.sort(key=lambda x: x["confidence"], reverse=True)
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                dets = [d for d in dets
                        if _iou(best["bbox"], d["bbox"]) < iou_threshold]
            result.extend(keep)

        return result

    # ==================== 预处理 ====================

    def _preprocess(self, image_path: str) -> np.ndarray:
        """
        统一预处理流程（根据 config.preprocess 配置）：
        - 读取图片（OpenCV）
        - 通道转换（BGR->RGB 或保持）
        - Resize（letterbox 或 direct）
        - Normalize（mean/std）
        - To numpy tensor (C, H, W)
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        preproc = self._config.get("preprocess", {})

        # 通道转换
        channel_order = preproc.get("channel_order", "RGB")
        if channel_order == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        input_size = self._config.get("input_size", [640, 640])
        target_h, target_w = input_size
        resize_mode = preproc.get("resize_mode", "letterbox")

        if resize_mode == "letterbox":
            img = _letterbox(img, target_w, target_h)
        else:
            img = cv2.resize(img, (target_w, target_h))

        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array(preproc.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(preproc.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        img = (img - mean) / std

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return img


# ==================== 工具函数 ====================

def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _letterbox(img: np.ndarray, target_w: int, target_h: int,
               color: tuple = (114, 114, 114)) -> np.ndarray:
    """Letterbox resize，保持宽高比，填充灰边。"""
    import cv2
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    img = cv2.copyMakeBorder(
        img, pad_h, target_h - new_h - pad_h,
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=color
    )
    return img


def _iou(box1: list, box2: list) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

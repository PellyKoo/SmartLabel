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

    def __init__(self, categories: list[str] = None):
        self.categories = categories or []
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
        Qwen2-VL 坐标范围 [0, 1000] -> 像素坐标。
        """
        detections = []
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

import os
from typing import Optional, Callable

import cv2

from src.engine.base import BaseEngine, Capability, ClassificationResult, DetectionResult
from src.io.image_loader import scan_images
from src.io.classification_io import save_classification_results
from src.io.voc_xml import write_voc_xml
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreAnnotatePipeline:
    """预标注流水线：支持图片分类和目标检测。"""

    def __init__(self, engine: BaseEngine, config: dict):
        """
        Args:
            engine: 推理引擎实例（已加载）
            config: 任务配置
                categories: list[str]          # 分类类别
                targets: list[str]             # 检测目标
                output_format: str             # "folder" | "csv" | "both" | "voc_xml"
                file_operation: str            # "copy" | "symlink"
                num_io_workers: int
        """
        self.engine = engine
        self.config = config

    def run_classification(self, image_dir: str, output_dir: str,
                            categories: list[str],
                            progress_callback: Optional[Callable] = None) -> dict:
        """
        分类预标注。

        Args:
            image_dir: 输入图片目录
            output_dir: 输出目录
            categories: 类别列表
            progress_callback: fn(current, total, latest_result)

        Returns:
            {
                "total": int,
                "results": list[ClassificationResult],
                "category_counts": dict[str, int],
                "uncertain_count": int,
            }
        """
        images = scan_images(image_dir)
        if not images:
            logger.warning(f"未找到图片: {image_dir}")
            return {"total": 0, "results": [], "category_counts": {}, "uncertain_count": 0}

        logger.info(f"开始分类预标注: {len(images)} 张图片, {len(categories)} 个类别")

        results = []
        category_counts = {}
        uncertain_count = 0

        for i, img_path in enumerate(images):
            try:
                result = self.engine.classify(img_path, categories)
                results.append(result)

                cls = result.predicted_class
                category_counts[cls] = category_counts.get(cls, 0) + 1
                if result.is_uncertain:
                    uncertain_count += 1

            except Exception as e:
                logger.error(f"分类失败: {img_path}, 错误: {e}")
                results.append(ClassificationResult(
                    image_path=img_path,
                    predicted_class="error",
                    confidence=None,
                    is_uncertain=True,
                    raw_output=str(e)
                ))

            if progress_callback:
                interval = self.config.get("ui_update_interval", 10)
                if (i + 1) % interval == 0 or (i + 1) == len(images):
                    progress_callback(i + 1, len(images), results[-1])

        # 保存结果
        save_classification_results(
            results, output_dir,
            output_format=self.config.get("output_format", "both"),
            file_operation=self.config.get("file_operation", "copy"),
            num_workers=self.config.get("num_io_workers", 4)
        )

        summary = {
            "total": len(images),
            "results": results,
            "category_counts": category_counts,
            "uncertain_count": uncertain_count,
        }

        logger.info(
            f"分类预标注完成: 共 {len(images)} 张, "
            f"各类别: {category_counts}, 不确定: {uncertain_count}"
        )
        return summary

    def run_detection(self, image_dir: str, output_dir: str,
                       targets: list[str],
                       progress_callback: Optional[Callable] = None) -> dict:
        """
        检测预标注，输出 VOC XML。

        Args:
            image_dir: 输入图片目录
            output_dir: 输出 XML 目录
            targets: 检测目标列表
            progress_callback: fn(current, total, latest_result)

        Returns:
            {
                "total": int,
                "results": list[DetectionResult],
                "total_detections": int,
                "label_counts": dict[str, int],
            }
        """
        if not self.engine.supports(Capability.DETECT):
            logger.warning("当前引擎不支持检测，跳过检测预标注")
            return {"total": 0, "results": [], "total_detections": 0, "label_counts": {}}

        images = scan_images(image_dir)
        if not images:
            logger.warning(f"未找到图片: {image_dir}")
            return {"total": 0, "results": [], "total_detections": 0, "label_counts": {}}

        logger.info(f"开始检测预标注: {len(images)} 张图片, 目标: {targets}")

        os.makedirs(output_dir, exist_ok=True)
        results = []
        total_detections = 0
        label_counts = {}

        for i, img_path in enumerate(images):
            try:
                result = self.engine.detect(img_path, targets)
                results.append(result)

                # 写 VOC XML
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    filename = os.path.basename(img_path)
                    xml_name = os.path.splitext(filename)[0] + ".xml"
                    xml_path = os.path.join(output_dir, xml_name)
                    write_voc_xml(xml_path, filename, w, h, result.detections)

                # 统计
                total_detections += len(result.detections)
                for det in result.detections:
                    lbl = det["label"]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

            except Exception as e:
                logger.error(f"检测失败: {img_path}, 错误: {e}")
                results.append(DetectionResult(image_path=img_path, detections=[]))

            if progress_callback:
                interval = self.config.get("ui_update_interval", 10)
                if (i + 1) % interval == 0 or (i + 1) == len(images):
                    progress_callback(i + 1, len(images), results[-1])

        summary = {
            "total": len(images),
            "results": results,
            "total_detections": total_detections,
            "label_counts": label_counts,
        }

        logger.info(
            f"检测预标注完成: 共 {len(images)} 张, "
            f"总检出 {total_detections} 个目标, 各类: {label_counts}"
        )
        return summary

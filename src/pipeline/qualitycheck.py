import os
from typing import Optional, Callable

from src.engine.base import BaseEngine, Capability, ClassificationResult, QCResult
from src.io.image_loader import scan_images
from src.io.classification_io import read_classification_folders, read_classification_csv
from src.io.voc_xml import parse_voc_xml, read_voc_annotations
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityCheckPipeline:
    """
    质检流水线。

    核心策略：如果同时加载了自有模型和 VLM 两个引擎，
    高置信度样本直接由自有模型判定，低置信度样本升级到 VLM 做深度复核。
    """

    def __init__(self, primary_engine: BaseEngine, config: dict,
                 vlm_engine: BaseEngine = None):
        """
        Args:
            primary_engine: 主引擎（自有模型或 VLM）
            vlm_engine: 可选的 VLM 引擎，用于低置信度升级
            config:
                escalation_threshold: 0.8
                escalation_enabled: True
                ui_update_interval: 10
                num_io_workers: 4
        """
        self.primary = primary_engine
        self.vlm = vlm_engine
        self.config = config

    def run_classification_qc(self, image_dir: str, annotation_dir: str,
                                output_dir: str, categories: list[str],
                                progress_callback: Optional[Callable] = None) -> dict:
        """
        分类质检。

        对每张图片：
        1. primary_engine 分类 -> 得到 engine_label + confidence
        2. 低置信度 -> VLM 二次复核（如开启）
        3. 对比人工标注 vs 最终引擎判断

        Returns:
            {
                "total_checked": int,
                "pass_count": int,
                "review_count": int,
                "review_ratio": float,
                "escalated_count": int,
                "category_review_stats": {label: {"total": N, "review": M, "ratio": R}},
                "review_samples": [
                    {
                        "image_path", "human_label", "engine_label",
                        "confidence", "escalated", "vlm_reason"
                    }, ...
                ],
                "all_results": [...],
                "error_cases": [...]
            }
        """
        # 读取标注
        if os.path.isfile(annotation_dir) and annotation_dir.endswith(".csv"):
            annotations = read_classification_csv(annotation_dir)
        else:
            annotations = read_classification_folders(annotation_dir)

        images = scan_images(image_dir)
        if not images:
            logger.warning(f"未找到图片: {image_dir}")
            return self._empty_result()

        escalation_threshold = self.config.get("escalation_threshold", 0.8)
        escalation_enabled = (
            self.config.get("escalation_enabled", True) and self.vlm is not None
        )

        logger.info(
            f"开始分类质检: {len(images)} 张图片, "
            f"升级策略={'开启' if escalation_enabled else '关闭'}"
        )

        all_results = []
        escalated_count = 0

        for i, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            human_label = annotations.get(filename)
            if human_label is None:
                continue

            try:
                # Step 1: 主引擎分类
                primary_result = self.primary.classify(img_path, categories)

                # Step 2: 判断是否升级
                escalated = False
                vlm_reason = ""
                final_label = primary_result.predicted_class
                final_confidence = primary_result.confidence

                if (escalation_enabled
                        and primary_result.confidence is not None
                        and primary_result.confidence < escalation_threshold):
                    # 低置信度 -> VLM 复核
                    qc_result = self.vlm.classify_for_qc(img_path, human_label, categories)
                    final_label = qc_result.engine_label
                    vlm_reason = qc_result.reason
                    escalated = True
                    escalated_count += 1

                # Step 3: 对比
                is_consistent = (human_label == final_label)
                all_results.append({
                    "image_path": img_path,
                    "human_label": human_label,
                    "engine_label": final_label,
                    "confidence": final_confidence,
                    "is_consistent": is_consistent,
                    "escalated": escalated,
                    "vlm_reason": vlm_reason,
                })

            except Exception as e:
                logger.error(f"质检失败: {img_path}, 错误: {e}")
                all_results.append({
                    "image_path": img_path,
                    "human_label": human_label,
                    "engine_label": "error",
                    "confidence": None,
                    "is_consistent": False,
                    "escalated": False,
                    "vlm_reason": str(e),
                })

            if progress_callback:
                interval = self.config.get("ui_update_interval", 10)
                if (i + 1) % interval == 0 or (i + 1) == len(images):
                    progress_callback(i + 1, len(images), all_results[-1])

        # 汇总
        return self._build_summary(all_results, escalated_count, output_dir)

    def run_detection_qc(self, image_dir: str, xml_dir: str,
                           output_dir: str, targets: list[str],
                           iou_threshold: float = 0.5,
                           progress_callback: Optional[Callable] = None) -> dict:
        """
        检测质检：
        - 对比人工 VOC XML 与引擎检测结果
        - IoU 匹配 -> 标记漏标/多标/类别错误/框偏差
        """
        if not self.primary.supports(Capability.DETECT):
            raise RuntimeError("当前主引擎不支持检测能力，无法执行检测质检")

        gt_annotations = read_voc_annotations(xml_dir)
        images = scan_images(image_dir)
        if not images:
            logger.warning(f"未找到图片: {image_dir}")
            return {"total_checked": 0, "issues": []}

        logger.info(f"开始检测质检: {len(images)} 张图片, IoU 阈值={iou_threshold}")

        all_issues = []
        total_gt = 0
        total_pred = 0
        matched_count = 0

        for i, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            gt = gt_annotations.get(filename) or gt_annotations.get(
                os.path.splitext(filename)[0]
            )
            if gt is None:
                continue

            try:
                det_result = self.primary.detect(img_path, targets)
                gt_boxes = gt["objects"]
                pred_boxes = det_result.detections

                total_gt += len(gt_boxes)
                total_pred += len(pred_boxes)

                issues = self._match_detections(
                    img_path, gt_boxes, pred_boxes, iou_threshold
                )
                matched_count += sum(1 for iss in issues if iss["type"] == "matched")
                all_issues.extend([iss for iss in issues if iss["type"] != "matched"])

            except Exception as e:
                logger.error(f"检测质检失败: {img_path}, 错误: {e}")

            if progress_callback:
                interval = self.config.get("ui_update_interval", 10)
                if (i + 1) % interval == 0 or (i + 1) == len(images):
                    progress_callback(i + 1, len(images), None)

        summary = {
            "total_checked": len(images),
            "total_gt_boxes": total_gt,
            "total_pred_boxes": total_pred,
            "matched_count": matched_count,
            "miss_count": sum(1 for i in all_issues if i["type"] == "miss"),
            "extra_count": sum(1 for i in all_issues if i["type"] == "extra"),
            "label_mismatch_count": sum(1 for i in all_issues if i["type"] == "label_mismatch"),
            "issues": all_issues,
        }

        logger.info(
            f"检测质检完成: GT={total_gt}, 预测={total_pred}, "
            f"匹配={matched_count}, 漏标={summary['miss_count']}, "
            f"多标={summary['extra_count']}, 类别错误={summary['label_mismatch_count']}"
        )
        return summary

    # ==================== 内部方法 ====================

    def _match_detections(self, image_path: str, gt_boxes: list[dict],
                           pred_boxes: list[dict],
                           iou_threshold: float) -> list[dict]:
        """
        IoU 匹配 GT 和预测框。

        Returns:
            issue 列表，每条包含 type: "matched" | "miss" | "extra" | "label_mismatch"
        """
        issues = []
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # 计算 IoU 矩阵并贪心匹配
        iou_pairs = []
        for gi, gt in enumerate(gt_boxes):
            for pi, pred in enumerate(pred_boxes):
                iou = _compute_iou(gt["bbox"], pred["bbox"])
                if iou >= iou_threshold:
                    iou_pairs.append((iou, gi, pi))

        iou_pairs.sort(key=lambda x: x[0], reverse=True)

        for iou_val, gi, pi in iou_pairs:
            if gt_matched[gi] or pred_matched[pi]:
                continue
            gt_matched[gi] = True
            pred_matched[pi] = True

            gt_label = gt_boxes[gi]["label"]
            pred_label = pred_boxes[pi]["label"]

            if gt_label == pred_label:
                issues.append({
                    "type": "matched",
                    "image_path": image_path,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "iou": round(iou_val, 4),
                })
            else:
                issues.append({
                    "type": "label_mismatch",
                    "image_path": image_path,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "gt_bbox": gt_boxes[gi]["bbox"],
                    "pred_bbox": pred_boxes[pi]["bbox"],
                    "iou": round(iou_val, 4),
                })

        # 未匹配的 GT -> 漏标
        for gi, matched in enumerate(gt_matched):
            if not matched:
                issues.append({
                    "type": "miss",
                    "image_path": image_path,
                    "gt_label": gt_boxes[gi]["label"],
                    "gt_bbox": gt_boxes[gi]["bbox"],
                })

        # 未匹配的预测 -> 多标
        for pi, matched in enumerate(pred_matched):
            if not matched:
                issues.append({
                    "type": "extra",
                    "image_path": image_path,
                    "pred_label": pred_boxes[pi]["label"],
                    "pred_bbox": pred_boxes[pi]["bbox"],
                    "confidence": pred_boxes[pi].get("confidence"),
                })

        return issues

    def _build_summary(self, all_results: list[dict],
                        escalated_count: int, output_dir: str) -> dict:
        """构建分类质检汇总"""
        total = len(all_results)
        review_samples = [r for r in all_results if not r["is_consistent"]]
        pass_count = total - len(review_samples)

        # 各类别统计
        category_stats = {}
        for r in all_results:
            lbl = r["human_label"]
            if lbl not in category_stats:
                category_stats[lbl] = {"total": 0, "review": 0}
            category_stats[lbl]["total"] += 1
            if not r["is_consistent"]:
                category_stats[lbl]["review"] += 1

        for lbl, stats in category_stats.items():
            stats["ratio"] = round(stats["review"] / stats["total"], 4) if stats["total"] > 0 else 0

        # 错误模式分析
        error_patterns = {}
        for r in review_samples:
            key = f"{r['human_label']}_as_{r['engine_label']}"
            if key not in error_patterns:
                error_patterns[key] = {"count": 0, "samples": []}
            error_patterns[key]["count"] += 1
            if len(error_patterns[key]["samples"]) < 5:
                error_patterns[key]["samples"].append(r)

        # 按频次排序取 Top 错误模式
        error_cases = sorted(error_patterns.values(), key=lambda x: x["count"], reverse=True)

        return {
            "total_checked": total,
            "pass_count": pass_count,
            "review_count": len(review_samples),
            "review_ratio": round(len(review_samples) / total, 4) if total > 0 else 0,
            "escalated_count": escalated_count,
            "category_review_stats": category_stats,
            "review_samples": review_samples,
            "all_results": all_results,
            "error_cases": error_cases,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "total_checked": 0,
            "pass_count": 0,
            "review_count": 0,
            "review_ratio": 0,
            "escalated_count": 0,
            "category_review_stats": {},
            "review_samples": [],
            "all_results": [],
            "error_cases": [],
        }


def _compute_iou(box1: list, box2: list) -> float:
    """计算两个 bbox [x1,y1,x2,y2] 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """评估指标计算器"""

    @staticmethod
    def classification_report(y_true: list[str], y_pred: list[str],
                               class_names: list[str] = None) -> dict:
        """
        分类评估报告。

        Returns:
            {
                "accuracy": float,
                "macro_f1": float,
                "weighted_f1": float,
                "per_class": {
                    class_name: {"precision", "recall", "f1", "support"}
                },
                "confusion_matrix": list[list[int]],
                "class_names": list[str]
            }
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, f1_score
        )

        if class_names is None:
            class_names = sorted(set(y_true) | set(y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, zero_division=0
        )

        per_class = {}
        for i, name in enumerate(class_names):
            per_class[name] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }

        cm = confusion_matrix(y_true, y_pred, labels=class_names)

        return {
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }

    @staticmethod
    def detection_report(pred_boxes_all: list[list[dict]],
                          gt_boxes_all: list[list[dict]],
                          iou_thresholds: list[float] = None) -> dict:
        """
        检测评估报告。

        Args:
            pred_boxes_all: 每张图的预测列表 [[{"label","bbox","confidence"},...], ...]
            gt_boxes_all: 每张图的GT列表 [[{"label","bbox"},...], ...]
            iou_thresholds: IoU 阈值列表

        Returns:
            {
                "mAP": {iou: float},
                "per_class_ap": {iou: {class: float}},
                "total_gt": int,
                "total_pred": int,
            }
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]

        # 收集所有类别
        all_classes = set()
        for gt_boxes in gt_boxes_all:
            for gt in gt_boxes:
                all_classes.add(gt["label"])
        all_classes = sorted(all_classes)

        total_gt = sum(len(g) for g in gt_boxes_all)
        total_pred = sum(len(p) for p in pred_boxes_all)

        result = {
            "mAP": {},
            "per_class_ap": {},
            "total_gt": total_gt,
            "total_pred": total_pred,
        }

        for iou_thr in iou_thresholds:
            class_aps = {}
            for cls in all_classes:
                ap = _compute_ap_for_class(
                    cls, pred_boxes_all, gt_boxes_all, iou_thr
                )
                class_aps[cls] = round(ap, 4)

            mean_ap = np.mean(list(class_aps.values())) if class_aps else 0.0
            result["mAP"][iou_thr] = round(float(mean_ap), 4)
            result["per_class_ap"][iou_thr] = class_aps

        return result

    @staticmethod
    def disagreement_analysis(results: list[dict]) -> dict:
        """
        分歧分析：输入质检结果列表，输出错误模式统计。

        Args:
            results: QC 结果列表，每条含 human_label, engine_label, is_consistent, escalated

        Returns:
            {
                "top_error_patterns": [
                    {"pattern": "yawning→normal", "count": 15, "ratio": 0.03},
                    ...
                ],
                "class_confusion": {
                    "yawning": {"normal": 5, "fatigue": 2},
                    ...
                },
                "escalation_stats": {
                    "total_escalated": int,
                    "escalation_corrected": int,     # VLM 纠正了多少
                    "correction_rate": float,
                }
            }
        """
        inconsistent = [r for r in results if not r.get("is_consistent", True)]
        total = len(results)

        # 错误模式
        pattern_counter = Counter()
        for r in inconsistent:
            pattern = f"{r['human_label']}→{r['engine_label']}"
            pattern_counter[pattern] += 1

        top_patterns = [
            {
                "pattern": p,
                "count": c,
                "ratio": round(c / total, 4) if total > 0 else 0,
            }
            for p, c in pattern_counter.most_common(20)
        ]

        # 类别混淆分布
        class_confusion = defaultdict(lambda: defaultdict(int))
        for r in inconsistent:
            class_confusion[r["human_label"]][r["engine_label"]] += 1
        class_confusion = {k: dict(v) for k, v in class_confusion.items()}

        # 升级统计
        escalated = [r for r in results if r.get("escalated", False)]
        escalated_and_consistent = [r for r in escalated if r.get("is_consistent", False)]

        escalation_stats = {
            "total_escalated": len(escalated),
            "escalation_corrected": len(escalated_and_consistent),
            "correction_rate": (
                round(len(escalated_and_consistent) / len(escalated), 4)
                if escalated else 0
            ),
        }

        return {
            "top_error_patterns": top_patterns,
            "class_confusion": class_confusion,
            "escalation_stats": escalation_stats,
        }

    @staticmethod
    def plot_confusion_matrix(cm: list[list[int]], class_names: list[str],
                               output_path: Optional[str] = None):
        """混淆矩阵热力图"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
        cm_array = np.array(cm)

        im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 在格子中显示数字
        thresh = cm_array.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm_array[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm_array[i, j] > thresh else "black")

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"混淆矩阵已保存: {output_path}")
        plt.close(fig)
        return fig


def _compute_ap_for_class(cls: str, pred_boxes_all: list[list[dict]],
                           gt_boxes_all: list[list[dict]],
                           iou_threshold: float) -> float:
    """计算单个类别的 AP（Average Precision）"""
    # 收集该类别的所有预测和 GT
    all_preds = []  # [(confidence, image_idx, pred_idx)]
    gt_count = 0
    gt_per_image = {}  # image_idx -> [{"bbox", "matched"}]

    for img_idx, (preds, gts) in enumerate(zip(pred_boxes_all, gt_boxes_all)):
        cls_gts = [g for g in gts if g["label"] == cls]
        gt_per_image[img_idx] = [{"bbox": g["bbox"], "matched": False} for g in cls_gts]
        gt_count += len(cls_gts)

        for pred_idx, p in enumerate(preds):
            if p["label"] == cls:
                conf = p.get("confidence", 0.0) or 0.0
                all_preds.append((conf, img_idx, pred_idx, p["bbox"]))

    if gt_count == 0:
        return 0.0

    # 按 confidence 降序排列
    all_preds.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, (conf, img_idx, pred_idx, pred_bbox) in enumerate(all_preds):
        gts = gt_per_image.get(img_idx, [])
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            iou = _compute_iou(pred_bbox, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0 and not gts[best_gt_idx]["matched"]:
            tp[i] = 1
            gts[best_gt_idx]["matched"] = True
        else:
            fp[i] = 1

    # 计算 precision-recall 曲线
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / gt_count
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # AP (11-point interpolation)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_recall = precision[recall >= t]
        if len(precisions_at_recall) > 0:
            ap += np.max(precisions_at_recall)
    ap /= 11.0

    return ap


def _compute_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

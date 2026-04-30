"""
质检 Tab。

功能：
- 选择图片目录、标注目录（分类=文件夹/CSV；检测=VOC XML）、输出目录
- 双引擎升级策略：从引擎面板拿 vlm 引擎 + 升级阈值
- 运行质检流水线，输出 HTML/CSV 报告、错误 case
- 展示待复核样本（含 VLM 理由）
"""
import os
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QComboBox, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QCheckBox, QDoubleSpinBox, QSpinBox,
)

from src.engine.base import Capability
from src.gui.threads.worker import UnifiedWorker
from src.gui.widgets._category_input import make_category_input
from src.gui.widgets.result_browser import ResultBrowser
from src.io.classification_io import relabel_classification
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityCheckTab(QWidget):
    """质检 Tab"""

    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._primary_engine = None
        self._vlm_engine = None
        self._worker: Optional[UnifiedWorker] = None
        self._last_result: Optional[dict] = None
        # 记忆标注目录 + 类别，供人工标签修改定位文件
        self._last_annotation_dir: Optional[str] = None
        self._last_categories: list[str] = []
        self._build_ui()

        self._browser.label_changed.connect(self._on_label_changed)

    # -------- public --------

    def set_primary_engine(self, engine):
        self._primary_engine = engine
        self._refresh_engine_status()

    def set_vlm_engine(self, engine):
        self._vlm_engine = engine
        self._refresh_engine_status()

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_escalation_group())
        layout.addLayout(self._build_control_row())
        layout.addWidget(self._build_result_group(), stretch=1)

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("配置")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._task_cb = QComboBox()
        self._task_cb.addItems(["classification", "detection"])
        self._task_cb.currentTextChanged.connect(self._on_task_changed)
        form.addRow("任务类型:", self._task_cb)

        self._image_dir_edit, img_row = self._path_picker("选择图片目录", is_dir=True)
        form.addRow("图片目录:", img_row)

        self._annotation_edit, ann_row = self._path_picker(
            "选择标注目录（分类=类别文件夹；检测=VOC XML 目录）", is_dir=True
        )
        form.addRow("标注目录:", ann_row)

        self._output_dir_edit, out_row = self._path_picker("选择输出目录", is_dir=True)
        form.addRow("输出目录:", out_row)

        self._categories_edit, self._categories_row = make_category_input(
            placeholder="从 txt 载入类别，或手动逗号分隔",
            file_dialog_caption="选择分类类别 txt（每行一个）",
            parent=self,
        )
        form.addRow("类别（分类）:", self._categories_row)

        self._targets_edit, self._targets_row = make_category_input(
            placeholder="从 txt 载入目标，或手动逗号分隔",
            file_dialog_caption="选择检测目标 txt（每行一个）",
            parent=self,
        )
        self._targets_row.setEnabled(False)
        form.addRow("目标（检测）:", self._targets_row)

        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.0, 1.0)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.5)
        self._iou_spin.setEnabled(False)
        form.addRow("IoU 阈值（检测）:", self._iou_spin)

        return gb

    def _build_escalation_group(self) -> QGroupBox:
        gb = QGroupBox("低置信度 VLM 升级策略")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._escalation_cb = QCheckBox("启用升级（需在左侧面板加载 VLM 辅助引擎）")
        self._escalation_cb.setChecked(True)
        form.addRow(self._escalation_cb)

        self._escalation_thr_spin = QDoubleSpinBox()
        self._escalation_thr_spin.setRange(0.0, 1.0)
        self._escalation_thr_spin.setSingleStep(0.05)
        self._escalation_thr_spin.setValue(0.8)
        form.addRow("升级阈值:", self._escalation_thr_spin)

        return gb

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._start_btn = QPushButton("▶ 开始质检")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self._stop_btn)

        self._export_btn = QPushButton("📊 导出报告")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        row.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setFormat("%v / %m")
        row.addWidget(self._progress, stretch=1)

        self._engine_status_lbl = QLabel("主: 未加载 | VLM: 未加载")
        self._engine_status_lbl.setStyleSheet("color: #888;")
        row.addWidget(self._engine_status_lbl)

        return row

    def _build_result_group(self) -> QGroupBox:
        gb = QGroupBox("待复核样本")
        v = QVBoxLayout(gb)

        self._summary_lbl = QLabel("未开始")
        self._summary_lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
        v.addWidget(self._summary_lbl)

        self._browser = ResultBrowser()
        v.addWidget(self._browser, stretch=1)

        return gb

    def _path_picker(self, caption: str, is_dir: bool = True):
        edit = QLineEdit()
        btn = QPushButton("...")
        btn.setFixedWidth(32)

        def _browse():
            if is_dir:
                path = QFileDialog.getExistingDirectory(self, caption, edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, caption, edit.text())
            if path:
                edit.setText(path)

        btn.clicked.connect(_browse)
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        row.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setLayout(row)
        return edit, container

    # -------- slots --------

    def _on_task_changed(self, task: str):
        is_cls = task == "classification"
        self._categories_row.setEnabled(is_cls)
        self._targets_row.setEnabled(not is_cls)
        self._iou_spin.setEnabled(not is_cls)

    def _refresh_engine_status(self):
        p = "已加载" if self._primary_engine is not None else "未加载"
        v = "已加载" if self._vlm_engine is not None else "未加载"
        self._engine_status_lbl.setText(f"主: {p} | VLM: {v}")
        color = "#4CAF50" if self._primary_engine is not None else "#888"
        self._engine_status_lbl.setStyleSheet(f"color: {color};")
        self._start_btn.setEnabled(self._primary_engine is not None)

    def _validate(self) -> Optional[dict]:
        if self._primary_engine is None:
            QMessageBox.warning(self, "提示", "请先加载主引擎")
            return None

        image_dir = self._image_dir_edit.text().strip()
        annotation_dir = self._annotation_edit.text().strip()
        output_dir = self._output_dir_edit.text().strip()

        if not image_dir or not os.path.isdir(image_dir):
            QMessageBox.warning(self, "提示", "图片目录无效")
            return None
        if not annotation_dir or not (
            os.path.isdir(annotation_dir) or os.path.isfile(annotation_dir)
        ):
            QMessageBox.warning(self, "提示", "标注目录/文件无效")
            return None
        if not output_dir:
            QMessageBox.warning(self, "提示", "请指定输出目录")
            return None

        task = self._task_cb.currentText()
        if task == "classification":
            cats = [c.strip() for c in self._categories_edit.text().split(",") if c.strip()]
            if not cats:
                QMessageBox.warning(self, "提示", "请输入分类类别")
                return None
            return {
                "task": task,
                "image_dir": image_dir,
                "annotation_dir": annotation_dir,
                "output_dir": output_dir,
                "categories": cats,
            }
        else:
            if not self._primary_engine.supports(Capability.DETECT):
                QMessageBox.warning(self, "提示", "当前主引擎不支持检测")
                return None
            targets = [t.strip() for t in self._targets_edit.text().split(",") if t.strip()]
            if not targets:
                QMessageBox.warning(self, "提示", "请输入检测目标")
                return None
            return {
                "task": task,
                "image_dir": image_dir,
                "annotation_dir": annotation_dir,
                "output_dir": output_dir,
                "targets": targets,
                "iou_threshold": self._iou_spin.value(),
            }

    def _on_start(self):
        args = self._validate()
        if args is None:
            return

        from src.pipeline.qualitycheck import QualityCheckPipeline

        escalation_enabled = self._escalation_cb.isChecked() and self._vlm_engine is not None
        if self._escalation_cb.isChecked() and self._vlm_engine is None:
            ret = QMessageBox.question(
                self, "提示",
                "已勾选升级策略但未加载 VLM 辅助引擎。\n继续执行将跳过升级，是否继续？",
            )
            if ret != QMessageBox.Yes:
                return

        qc_cfg = {
            "escalation_enabled": escalation_enabled,
            "escalation_threshold": self._escalation_thr_spin.value(),
            "ui_update_interval": 10,
            "num_io_workers": 4,
        }
        pipeline = QualityCheckPipeline(
            self._primary_engine, qc_cfg,
            vlm_engine=self._vlm_engine if escalation_enabled else None,
        )

        if args["task"] == "classification":
            def task_fn(cb):
                return pipeline.run_classification_qc(
                    args["image_dir"], args["annotation_dir"],
                    args["output_dir"], args["categories"], cb,
                )
        else:
            def task_fn(cb):
                return pipeline.run_detection_qc(
                    args["image_dir"], args["annotation_dir"],
                    args["output_dir"], args["targets"],
                    iou_threshold=args["iou_threshold"],
                    progress_callback=cb,
                )

        self._current_task = args["task"]
        self._current_output_dir = args["output_dir"]
        # 记忆标注目录与类别，供人工标签修改定位文件
        self._last_annotation_dir = args["annotation_dir"]
        self._last_categories = args.get("categories", [])
        self._worker = UnifiedWorker(task_fn, ui_update_interval=10)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._set_running(True)
        self._summary_lbl.setText("运行中…")
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self._summary_lbl.setText("停止中…")

    def _on_progress(self, current: int, total: int):
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(current)

    def _on_finished(self, result: dict):
        self._set_running(False)
        self._last_result = result
        self._export_btn.setEnabled(self._current_task == "classification")

        if self._current_task == "classification":
            total = result.get("total_checked", 0)
            reviewed = result.get("review_count", 0)
            ratio = result.get("review_ratio", 0)
            escalated = result.get("escalated_count", 0)
            self._summary_lbl.setText(
                f"完成: 共 {total} 张 | 异议率 {ratio*100:.1f}% ({reviewed} 条待复核) | "
                f"升级 VLM: {escalated}"
            )
            self._show_review_samples(result.get("review_samples", []))
            # 分类质检完成后开启人工标签编辑
            self._browser.set_editable_categories(self._last_categories)
        else:
            self._summary_lbl.setText(
                f"完成: GT {result.get('total_gt_boxes', 0)} / 预测 {result.get('total_pred_boxes', 0)} | "
                f"匹配 {result.get('matched_count', 0)} | "
                f"漏 {result.get('miss_count', 0)} | "
                f"多 {result.get('extra_count', 0)} | "
                f"类别错 {result.get('label_mismatch_count', 0)}"
            )
            self._show_detection_issues(result.get("issues", []))
            self._browser.set_editable_categories(None)

    def _on_error(self, msg: str):
        self._set_running(False)
        self._summary_lbl.setText(f"失败: {msg}")
        QMessageBox.critical(self, "质检失败", msg)

    def _set_running(self, running: bool):
        self._start_btn.setEnabled(not running and self._primary_engine is not None)
        self._stop_btn.setEnabled(running)
        for w in (self._task_cb, self._image_dir_edit, self._annotation_edit,
                   self._output_dir_edit, self._categories_row, self._targets_row,
                   self._iou_spin, self._escalation_cb, self._escalation_thr_spin):
            w.setEnabled(not running)
        if not running:
            self._on_task_changed(self._task_cb.currentText())

    def _show_review_samples(self, samples: list[dict]):
        items = []
        for s in samples:
            human = s.get("human_label", "?")
            engine = s.get("engine_label", "?")
            conf = s.get("confidence")
            escalated = s.get("escalated", False)
            conf_str = f" ({conf:.2f})" if conf is not None else ""
            tag = " [VLM]" if escalated else ""
            items.append({
                "image_path": s.get("image_path", ""),
                "title": os.path.basename(s.get("image_path", "")),
                "subtitle": f"人工: {human} → 引擎: {engine}{conf_str}{tag}",
                "current_label": human,  # 编辑的是人工标签
                "meta": {
                    "human_label": human,
                    "engine_label": engine,
                    "confidence": conf,
                    "escalated": escalated,
                    "vlm_reason": s.get("vlm_reason", ""),
                },
            })
        self._browser.set_items(items)

    def _show_detection_issues(self, issues: list[dict]):
        items = []
        for iss in issues:
            t = iss.get("type", "?")
            img = iss.get("image_path", "")
            meta = {k: v for k, v in iss.items() if k != "image_path"}

            if t == "miss":
                sub = f"漏标: {iss.get('gt_label', '?')}"
                dets = [{"label": iss.get("gt_label"), "bbox": iss.get("gt_bbox")}]
            elif t == "extra":
                sub = f"多标: {iss.get('pred_label', '?')}"
                dets = [{"label": iss.get("pred_label"),
                         "bbox": iss.get("pred_bbox"),
                         "confidence": iss.get("confidence")}]
            elif t == "label_mismatch":
                sub = f"类别错: {iss.get('gt_label')} → {iss.get('pred_label')}"
                dets = [
                    {"label": f"GT:{iss.get('gt_label')}", "bbox": iss.get("gt_bbox")},
                    {"label": f"PRED:{iss.get('pred_label')}", "bbox": iss.get("pred_bbox")},
                ]
            else:
                sub = t
                dets = None

            items.append({
                "image_path": img,
                "title": os.path.basename(img),
                "subtitle": sub,
                "meta": meta,
                "detections": dets,
            })
        self._browser.set_items(items)

    def _on_export(self):
        if not self._last_result or self._current_task != "classification":
            return
        from src.report.generator import ReportGenerator

        out_dir = self._current_output_dir
        os.makedirs(out_dir, exist_ok=True)
        reporter = ReportGenerator()

        try:
            html_path = os.path.join(out_dir, "qc_report.html")
            csv_path = os.path.join(out_dir, "qc_report.csv")
            reporter.generate_qc_html_report(self._last_result, html_path)
            reporter.generate_qc_csv_report(self._last_result, csv_path)
            reporter.export_error_cases(
                self._last_result, os.path.join(out_dir, "error_cases"),
            )
            QMessageBox.information(
                self, "导出完成",
                f"报告已生成:\n{html_path}\n{csv_path}\n错误 case 已导出到 error_cases/",
            )
        except Exception as e:
            logger.exception("导出报告失败")
            QMessageBox.critical(self, "导出失败", str(e))

    # -------- 人工标签修改 --------

    def _on_label_changed(self, item: dict, new_label: str):
        """修改人工标注：在 annotation_dir 下移动文件 + 刷新显示"""
        old_label = item.get("current_label")
        if not old_label:
            QMessageBox.warning(self, "提示", "该项无当前人工标签")
            return
        if old_label == new_label:
            return
        if not self._last_annotation_dir or not os.path.isdir(self._last_annotation_dir):
            QMessageBox.warning(
                self, "提示", "标注目录无效，无法修改（仅支持文件夹式标注）"
            )
            return

        filename = os.path.basename(item.get("image_path", ""))
        if not filename:
            return

        ret = QMessageBox.question(
            self, "确认修改人工标签",
            f"将 <b>{filename}</b> 的人工标注<br>"
            f"从 <b style='color:#FF9800'>{old_label}</b> 改为 "
            f"<b style='color:#4CAF50'>{new_label}</b>?<br><br>"
            f"会移动 <code>{self._last_annotation_dir}/{old_label}/</code> 下的文件。",
        )
        if ret != QMessageBox.Yes:
            return

        try:
            relabel_classification(
                self._last_annotation_dir, filename,
                old_label, new_label, csv_path=None,
            )
        except FileExistsError:
            QMessageBox.critical(
                self, "冲突",
                f"标注目录下 '{new_label}/' 已有同名文件：{filename}\n"
                f"请先手动处理。",
            )
            return
        except FileNotFoundError:
            QMessageBox.critical(
                self, "未找到",
                f"{self._last_annotation_dir}/{old_label}/{filename} 不存在。\n"
                f"标注可能是 CSV 形式，本功能暂仅支持文件夹结构。",
            )
            return
        except Exception as e:
            logger.exception("人工标签修改失败")
            QMessageBox.critical(self, "修改失败", str(e))
            return

        # 重新计算是否与引擎一致
        engine_label = item.get("meta", {}).get("engine_label", "")
        now_consistent = (new_label == engine_label)
        conf = item.get("meta", {}).get("confidence")
        conf_str = f" ({conf:.2f})" if conf is not None else ""
        status_tag = " ✓一致" if now_consistent else ""
        self._browser.update_current_item(
            new_label=new_label,
            new_subtitle=(
                f"人工: {new_label} → 引擎: {engine_label}{conf_str}"
                f" [人工已修正]{status_tag}"
            ),
            new_meta_patch={
                "human_label": new_label,
                "human_edited": True,
                "consistent_after_edit": now_consistent,
            },
        )
        self._summary_lbl.setText(
            f"✓ 已将 {filename} 的人工标签从 {old_label} 改为 {new_label}"
        )

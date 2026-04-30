"""
评估 Tab。

流程：
- 输入图片目录 + GT 标注 + 输出目录 + 类别
- 跑引擎推理（Worker 线程）
- 对比 GT 计算指标
- 展示指标 + 混淆矩阵图（分类）
"""
import json
import os
import tempfile
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QComboBox, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from src.engine.base import Capability
from src.gui.threads.worker import UnifiedWorker
from src.gui.widgets._category_input import make_category_input
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkTab(QWidget):
    """评估 Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._primary_engine = None
        self._worker: Optional[UnifiedWorker] = None
        self._last_report: Optional[dict] = None
        self._build_ui()

    def set_primary_engine(self, engine):
        self._primary_engine = engine
        ok = engine is not None
        self._start_btn.setEnabled(ok)
        self._engine_status_lbl.setText("引擎: 已加载" if ok else "引擎: 未加载")
        self._engine_status_lbl.setStyleSheet("color: #4CAF50;" if ok else "color: #888;")

    def set_vlm_engine(self, engine):
        """本 Tab 不用 VLM 辅助"""

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_config_group())
        layout.addLayout(self._build_control_row())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_metrics_group())
        splitter.addWidget(self._build_matrix_group())
        splitter.setSizes([600, 600])
        layout.addWidget(splitter, stretch=1)

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

        self._gt_dir_edit, gt_row = self._path_picker(
            "选择真值目录（分类=类别文件夹或CSV；检测=VOC XML）", is_dir=True,
        )
        form.addRow("真值目录:", gt_row)

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

        return gb

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始评估")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        self._export_btn = QPushButton("💾 导出 JSON")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)

        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        row.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setFormat("%v / %m")
        row.addWidget(self._progress, stretch=1)

        self._engine_status_lbl = QLabel("引擎: 未加载")
        self._engine_status_lbl.setStyleSheet("color: #888;")
        row.addWidget(self._engine_status_lbl)
        return row

    def _build_metrics_group(self) -> QGroupBox:
        gb = QGroupBox("评估指标")
        v = QVBoxLayout(gb)

        self._summary_lbl = QLabel("未开始")
        self._summary_lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
        v.addWidget(self._summary_lbl)

        self._metrics_table = QTableWidget(0, 5)
        self._metrics_table.setHorizontalHeaderLabels(
            ["类别", "Precision", "Recall", "F1", "Support"]
        )
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v.addWidget(self._metrics_table)

        return gb

    def _build_matrix_group(self) -> QGroupBox:
        gb = QGroupBox("混淆矩阵")
        v = QVBoxLayout(gb)

        self._matrix_lbl = QLabel("尚无数据")
        self._matrix_lbl.setAlignment(Qt.AlignCenter)
        self._matrix_lbl.setMinimumSize(300, 300)
        self._matrix_lbl.setStyleSheet(
            "background: #1E1E1E; color: #666; border: 1px solid #3C3C3C;"
        )
        v.addWidget(self._matrix_lbl)

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

    def _validate(self) -> Optional[dict]:
        if self._primary_engine is None:
            QMessageBox.warning(self, "提示", "请先加载主引擎")
            return None
        image_dir = self._image_dir_edit.text().strip()
        gt_dir = self._gt_dir_edit.text().strip()
        output_dir = self._output_dir_edit.text().strip()
        if not image_dir or not os.path.isdir(image_dir):
            QMessageBox.warning(self, "提示", "图片目录无效")
            return None
        if not gt_dir or not os.path.exists(gt_dir):
            QMessageBox.warning(self, "提示", "真值目录无效")
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
            return {"task": task, "image_dir": image_dir, "gt_dir": gt_dir,
                    "output_dir": output_dir, "categories": cats}
        else:
            if not self._primary_engine.supports(Capability.DETECT):
                QMessageBox.warning(self, "提示", "当前主引擎不支持检测")
                return None
            targets = [t.strip() for t in self._targets_edit.text().split(",") if t.strip()]
            if not targets:
                QMessageBox.warning(self, "提示", "请输入检测目标")
                return None
            return {"task": task, "image_dir": image_dir, "gt_dir": gt_dir,
                    "output_dir": output_dir, "targets": targets}

    def _on_start(self):
        args = self._validate()
        if args is None:
            return

        from src.io.image_loader import scan_images
        from src.io.classification_io import (
            read_classification_folders, read_classification_csv,
        )
        from src.io.voc_xml import read_voc_annotations
        from src.utils.metrics import Evaluator

        engine = self._primary_engine
        self._current_task = args["task"]
        self._current_output_dir = args["output_dir"]
        os.makedirs(args["output_dir"], exist_ok=True)

        if args["task"] == "classification":
            cats = args["categories"]
            if os.path.isfile(args["gt_dir"]) and args["gt_dir"].endswith(".csv"):
                gt_map = read_classification_csv(args["gt_dir"])
            else:
                gt_map = read_classification_folders(args["gt_dir"])
            images = scan_images(args["image_dir"])

            def task_fn(cb):
                y_true, y_pred = [], []
                for i, img_path in enumerate(images):
                    fn = os.path.basename(img_path)
                    if fn in gt_map:
                        pred = engine.classify(img_path, cats)
                        y_true.append(gt_map[fn])
                        y_pred.append(pred.predicted_class)
                    cb(i + 1, len(images), None)
                if not y_true:
                    return {"error": "未匹配到任何 GT 样本", "task": "classification"}
                report = Evaluator.classification_report(y_true, y_pred, cats)
                report["task"] = "classification"
                return report
        else:
            targets = args["targets"]
            gt_annotations = read_voc_annotations(args["gt_dir"])
            images = scan_images(args["image_dir"])

            def task_fn(cb):
                pred_all, gt_all = [], []
                for i, img_path in enumerate(images):
                    fn = os.path.basename(img_path)
                    stem = os.path.splitext(fn)[0]
                    gt = gt_annotations.get(fn) or gt_annotations.get(stem)
                    if gt is not None:
                        det = engine.detect(img_path, targets)
                        pred_all.append(det.detections)
                        gt_all.append(gt["objects"])
                    cb(i + 1, len(images), None)
                if not gt_all:
                    return {"error": "未匹配到任何 GT", "task": "detection"}
                report = Evaluator.detection_report(pred_all, gt_all, iou_thresholds=[0.5])
                report["task"] = "detection"
                return report

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

    def _on_finished(self, report: dict):
        self._set_running(False)
        if "error" in report:
            self._summary_lbl.setText(f"失败: {report['error']}")
            QMessageBox.warning(self, "评估失败", report["error"])
            return

        self._last_report = report
        self._export_btn.setEnabled(True)

        if report.get("task") == "classification":
            self._summary_lbl.setText(
                f"Accuracy: {report['accuracy']:.4f} | "
                f"Macro-F1: {report['macro_f1']:.4f} | "
                f"Weighted-F1: {report['weighted_f1']:.4f}"
            )
            self._fill_classification_table(report["per_class"])
            self._render_confusion_matrix(report)
        else:
            self._summary_lbl.setText(
                f"mAP@0.5: {list(report['mAP'].values())[0]:.4f} | "
                f"GT: {report['total_gt']} | 预测: {report['total_pred']}"
            )
            self._fill_detection_table(report["per_class_ap"])
            self._matrix_lbl.setText("检测任务无混淆矩阵")
            self._matrix_lbl.setPixmap(QPixmap())

    def _on_error(self, msg: str):
        self._set_running(False)
        self._summary_lbl.setText(f"失败: {msg}")
        QMessageBox.critical(self, "评估失败", msg)

    def _set_running(self, running: bool):
        self._start_btn.setEnabled(not running and self._primary_engine is not None)
        self._stop_btn.setEnabled(running)
        for w in (self._task_cb, self._image_dir_edit, self._gt_dir_edit,
                   self._output_dir_edit, self._categories_row, self._targets_row):
            w.setEnabled(not running)
        if not running:
            self._on_task_changed(self._task_cb.currentText())

    def _fill_classification_table(self, per_class: dict):
        self._metrics_table.setColumnCount(5)
        self._metrics_table.setHorizontalHeaderLabels(
            ["类别", "Precision", "Recall", "F1", "Support"]
        )
        self._metrics_table.setRowCount(0)
        for name, m in per_class.items():
            row = self._metrics_table.rowCount()
            self._metrics_table.insertRow(row)
            self._metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self._metrics_table.setItem(row, 1, QTableWidgetItem(f"{m['precision']:.4f}"))
            self._metrics_table.setItem(row, 2, QTableWidgetItem(f"{m['recall']:.4f}"))
            self._metrics_table.setItem(row, 3, QTableWidgetItem(f"{m['f1']:.4f}"))
            self._metrics_table.setItem(row, 4, QTableWidgetItem(str(m["support"])))

    def _fill_detection_table(self, per_class_ap: dict):
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["类别", "AP@0.5"])
        self._metrics_table.setRowCount(0)
        aps = list(per_class_ap.values())[0] if per_class_ap else {}
        for name, ap in aps.items():
            row = self._metrics_table.rowCount()
            self._metrics_table.insertRow(row)
            self._metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self._metrics_table.setItem(row, 1, QTableWidgetItem(f"{ap:.4f}"))

    def _render_confusion_matrix(self, report: dict):
        """调用 Evaluator.plot_confusion_matrix 保存 PNG 再加载展示"""
        from src.utils.metrics import Evaluator
        out_path = os.path.join(self._current_output_dir, "confusion_matrix.png")
        try:
            Evaluator.plot_confusion_matrix(
                report["confusion_matrix"], report["class_names"],
                output_path=out_path,
            )
            pix = QPixmap(out_path)
            if pix.isNull():
                self._matrix_lbl.setText("混淆矩阵生成失败")
                return
            scaled = pix.scaled(
                self._matrix_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._matrix_lbl.setPixmap(scaled)
        except Exception as e:
            logger.exception("渲染混淆矩阵失败")
            self._matrix_lbl.setText(f"渲染失败: {e}")

    def _on_export(self):
        if self._last_report is None:
            return
        out_path = os.path.join(self._current_output_dir, "benchmark.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._last_report, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "导出完成", f"报告已保存:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

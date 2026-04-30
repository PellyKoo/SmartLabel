"""
预标注 Tab。

功能：
- 选择图片目录、输出目录、类别/目标
- 任务类型：分类 / 检测
- 后台 Worker 执行流水线，进度条 + 结果浏览
- 运行完成后展示统计信息
"""
import os
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QComboBox, QLabel, QFileDialog,
    QProgressBar, QMessageBox,
)

from src.engine.base import Capability, ClassificationResult, DetectionResult
from src.gui.threads.worker import UnifiedWorker
from src.gui.widgets._category_input import make_category_input
from src.gui.widgets.result_browser import ResultBrowser
from src.io.classification_io import relabel_classification
from src.io.image_loader import scan_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreAnnotateTab(QWidget):
    """预标注 Tab"""

    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._primary_engine = None
        self._worker: Optional[UnifiedWorker] = None
        # 记忆上次运行的参数，供标签修改时定位文件
        self._last_output_dir: Optional[str] = None
        self._last_categories: list[str] = []
        self._last_file_operation: str = "copy"
        self._build_ui()

        # 图片目录变更 → 防抖 300ms 后触发预览
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._do_preview)
        self._image_dir_edit.textChanged.connect(
            lambda _t: self._preview_timer.start(300)
        )
        # 监听标签修改
        self._browser.label_changed.connect(self._on_label_changed)

    # -------- public --------

    def set_primary_engine(self, engine):
        self._primary_engine = engine
        ok = engine is not None
        self._start_btn.setEnabled(ok)
        self._engine_status_lbl.setText("引擎: 已加载" if ok else "引擎: 未加载")
        self._engine_status_lbl.setStyleSheet(
            "color: #4CAF50;" if ok else "color: #888;"
        )

    def set_vlm_engine(self, engine):
        """本 Tab 不使用 VLM 辅助"""

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_config_group())
        layout.addLayout(self._build_control_row())
        layout.addWidget(self._build_result_group(), stretch=1)

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("配置")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        # 任务类型
        self._task_cb = QComboBox()
        self._task_cb.addItems(["classification", "detection"])
        self._task_cb.currentTextChanged.connect(self._on_task_changed)
        form.addRow("任务类型:", self._task_cb)

        # 图片目录
        self._image_dir_edit, img_row = self._path_picker("选择图片目录", is_dir=True)
        form.addRow("图片目录:", img_row)

        # 输出目录
        self._output_dir_edit, out_row = self._path_picker("选择输出目录", is_dir=True)
        form.addRow("输出目录:", out_row)

        # 类别/目标：从 txt 加载（每行一个）
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
        form.addRow("目标（检测）:", self._targets_row)
        self._targets_row.setEnabled(False)

        # 输出格式
        self._output_format_cb = QComboBox()
        self._output_format_cb.addItems(["both", "folder", "csv"])
        form.addRow("输出格式（分类）:", self._output_format_cb)

        return gb

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._start_btn = QPushButton("▶ 开始预标注")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self._stop_btn)

        self._progress = QProgressBar()
        self._progress.setFormat("%v / %m")
        row.addWidget(self._progress, stretch=1)

        self._engine_status_lbl = QLabel("引擎: 未加载")
        self._engine_status_lbl.setStyleSheet("color: #888;")
        row.addWidget(self._engine_status_lbl)

        return row

    def _build_result_group(self) -> QGroupBox:
        gb = QGroupBox("结果预览")
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
        self._output_format_cb.setEnabled(is_cls)

    def _validate(self) -> Optional[dict]:
        if self._primary_engine is None:
            QMessageBox.warning(self, "提示", "请先在左侧引擎面板加载主引擎")
            return None

        image_dir = self._image_dir_edit.text().strip()
        output_dir = self._output_dir_edit.text().strip()
        if not image_dir or not os.path.isdir(image_dir):
            QMessageBox.warning(self, "提示", "图片目录无效")
            return None
        if not output_dir:
            QMessageBox.warning(self, "提示", "请指定输出目录")
            return None

        task = self._task_cb.currentText()
        if task == "classification":
            cats = [
                c.strip() for c in self._categories_edit.text().split(",") if c.strip()
            ]
            if not cats:
                QMessageBox.warning(self, "提示", "请输入分类类别")
                return None
            return {
                "task": task,
                "image_dir": image_dir,
                "output_dir": output_dir,
                "categories": cats,
                "output_format": self._output_format_cb.currentText(),
            }
        else:
            if not self._primary_engine.supports(Capability.DETECT):
                QMessageBox.warning(self, "提示", "当前主引擎不支持检测")
                return None
            targets = [
                t.strip() for t in self._targets_edit.text().split(",") if t.strip()
            ]
            if not targets:
                QMessageBox.warning(self, "提示", "请输入检测目标")
                return None
            return {
                "task": task,
                "image_dir": image_dir,
                "output_dir": output_dir,
                "targets": targets,
            }

    def _on_start(self):
        args = self._validate()
        if args is None:
            return

        # 记忆参数，供后续标签修改定位文件
        self._last_output_dir = args["output_dir"]
        self._last_categories = args.get("categories", [])
        self._last_file_operation = "copy"

        # 延迟导入，避免启动时加载
        from src.pipeline.preannotate import PreAnnotatePipeline

        pipeline_cfg = {
            "output_format": args.get("output_format", "both"),
            "file_operation": self._last_file_operation,
            "num_io_workers": 4,
            "ui_update_interval": 10,
        }
        pipeline = PreAnnotatePipeline(self._primary_engine, pipeline_cfg)

        # 构造 task_fn
        if args["task"] == "classification":
            def task_fn(cb):
                return pipeline.run_classification(
                    args["image_dir"], args["output_dir"],
                    args["categories"], cb,
                )
        else:
            def task_fn(cb):
                return pipeline.run_detection(
                    args["image_dir"], args["output_dir"],
                    args["targets"], cb,
                )

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
        task = self._task_cb.currentText()
        if task == "classification":
            self._summary_lbl.setText(
                f"完成: 共 {result.get('total', 0)} 张 | "
                f"各类别: {result.get('category_counts', {})} | "
                f"不确定: {result.get('uncertain_count', 0)}"
            )
            self._show_classification_results(result.get("results", []))
            # 分类完成后启用标签编辑（检测暂不支持）
            self._browser.set_editable_categories(self._last_categories)
        else:
            self._summary_lbl.setText(
                f"完成: 共 {result.get('total', 0)} 张 | "
                f"总检出 {result.get('total_detections', 0)} 个目标 | "
                f"各类: {result.get('label_counts', {})}"
            )
            self._show_detection_results(result.get("results", []))
            self._browser.set_editable_categories(None)

    def _on_error(self, msg: str):
        self._set_running(False)
        self._summary_lbl.setText(f"失败: {msg}")
        QMessageBox.critical(self, "预标注失败", msg)

    def _set_running(self, running: bool):
        self._start_btn.setEnabled(not running and self._primary_engine is not None)
        self._stop_btn.setEnabled(running)
        # 运行中禁用配置编辑
        for w in (self._task_cb, self._image_dir_edit, self._output_dir_edit,
                   self._categories_row, self._targets_row, self._output_format_cb):
            w.setEnabled(not running)
        if not running:
            # 恢复分类/检测的互斥禁用
            self._on_task_changed(self._task_cb.currentText())

    def _show_classification_results(self, results: list):
        items = []
        for r in results:
            if not isinstance(r, ClassificationResult):
                continue
            items.append({
                "image_path": r.image_path,
                "title": os.path.basename(r.image_path),
                "subtitle": f"→ {r.predicted_class}"
                             + (f" ({r.confidence:.2f})" if r.confidence is not None else ""),
                "current_label": r.predicted_class,
                "meta": {
                    "predicted_class": r.predicted_class,
                    "confidence": r.confidence,
                    "is_uncertain": r.is_uncertain,
                    "raw_output": r.raw_output[:200],
                },
            })
        self._browser.set_items(items)

    def _show_detection_results(self, results: list):
        items = []
        for r in results:
            if not isinstance(r, DetectionResult):
                continue
            dets = r.detections or []
            labels = [d.get("label", "?") for d in dets]
            items.append({
                "image_path": r.image_path,
                "title": os.path.basename(r.image_path),
                "subtitle": f"{len(dets)} 个检出: {', '.join(labels[:3])}"
                             + ("..." if len(labels) > 3 else ""),
                "detections": dets,
                "meta": {
                    "num_detections": len(dets),
                    "labels": labels,
                },
            })
        self._browser.set_items(items)

    # -------- 待标注预览 --------

    def _do_preview(self):
        """防抖触发：图片目录合法则填充"待标注"预览（不加载缩略图以免阻塞）"""
        # 运行中不刷新预览，避免覆盖进度显示
        if self._worker is not None and self._worker.isRunning():
            return
        path = self._image_dir_edit.text().strip()
        if not path or not os.path.isdir(path):
            return
        try:
            images = scan_images(path)
        except Exception as e:
            logger.warning(f"扫描图片目录失败: {e}")
            return

        if not images:
            self._browser.clear()
            self._summary_lbl.setText("目录中无图片")
            return

        items = [
            {
                "image_path": img,
                "title": os.path.basename(img),
                "subtitle": "待预标注",
                "meta": {"status": "pending"},
            }
            for img in images
        ]
        # 预览阶段不开启编辑；跳过缩略图生成（仅显示文件名列表），避免大目录卡顿
        self._browser.set_editable_categories(None)
        self._browser.set_items(items, load_thumbnails=False)
        self._summary_lbl.setText(f"待预标注: {len(images)} 张")

    # -------- 标签修改 --------

    def _on_label_changed(self, item: dict, new_label: str):
        """用户点击"修改" → 确认后落盘"""
        old_label = item.get("current_label")
        if not old_label:
            QMessageBox.warning(self, "提示", "该项无当前标签信息")
            return
        if old_label == new_label:
            return
        if not self._last_output_dir or not os.path.isdir(self._last_output_dir):
            QMessageBox.warning(self, "提示", "输出目录无效，无法修改")
            return

        filename = os.path.basename(item.get("image_path", ""))
        if not filename:
            return

        ret = QMessageBox.question(
            self, "确认修改",
            f"将 <b>{filename}</b><br>"
            f"从 <b style='color:#FF9800'>{old_label}</b> 改为 "
            f"<b style='color:#4CAF50'>{new_label}</b>?<br><br>"
            f"会立即移动输出目录下的文件并更新 CSV。",
        )
        if ret != QMessageBox.Yes:
            return

        csv_path = os.path.join(self._last_output_dir, "classification_results.csv")
        csv_path = csv_path if os.path.isfile(csv_path) else None

        try:
            relabel_classification(
                self._last_output_dir, filename,
                old_label, new_label, csv_path=csv_path,
            )
        except FileExistsError:
            QMessageBox.critical(
                self, "冲突",
                f"目标类别 '{new_label}' 下已存在同名文件：{filename}\n"
                f"请先手动处理冲突后再尝试。",
            )
            return
        except Exception as e:
            logger.exception("标签修改失败")
            QMessageBox.critical(self, "修改失败", str(e))
            return

        # 更新 browser 显示
        conf = item.get("meta", {}).get("confidence")
        conf_str = f" ({conf:.2f})" if conf is not None else ""
        self._browser.update_current_item(
            new_label=new_label,
            new_subtitle=f"→ {new_label}{conf_str} [已修改]",
            new_meta_patch={"predicted_class": new_label, "edited": True},
        )
        self._summary_lbl.setText(
            f"✓ 已将 {filename} 从 {old_label} 改为 {new_label}"
        )

"""
引擎配置面板：左侧主引擎 + 可选 VLM 辅助引擎 + 采样设置。

职责：
- 读取 configs/default.yaml 作为初始值
- 允许用户切换引擎类型、选择模型路径
- 点击"加载/释放"按钮异步触发引擎 load()/unload()
- 通过 Qt 信号广播引擎状态（engine_loaded / engine_unloaded）
- 其他 Tab 通过 get_primary_engine() / get_vlm_engine() 拿引擎实例
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QLineEdit, QPushButton, QCheckBox, QDoubleSpinBox,
    QSpinBox, QFileDialog, QFormLayout, QSizePolicy,
)

from src.engine.base import BaseEngine
from src.engine.engine_factory import EngineFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


# ---------------- 后台加载引擎 ----------------

class _EngineLoadThread(QThread):
    """异步加载/释放引擎，避免阻塞 UI"""
    done = pyqtSignal(object, str)  # (engine_or_None, error_msg)

    def __init__(self, config: dict, action: str = "load", engine=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._action = action
        self._engine = engine

    def run(self):
        try:
            if self._action == "load":
                engine = EngineFactory.create(self._config)
                engine.load()
                self.done.emit(engine, "")
            elif self._action == "unload":
                if self._engine is not None:
                    self._engine.unload()
                self.done.emit(None, "")
        except Exception as e:
            logger.exception("引擎加载/释放失败")
            self.done.emit(None, f"{type(e).__name__}: {e}")


# ---------------- 引擎面板 ----------------

class EnginePanel(QWidget):
    """引擎配置面板"""

    primary_engine_changed = pyqtSignal(object)   # BaseEngine | None
    vlm_engine_changed = pyqtSignal(object)       # BaseEngine | None
    status_message = pyqtSignal(str)              # 文本消息，供主窗口状态栏

    def __init__(self, parent=None):
        super().__init__(parent)
        self._primary_engine: Optional[BaseEngine] = None
        self._vlm_engine: Optional[BaseEngine] = None
        self._load_thread: Optional[_EngineLoadThread] = None

        self._default_cfg = self._load_default_config()

        self._build_ui()
        self._apply_defaults()

    # -------- public --------

    def get_primary_engine(self) -> Optional[BaseEngine]:
        return self._primary_engine

    def get_vlm_engine(self) -> Optional[BaseEngine]:
        return self._vlm_engine

    def get_sampling_config(self) -> dict:
        """返回 VLM 多次采样配置"""
        return {
            "multi_sample": self._multi_sample_cb.isChecked(),
            "sample_count": self._sample_count_spin.value(),
            "temperature": self._temperature_spin.value(),
        }

    def get_escalation_config(self) -> dict:
        """返回质检升级配置"""
        return {
            "escalation_enabled": self._vlm_enable_cb.isChecked(),
            "escalation_threshold": self._escalation_thr_spin.value(),
        }

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_primary_group())
        layout.addWidget(self._build_vlm_aux_group())
        layout.addWidget(self._build_sampling_group())
        layout.addStretch()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(340)

    def _build_primary_group(self) -> QGroupBox:
        gb = QGroupBox("主引擎")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._primary_type_cb = QComboBox()
        self._primary_type_cb.addItems(["vlm", "pytorch", "onnx", "tensorrt"])
        form.addRow("类型:", self._primary_type_cb)

        path_row = QHBoxLayout()
        self._primary_path_edit = QLineEdit()
        self._primary_path_edit.setPlaceholderText("模型路径/目录")
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(lambda: self._browse_path(self._primary_path_edit))
        path_row.addWidget(self._primary_path_edit)
        path_row.addWidget(browse_btn)
        form.addRow("路径:", path_row)

        btn_row = QHBoxLayout()
        self._primary_load_btn = QPushButton("加载")
        self._primary_unload_btn = QPushButton("释放")
        self._primary_unload_btn.setEnabled(False)
        self._primary_load_btn.clicked.connect(self._on_load_primary)
        self._primary_unload_btn.clicked.connect(self._on_unload_primary)
        btn_row.addWidget(self._primary_load_btn)
        btn_row.addWidget(self._primary_unload_btn)
        form.addRow(btn_row)

        self._primary_status_lbl = QLabel("○ 未加载")
        self._primary_status_lbl.setStyleSheet("color: #888;")
        form.addRow("状态:", self._primary_status_lbl)

        return gb

    def _build_vlm_aux_group(self) -> QGroupBox:
        gb = QGroupBox("VLM 辅助引擎（可选）")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._vlm_enable_cb = QCheckBox("启用低置信度 VLM 复核")
        form.addRow(self._vlm_enable_cb)

        path_row = QHBoxLayout()
        self._vlm_path_edit = QLineEdit()
        self._vlm_path_edit.setPlaceholderText("VLM 模型路径")
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(lambda: self._browse_path(self._vlm_path_edit))
        path_row.addWidget(self._vlm_path_edit)
        path_row.addWidget(browse_btn)
        form.addRow("路径:", path_row)

        self._escalation_thr_spin = QDoubleSpinBox()
        self._escalation_thr_spin.setRange(0.0, 1.0)
        self._escalation_thr_spin.setSingleStep(0.05)
        self._escalation_thr_spin.setValue(0.8)
        form.addRow("升级阈值:", self._escalation_thr_spin)

        btn_row = QHBoxLayout()
        self._vlm_load_btn = QPushButton("加载")
        self._vlm_unload_btn = QPushButton("释放")
        self._vlm_unload_btn.setEnabled(False)
        self._vlm_load_btn.clicked.connect(self._on_load_vlm)
        self._vlm_unload_btn.clicked.connect(self._on_unload_vlm)
        btn_row.addWidget(self._vlm_load_btn)
        btn_row.addWidget(self._vlm_unload_btn)
        form.addRow(btn_row)

        self._vlm_status_lbl = QLabel("○ 未加载")
        self._vlm_status_lbl.setStyleSheet("color: #888;")
        form.addRow("状态:", self._vlm_status_lbl)

        return gb

    def _build_sampling_group(self) -> QGroupBox:
        gb = QGroupBox("VLM 采样设置")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._multi_sample_cb = QCheckBox("启用多次采样")
        form.addRow(self._multi_sample_cb)

        self._sample_count_spin = QSpinBox()
        self._sample_count_spin.setRange(1, 10)
        self._sample_count_spin.setValue(3)
        form.addRow("次数:", self._sample_count_spin)

        self._temperature_spin = QDoubleSpinBox()
        self._temperature_spin.setRange(0.0, 2.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setValue(0.7)
        form.addRow("温度:", self._temperature_spin)

        return gb

    # -------- default --------

    def _load_default_config(self) -> dict:
        if DEFAULT_CONFIG.exists():
            with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _apply_defaults(self):
        engine_cfg = self._default_cfg.get("engine", {})
        t = engine_cfg.get("type", "vlm")
        idx = self._primary_type_cb.findText(t)
        if idx >= 0:
            self._primary_type_cb.setCurrentIndex(idx)

        vlm_cfg = engine_cfg.get("vlm", {})
        self._primary_path_edit.setText(vlm_cfg.get("model_path", ""))
        self._vlm_path_edit.setText(vlm_cfg.get("model_path", ""))

        qc_cfg = self._default_cfg.get("quality_check", {})
        self._escalation_thr_spin.setValue(qc_cfg.get("escalation_threshold", 0.8))
        self._vlm_enable_cb.setChecked(qc_cfg.get("escalation_enabled", False))

        self._multi_sample_cb.setChecked(vlm_cfg.get("multi_sample", False))
        self._sample_count_spin.setValue(vlm_cfg.get("sample_count", 3))
        self._temperature_spin.setValue(vlm_cfg.get("temperature", 0.7))

    # -------- build config for EngineFactory --------

    def _build_primary_engine_config(self) -> dict:
        """从 UI + default 构造引擎配置"""
        cfg = dict(self._default_cfg.get("engine", {}))
        cfg["type"] = self._primary_type_cb.currentText()

        path = self._primary_path_edit.text().strip()
        if cfg["type"] == "vlm":
            vlm_cfg = dict(cfg.get("vlm", {}))
            if path:
                vlm_cfg["model_path"] = path
            vlm_cfg.update(self.get_sampling_config())
            cfg["vlm"] = vlm_cfg
        else:
            custom_cfg = dict(cfg.get("custom", {}))
            if path:
                custom_cfg["model_path"] = path
            cfg["custom"] = custom_cfg

        # 注入 prompts_dir
        prompts_dir = self._default_cfg.get("prompts", {}).get("dir", "configs/prompts")
        if not os.path.isabs(prompts_dir):
            prompts_dir = str(PROJECT_ROOT / prompts_dir)
        cfg["prompts_dir"] = prompts_dir
        return cfg

    def _build_vlm_aux_config(self) -> dict:
        """VLM 辅助引擎配置（强制 type=vlm）"""
        cfg = dict(self._default_cfg.get("engine", {}))
        cfg["type"] = "vlm"
        vlm_cfg = dict(cfg.get("vlm", {}))
        path = self._vlm_path_edit.text().strip()
        if path:
            vlm_cfg["model_path"] = path
        cfg["vlm"] = vlm_cfg

        prompts_dir = self._default_cfg.get("prompts", {}).get("dir", "configs/prompts")
        if not os.path.isabs(prompts_dir):
            prompts_dir = str(PROJECT_ROOT / prompts_dir)
        cfg["prompts_dir"] = prompts_dir
        return cfg

    # -------- actions --------

    def _browse_path(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(
            self, "选择模型目录", line_edit.text() or str(Path.home())
        )
        if path:
            line_edit.setText(path)

    def _busy(self, busy: bool, target: str = "primary"):
        if target == "primary":
            self._primary_load_btn.setEnabled(not busy)
            self._primary_type_cb.setEnabled(not busy)
            self._primary_path_edit.setEnabled(not busy)
        elif target == "vlm":
            self._vlm_load_btn.setEnabled(not busy)
            self._vlm_path_edit.setEnabled(not busy)

    def _on_load_primary(self):
        if self._primary_engine is not None:
            self.status_message.emit("主引擎已加载，请先释放")
            return
        cfg = self._build_primary_engine_config()
        self._primary_status_lbl.setText("⏳ 加载中...")
        self._primary_status_lbl.setStyleSheet("color: #FF9800;")
        self._busy(True, "primary")
        self.status_message.emit(f"加载主引擎: {cfg.get('type')}")

        self._load_thread = _EngineLoadThread(cfg, "load", parent=self)
        self._load_thread.done.connect(self._on_primary_load_done)
        self._load_thread.start()

    def _on_primary_load_done(self, engine, err):
        self._busy(False, "primary")
        if err:
            self._primary_status_lbl.setText("✗ 加载失败")
            self._primary_status_lbl.setStyleSheet("color: #EF5350;")
            self.status_message.emit(f"主引擎加载失败: {err}")
        else:
            self._primary_engine = engine
            self._primary_status_lbl.setText("● 已就绪")
            self._primary_status_lbl.setStyleSheet("color: #4CAF50;")
            self._primary_unload_btn.setEnabled(True)
            self.primary_engine_changed.emit(engine)
            self.status_message.emit("主引擎已加载")

    def _on_unload_primary(self):
        if self._primary_engine is None:
            return
        engine = self._primary_engine
        self._primary_engine = None
        self.primary_engine_changed.emit(None)
        self._primary_status_lbl.setText("⏳ 释放中...")
        self._primary_unload_btn.setEnabled(False)

        self._load_thread = _EngineLoadThread({}, "unload", engine=engine, parent=self)
        self._load_thread.done.connect(lambda _e, err: self._after_primary_unload(err))
        self._load_thread.start()

    def _after_primary_unload(self, err: str):
        if err:
            self._primary_status_lbl.setText("✗ 释放失败")
            self.status_message.emit(f"主引擎释放失败: {err}")
        else:
            self._primary_status_lbl.setText("○ 未加载")
            self._primary_status_lbl.setStyleSheet("color: #888;")
            self.status_message.emit("主引擎已释放")
        self._busy(False, "primary")

    def _on_load_vlm(self):
        if self._vlm_engine is not None:
            self.status_message.emit("VLM 辅助引擎已加载")
            return
        cfg = self._build_vlm_aux_config()
        self._vlm_status_lbl.setText("⏳ 加载中...")
        self._vlm_status_lbl.setStyleSheet("color: #FF9800;")
        self._busy(True, "vlm")
        self.status_message.emit("加载 VLM 辅助引擎")

        self._load_thread = _EngineLoadThread(cfg, "load", parent=self)
        self._load_thread.done.connect(self._on_vlm_load_done)
        self._load_thread.start()

    def _on_vlm_load_done(self, engine, err):
        self._busy(False, "vlm")
        if err:
            self._vlm_status_lbl.setText("✗ 加载失败")
            self._vlm_status_lbl.setStyleSheet("color: #EF5350;")
            self.status_message.emit(f"VLM 辅助加载失败: {err}")
        else:
            self._vlm_engine = engine
            self._vlm_status_lbl.setText("● 已就绪")
            self._vlm_status_lbl.setStyleSheet("color: #4CAF50;")
            self._vlm_unload_btn.setEnabled(True)
            self.vlm_engine_changed.emit(engine)
            self.status_message.emit("VLM 辅助引擎已加载")

    def _on_unload_vlm(self):
        if self._vlm_engine is None:
            return
        engine = self._vlm_engine
        self._vlm_engine = None
        self.vlm_engine_changed.emit(None)
        self._vlm_status_lbl.setText("⏳ 释放中...")
        self._vlm_unload_btn.setEnabled(False)

        self._load_thread = _EngineLoadThread({}, "unload", engine=engine, parent=self)
        self._load_thread.done.connect(lambda _e, err: self._after_vlm_unload(err))
        self._load_thread.start()

    def _after_vlm_unload(self, err: str):
        if err:
            self._vlm_status_lbl.setText("✗ 释放失败")
            self.status_message.emit(f"VLM 辅助释放失败: {err}")
        else:
            self._vlm_status_lbl.setText("○ 未加载")
            self._vlm_status_lbl.setStyleSheet("color: #888;")
            self.status_message.emit("VLM 辅助已释放")
        self._busy(False, "vlm")

    def shutdown(self):
        """窗口关闭时调用，释放引擎"""
        if self._primary_engine is not None:
            try:
                self._primary_engine.unload()
            except Exception:
                logger.exception("释放主引擎出错")
        if self._vlm_engine is not None:
            try:
                self._vlm_engine.unload()
            except Exception:
                logger.exception("释放 VLM 引擎出错")

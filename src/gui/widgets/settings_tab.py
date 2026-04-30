"""
设置 Tab。

- 运行参数（num_io_workers / log_level / ui_update_interval / temp_dir）
- Prompt 模板查看 & 编辑
- 保存到 configs/default.yaml（可选）

运行参数改动只作用于当前会话（通过 GlobalSettings 模块级单例共享），
保存按钮会写回 default.yaml。
"""
import os
from pathlib import Path
from typing import Optional

import yaml

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QSpinBox, QPushButton, QLabel, QFileDialog,
    QPlainTextEdit, QTabWidget, QMessageBox, QLineEdit,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"


class SettingsTab(QWidget):
    """设置 Tab"""

    settings_applied = pyqtSignal(dict)   # 运行参数变更时广播

    def __init__(self, parent=None):
        super().__init__(parent)
        self._default_cfg = self._load_default()
        self._build_ui()
        self._apply_defaults_to_ui()

    def set_primary_engine(self, engine):
        pass

    def set_vlm_engine(self, engine):
        pass

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_runtime_group())
        layout.addWidget(self._build_prompt_group(), stretch=1)
        layout.addLayout(self._build_button_row())

    def _build_runtime_group(self) -> QGroupBox:
        gb = QGroupBox("运行参数")
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignRight)

        self._num_workers_spin = QSpinBox()
        self._num_workers_spin.setRange(1, 32)
        form.addRow("IO 线程数:", self._num_workers_spin)

        self._log_level_cb = QComboBox()
        self._log_level_cb.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        form.addRow("日志级别:", self._log_level_cb)

        self._ui_interval_spin = QSpinBox()
        self._ui_interval_spin.setRange(1, 200)
        form.addRow("UI 更新间隔（条）:", self._ui_interval_spin)

        temp_row = QHBoxLayout()
        self._temp_dir_edit = QLineEdit()
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse_temp_dir)
        temp_row.addWidget(self._temp_dir_edit)
        temp_row.addWidget(browse_btn)
        temp_row.setContentsMargins(0, 0, 0, 0)
        temp_w = QWidget()
        temp_w.setLayout(temp_row)
        form.addRow("临时目录:", temp_w)

        return gb

    def _build_prompt_group(self) -> QGroupBox:
        gb = QGroupBox("Prompt 模板")
        v = QVBoxLayout(gb)

        hint = QLabel(
            "修改后点击「保存 Prompt」写入对应文件。模板中 {categories} 等占位符由引擎运行时填充。"
        )
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self._prompt_tabs = QTabWidget()
        self._prompt_editors: dict[str, QPlainTextEdit] = {}

        # 动态加载 configs/prompts/ 下所有 .txt
        if PROMPTS_DIR.is_dir():
            for f in sorted(PROMPTS_DIR.glob("*.txt")):
                editor = QPlainTextEdit()
                editor.setPlainText(f.read_text(encoding="utf-8"))
                self._prompt_tabs.addTab(editor, f.name)
                self._prompt_editors[f.name] = editor
        else:
            self._prompt_tabs.addTab(QPlainTextEdit("configs/prompts 目录不存在"), "错误")

        v.addWidget(self._prompt_tabs, stretch=1)
        return gb

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()

        self._apply_btn = QPushButton("应用（当前会话）")
        self._apply_btn.clicked.connect(self._on_apply)
        row.addWidget(self._apply_btn)

        self._save_runtime_btn = QPushButton("保存运行参数到 default.yaml")
        self._save_runtime_btn.clicked.connect(self._on_save_runtime)
        row.addWidget(self._save_runtime_btn)

        self._save_prompt_btn = QPushButton("保存 Prompt")
        self._save_prompt_btn.clicked.connect(self._on_save_prompts)
        row.addWidget(self._save_prompt_btn)

        self._reset_btn = QPushButton("重置为文件值")
        self._reset_btn.clicked.connect(self._reset)
        row.addWidget(self._reset_btn)

        return row

    # -------- data --------

    def _load_default(self) -> dict:
        if DEFAULT_CONFIG.exists():
            with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _apply_defaults_to_ui(self):
        runtime = self._default_cfg.get("runtime", {})
        self._num_workers_spin.setValue(runtime.get("num_io_workers", 4))
        self._ui_interval_spin.setValue(runtime.get("ui_update_interval", 10))
        idx = self._log_level_cb.findText(runtime.get("log_level", "INFO"))
        if idx >= 0:
            self._log_level_cb.setCurrentIndex(idx)
        self._temp_dir_edit.setText(runtime.get("temp_dir", ""))

    def _collect_runtime(self) -> dict:
        return {
            "num_io_workers": self._num_workers_spin.value(),
            "ui_update_interval": self._ui_interval_spin.value(),
            "log_level": self._log_level_cb.currentText(),
            "temp_dir": self._temp_dir_edit.text().strip(),
        }

    # -------- slots --------

    def _browse_temp_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择临时目录", self._temp_dir_edit.text()
        )
        if path:
            self._temp_dir_edit.setText(path)

    def _on_apply(self):
        """仅影响当前会话：广播给关心的组件"""
        settings = self._collect_runtime()
        self.settings_applied.emit(settings)

        # 调整 logger 级别
        import logging
        logging.getLogger("smartlabel").setLevel(
            getattr(logging, settings["log_level"].upper(), logging.INFO)
        )
        QMessageBox.information(self, "已应用", "当前会话的运行参数已更新")

    def _on_save_runtime(self):
        settings = self._collect_runtime()
        try:
            cfg = self._load_default()
            cfg.setdefault("runtime", {}).update(settings)
            with open(DEFAULT_CONFIG, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            self._default_cfg = cfg
            QMessageBox.information(
                self, "保存成功",
                f"运行参数已写入:\n{DEFAULT_CONFIG}"
            )
        except Exception as e:
            logger.exception("保存运行参数失败")
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_save_prompts(self):
        if not PROMPTS_DIR.is_dir():
            QMessageBox.critical(self, "失败", "configs/prompts 目录不存在")
            return

        saved = []
        try:
            for fname, editor in self._prompt_editors.items():
                path = PROMPTS_DIR / fname
                path.write_text(editor.toPlainText(), encoding="utf-8")
                saved.append(fname)
            QMessageBox.information(
                self, "保存成功",
                f"已保存 {len(saved)} 个 Prompt:\n" + "\n".join(saved)
            )
        except Exception as e:
            logger.exception("保存 Prompt 失败")
            QMessageBox.critical(self, "保存失败", str(e))

    def _reset(self):
        self._default_cfg = self._load_default()
        self._apply_defaults_to_ui()
        # 重新读 prompt
        for fname, editor in self._prompt_editors.items():
            path = PROMPTS_DIR / fname
            if path.is_file():
                editor.setPlainText(path.read_text(encoding="utf-8"))

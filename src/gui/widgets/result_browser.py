"""
结果浏览器：左侧列表 + 右侧详情。

特性：
- 列表项同步加载缩略图（MVP 方案，大批量时可扩展为懒加载）
- 选中项时右侧显示大图 + 元数据
- 可选标签编辑 UI：调用 set_editable_categories(cats) 启用
  - 详情区会出现"新标签"下拉框 + 修改按钮
  - 点击修改后发射 label_changed(item, new_label)，由调用方执行落盘
- 支持预标注结果、质检结果两种 item 格式
"""
import os
from typing import Optional

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QListWidget, QListWidgetItem,
    QLabel, QTextEdit, QGroupBox, QComboBox, QPushButton,
)

from src.gui.widgets.image_viewer import ImageViewer


THUMB_SIZE = 80
VISIBLE_LIMIT = 500
THUMB_BATCH_SIZE = 8       # 每批生成多少个缩略图
THUMB_BATCH_INTERVAL = 15  # 批次间隔 ms，让 UI 事件循环有机会跑


def _make_thumbnail(image_path: str, size: int = THUMB_SIZE) -> Optional[QIcon]:
    """生成缩略图 icon。失败返回 None。"""
    if not os.path.isfile(image_path):
        return None
    pix = QPixmap(image_path)
    if pix.isNull():
        return None
    scaled = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(scaled)


class ResultBrowser(QWidget):
    """
    结果浏览控件。

    Item 格式（dict）：
        {
            "image_path": str,
            "title": str,
            "subtitle": str,
            "meta": dict,
            "detections": list[dict] | None,
            "current_label": str | None,    # 用于编辑下拉框的"当前值"
        }

    Signals:
        current_changed(dict)                 # 选中项变更
        label_changed(dict, str)              # 用户点击"修改"：(item, new_label)
    """

    current_changed = pyqtSignal(dict)
    label_changed = pyqtSignal(dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[dict] = []
        self._editable_categories: Optional[list[str]] = None
        # 缩略图分批加载状态
        self._thumb_pending: list[int] = []   # 待加载缩略图的行号队列
        self._thumb_load_enabled: bool = True
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.timeout.connect(self._load_thumb_batch)
        self._build_ui()

    # -------- public --------

    def set_items(self, items: list[dict], load_thumbnails: bool = True):
        """
        设置并刷新列表。

        Args:
            items: item dict 列表
            load_thumbnails: True 后台分批生成缩略图；False 只显示文字（预览场景用）
        """
        self._items = items[:VISIBLE_LIMIT]
        self._thumb_load_enabled = load_thumbnails
        self._refresh_list()
        self._summary_lbl.setText(
            f"共 {len(items)} 条"
            + (f"（仅展示前 {VISIBLE_LIMIT} 条）" if len(items) > VISIBLE_LIMIT else "")
        )
        if self._items:
            self._list.setCurrentRow(0)

    def clear(self):
        self._thumb_timer.stop()
        self._thumb_pending = []
        self._items = []
        self._list.clear()
        self._viewer.clear()
        self._meta_text.clear()
        self._summary_lbl.setText("暂无结果")
        self._set_edit_enabled(False)

    def set_editable_categories(self, categories: Optional[list[str]]):
        """None / 空列表 → 隐藏编辑区；有内容 → 显示并填充下拉框"""
        self._editable_categories = list(categories) if categories else None
        if self._editable_categories:
            self._category_cb.clear()
            self._category_cb.addItems(self._editable_categories)
            self._edit_group.setVisible(True)
        else:
            self._edit_group.setVisible(False)
        # 重新渲染当前项，让编辑区的启用状态与当前项匹配
        row = self._list.currentRow()
        if 0 <= row < len(self._items):
            self._render_detail(self._items[row])

    def current_item(self) -> Optional[dict]:
        row = self._list.currentRow()
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def update_current_item(self, new_label: Optional[str] = None,
                             new_subtitle: Optional[str] = None,
                             new_meta_patch: Optional[dict] = None):
        """外部修改后更新当前选中项显示（不重绘整个列表）"""
        row = self._list.currentRow()
        if row < 0 or row >= len(self._items):
            return
        item = self._items[row]

        if new_label is not None:
            item["current_label"] = new_label
        if new_subtitle is not None:
            item["subtitle"] = new_subtitle
        if new_meta_patch:
            meta = dict(item.get("meta") or {})
            meta.update(new_meta_patch)
            item["meta"] = meta

        # 刷新列表项文本
        list_item = self._list.item(row)
        title = item.get("title", os.path.basename(item.get("image_path", "")))
        sub = item.get("subtitle", "")
        list_item.setText(f"{title}\n{sub}" if sub else title)

        # 刷新右侧详情
        self._render_detail(item)

    # -------- build UI --------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._summary_lbl = QLabel("暂无结果")
        self._summary_lbl.setStyleSheet("color: #888;")
        layout.addWidget(self._summary_lbl)

        splitter = QSplitter(Qt.Horizontal)

        # 左：列表
        self._list = QListWidget()
        self._list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self._list.setSpacing(2)
        self._list.currentRowChanged.connect(self._on_row_changed)
        splitter.addWidget(self._list)

        # 右：详情
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)

        self._viewer = ImageViewer()
        right_v.addWidget(self._viewer, stretch=3)

        meta_gb = QGroupBox("详情")
        meta_v = QVBoxLayout(meta_gb)
        self._meta_text = QTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setMaximumHeight(180)
        meta_v.addWidget(self._meta_text)
        right_v.addWidget(meta_gb, stretch=1)

        # 可选的编辑区（默认隐藏）
        self._edit_group = QGroupBox("修改标签")
        edit_v = QVBoxLayout(self._edit_group)
        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("新标签:"))
        self._category_cb = QComboBox()
        edit_row.addWidget(self._category_cb, stretch=1)
        self._modify_btn = QPushButton("修改")
        self._modify_btn.clicked.connect(self._on_modify_clicked)
        edit_row.addWidget(self._modify_btn)
        edit_v.addLayout(edit_row)
        self._edit_group.setVisible(False)
        right_v.addWidget(self._edit_group)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([300, 600])
        layout.addWidget(splitter)

    def _set_edit_enabled(self, enabled: bool):
        self._edit_group.setEnabled(enabled)

    # -------- internal --------

    def _refresh_list(self):
        # 先停掉上一轮未完成的缩略图加载（新一轮数据覆盖）
        self._thumb_timer.stop()
        self._thumb_pending = []

        self._list.clear()
        for item in self._items:
            list_item = QListWidgetItem()
            title = item.get("title", os.path.basename(item.get("image_path", "")))
            subtitle = item.get("subtitle", "")
            list_item.setText(f"{title}\n{subtitle}" if subtitle else title)
            list_item.setSizeHint(QSize(0, THUMB_SIZE + 16))
            self._list.addItem(list_item)

        # 分批后台加载缩略图，避免一次性阻塞主线程
        if self._thumb_load_enabled and self._items:
            self._thumb_pending = list(range(len(self._items)))
            self._thumb_timer.start(0)

    def _load_thumb_batch(self):
        """每次加载 THUMB_BATCH_SIZE 张，通过 timer 让出控制权"""
        if not self._thumb_pending:
            return
        for _ in range(THUMB_BATCH_SIZE):
            if not self._thumb_pending:
                break
            row = self._thumb_pending.pop(0)
            if row >= len(self._items):
                continue
            item = self._items[row]
            icon = _make_thumbnail(item.get("image_path", ""))
            list_item = self._list.item(row)
            if icon and list_item is not None:
                list_item.setIcon(icon)
        if self._thumb_pending:
            self._thumb_timer.start(THUMB_BATCH_INTERVAL)

    def _on_row_changed(self, row: int):
        if row < 0 or row >= len(self._items):
            self._viewer.clear()
            self._meta_text.clear()
            self._set_edit_enabled(False)
            return
        item = self._items[row]
        self._render_detail(item)
        self.current_changed.emit(item)

    def _render_detail(self, item: dict):
        self._viewer.load_image(
            item.get("image_path", ""),
            detections=item.get("detections"),
        )
        meta = item.get("meta") or {}
        self._meta_text.setPlainText(
            "\n".join(f"{k}: {v}" for k, v in meta.items())
        )

        # 编辑 UI：预选当前标签
        if self._editable_categories:
            current = item.get("current_label")
            if current and current in self._editable_categories:
                idx = self._category_cb.findText(current)
                if idx >= 0:
                    self._category_cb.setCurrentIndex(idx)
            self._set_edit_enabled(True)
        else:
            self._set_edit_enabled(False)

    def _on_modify_clicked(self):
        item = self.current_item()
        if item is None:
            return
        new_label = self._category_cb.currentText()
        if not new_label:
            return
        self.label_changed.emit(item, new_label)

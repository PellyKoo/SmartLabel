"""
类别/目标列表输入控件：QLineEdit + 📁 按钮（从 txt 加载）。

txt 格式：每行一个类别，空行和 # 开头的注释行自动跳过。

QLineEdit 内部始终以逗号分隔存储，保持与下游 `x.split(",")` 解析逻辑兼容，
用户也可手动微调。
"""
import os
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox,
)


def parse_categories_txt(path: str) -> list[str]:
    """
    读取 txt，每行一个类别。

    规则：
    - 去首尾空白
    - 跳过空行
    - 跳过以 # 开头的注释行
    - 保持原顺序，不自动去重（用户列表顺序可能有意义）
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cats: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cats.append(line)
    return cats


def make_category_input(
    placeholder: str = "类别1,类别2,...",
    file_dialog_caption: str = "选择类别列表 txt（每行一个）",
    parent: Optional[QWidget] = None,
) -> tuple[QLineEdit, QWidget]:
    """
    构造"文本框 + 加载 txt"组合控件。

    Returns:
        (edit, container_widget)
        - edit: QLineEdit，下游代码直接 `.text().split(",")` 即可
        - container_widget: 加入表单的整行，disable 它可一并屏蔽按钮
    """
    edit = QLineEdit(parent)
    edit.setPlaceholderText(placeholder)

    btn = QPushButton("📁", parent)
    btn.setFixedWidth(32)
    btn.setToolTip("从 txt 加载（每行一个类别，# 开头视为注释）")

    def _browse():
        path, _ = QFileDialog.getOpenFileName(
            parent, file_dialog_caption, edit.text(),
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            cats = parse_categories_txt(path)
        except Exception as e:
            QMessageBox.critical(parent, "读取失败", str(e))
            return
        if not cats:
            QMessageBox.warning(
                parent, "提示", f"未从文件解析到任何类别:\n{path}"
            )
            return
        edit.setText(",".join(cats))

    btn.clicked.connect(_browse)

    container = QWidget(parent)
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit)
    row.addWidget(btn)
    return edit, container

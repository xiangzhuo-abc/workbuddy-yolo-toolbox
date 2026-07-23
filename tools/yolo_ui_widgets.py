"""YOLO 工具箱共享的纯展示 Qt 组件。"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import yolo_ui_theme as theme


class SectionHeader(QWidget):
    """统一的区块标题，可选显示一行简短副标题。"""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("SectionHeader")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING["xs"])

        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(subtitle)
        theme.set_text_role(self.subtitle_label, "hint")
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)


class StatusBadge(QLabel):
    """带语义色但始终保留文字说明的状态标签。"""

    def __init__(self, text: str = "", tone: str = "info", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        theme.set_status_tone(self, tone)


class ToolButton(QPushButton):
    """具有稳定尺寸、图标和工具提示的紧凑命令按钮。"""

    def __init__(
        self,
        text: str = "",
        tooltip: str = "",
        icon: QIcon | None = None,
        parent=None,
    ):
        super().__init__(text, parent)
        self.setObjectName("ToolButton")
        self.setMinimumHeight(theme.CONTROL_HEIGHT)
        if not text:
            self.setFixedSize(theme.TOOL_BUTTON_SIZE, theme.TOOL_BUTTON_SIZE)
        if tooltip:
            self.setToolTip(tooltip)
        if icon is not None and not icon.isNull():
            self.setIcon(icon)
        theme.set_button_role(self, "secondary")


class PathBar(QWidget):
    """只负责显示路径并发出浏览请求，不直接访问文件系统。"""

    browse_clicked = pyqtSignal()

    def __init__(self, label: str, path: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("PathBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING["sm"])

        self.label = QLabel(label)
        self.label.setObjectName("PathBarLabel")
        self.label.setMinimumWidth(64)
        layout.addWidget(self.label)

        self.line_edit = QLineEdit(path)
        self.line_edit.setReadOnly(True)
        self.line_edit.setCursorPosition(0)
        layout.addWidget(self.line_edit, 1)

        self.browse_button = ToolButton("浏览", f"选择{label}目录")
        self.browse_button.clicked.connect(
            lambda _checked=False: self.browse_clicked.emit()
        )
        layout.addWidget(self.browse_button)

    def path(self) -> str:
        return self.line_edit.text()

    def set_path(self, value: str) -> None:
        self.line_edit.setText(value)
        self.line_edit.setCursorPosition(0)

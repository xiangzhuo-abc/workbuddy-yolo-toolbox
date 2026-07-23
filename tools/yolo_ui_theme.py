"""YOLO 工具箱共享界面主题。"""

from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget


FONT_FAMILY = "Microsoft YaHei UI"
CONTROL_HEIGHT = 32
TOOL_BUTTON_SIZE = 34
SPACING = {"xs": 4, "sm": 8, "md": 12, "lg": 16}


def apply_app_theme(app: QApplication) -> None:
    """应用统一的浅色专业工具风主题。"""
    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILY, 9))
    app.setStyleSheet(_STYLE_SHEET)


def set_button_role(button: QPushButton, role: str = "secondary") -> QPushButton:
    """设置按钮视觉角色。"""
    button.setProperty("role", role)
    button.setCursor(QtPointingHandCursor.cursor())
    refresh_style(button)
    return button


def set_panel(widget: QWidget, name: str = "Panel") -> QWidget:
    """标记为统一面板样式。"""
    widget.setObjectName(name)
    return widget


def set_status_tone(label: QLabel, tone: str = "info") -> QLabel:
    """设置状态提示的语义色。"""
    label.setObjectName("StatusPill")
    label.setProperty("tone", tone)
    refresh_style(label)
    return label


def set_text_role(label: QLabel, role: str = "muted") -> QLabel:
    """设置普通说明文字的视觉层级。"""
    label.setProperty("textRole", role)
    refresh_style(label)
    return label


def set_image_canvas(widget: QWidget) -> QWidget:
    """标记图片显示区域样式。"""
    widget.setObjectName("ImageCanvas")
    refresh_style(widget)
    return widget


def refresh_style(widget: QWidget) -> None:
    """刷新动态属性对应的 Qt 样式。"""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class QtPointingHandCursor:
    """延迟导入 Qt，避免主题模块导入时拉起额外命名。"""

    @staticmethod
    def cursor():
        from PyQt5.QtCore import Qt

        return Qt.PointingHandCursor


_STYLE_SHEET = """
QMainWindow, QDialog {
    background: #f4f7fb;
}

QWidget {
    color: #1f2937;
    font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
    font-size: 9pt;
}

QLabel#PageTitle {
    color: #0f172a;
    font-size: 15pt;
    font-weight: 700;
}

QLabel#PageSubtitle {
    color: #64748b;
    font-size: 9pt;
}

QLabel#SectionTitle {
    color: #334155;
    font-weight: 700;
}

QLabel#StatusPill {
    background: #e8f3ff;
    border: 1px solid #b9d8ff;
    border-radius: 6px;
    color: #1d4f91;
    padding: 6px 10px;
}

QLabel#StatusPill[tone="success"] {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #047857;
}

QLabel#StatusPill[tone="warning"] {
    background: #fff7ed;
    border-color: #fed7aa;
    color: #9a3412;
}

QLabel#StatusPill[tone="danger"] {
    background: #fff1f2;
    border-color: #fecdd3;
    color: #be123c;
}

QLabel[textRole="muted"] {
    color: #64748b;
}

QLabel[textRole="hint"] {
    color: #64748b;
    font-size: 8pt;
}

QLabel#ImageCanvas {
    background: #eef3f8;
    border: 1px solid #d6dee9;
    border-radius: 8px;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 700;
    color: #334155;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background: #f4f7fb;
}

QWidget#Panel, QWidget#SidePanel, QWidget#LogPanel {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
}

QWidget#AppHeader {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
}

QWidget#WorkflowNav, QWidget#ConsolePanel {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
}

QWidget#ImageNavigator, QWidget#AnnotationWorkspace, QWidget#AnnotationInspector {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
}

QWidget#DetectionSetup, QWidget#DetectionWorkspace, QWidget#DetectionInspector {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
}

QWidget#TrainingSource {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
}

QWidget#TrainingFooter {
    background: transparent;
    border-top: 1px solid #dbe4ef;
}

QTabWidget#TrainingTabs::pane {
    background: #ffffff;
    border-color: #dbe4ef;
}

QWidget#TrainingBasic, QWidget#TrainingAdvanced, QWidget#TrainingDataCheck {
    background: #ffffff;
}

QWidget#AnnotationToolbar, QWidget#DetectionToolbar, QWidget#DetectionControls {
    background: transparent;
}

QWidget#AnnotationWorkspace QLabel#ImageCanvas,
QWidget#DetectionWorkspace QLabel#ImageCanvas {
    background: #20262e;
    border-color: #111827;
    border-radius: 6px;
}

QWidget#SectionHeader, QWidget#PathBar {
    background: transparent;
}

QLabel#PathBarLabel {
    color: #475569;
    font-weight: 700;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    min-height: 28px;
    padding: 4px 8px;
    selection-background-color: #2563eb;
}

QLineEdit:read-only {
    background: #f8fafc;
    color: #475569;
}

QTextEdit, QListWidget, QTableWidget {
    background: #ffffff;
    border: 1px solid #d6dee9;
    border-radius: 8px;
    padding: 6px;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}

QTextEdit {
    background: #fbfdff;
}

QHeaderView::section {
    background: #f1f5f9;
    border: 0;
    border-bottom: 1px solid #d6dee9;
    color: #334155;
    font-weight: 700;
    padding: 6px 8px;
}

QTableWidget {
    gridline-color: #e2e8f0;
    alternate-background-color: #f8fafc;
}

QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 6px;
    top: -1px;
}

QTabBar::tab {
    background: #f8fafc;
    border: 1px solid #dbe4ef;
    border-bottom: 0;
    min-width: 72px;
    min-height: 28px;
    padding: 5px 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #1d4f91;
    font-weight: 700;
}

QTabBar::tab:hover:!selected {
    background: #eef5ff;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    color: #1f2937;
    font-weight: 600;
    min-height: 30px;
    padding: 6px 12px;
}

QPushButton:hover {
    background: #f1f7ff;
    border-color: #93c5fd;
}

QPushButton:pressed {
    background: #dbeafe;
}

QPushButton:disabled {
    background: #f1f5f9;
    border-color: #e2e8f0;
    color: #94a3b8;
}

QPushButton[role="primary"] {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
}

QPushButton[role="primary"]:hover {
    background: #1d4ed8;
}

QPushButton[role="accent"] {
    background: #0f766e;
    border-color: #0f766e;
    color: #ffffff;
}

QPushButton[role="accent"]:hover {
    background: #0d665f;
}

QPushButton[role="warning"] {
    background: #fff7ed;
    border-color: #fdba74;
    color: #9a3412;
}

QPushButton[role="danger"] {
    background: #fff1f2;
    border-color: #fda4af;
    color: #be123c;
}

QPushButton[role="primary"]:disabled,
QPushButton[role="accent"]:disabled,
QPushButton[role="warning"]:disabled,
QPushButton[role="danger"]:disabled {
    background: #f1f5f9;
    border-color: #e2e8f0;
    color: #94a3b8;
}

QPushButton[role="navigation"] {
    background: transparent;
    border: 1px solid transparent;
    color: #334155;
    font-weight: 600;
    min-height: 34px;
    padding: 7px 10px;
    text-align: left;
}

QPushButton[role="navigation"]:hover {
    background: #f1f5f9;
    border-color: #dbe4ef;
}

QPushButton[role="navigation"]:pressed {
    background: #e8f3ff;
    border-color: #b9d8ff;
    color: #1d4f91;
}

QPushButton[role="navigation"]:disabled {
    background: transparent;
    border-color: transparent;
    color: #94a3b8;
}

QPushButton#ToolButton {
    min-height: 32px;
    padding: 5px 10px;
}

QProgressBar {
    background: #e2e8f0;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background: #2563eb;
    border-radius: 5px;
}

QSplitter::handle {
    background: #e2e8f0;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #dbe4ef;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #2563eb;
    border: 1px solid #1d4ed8;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
"""

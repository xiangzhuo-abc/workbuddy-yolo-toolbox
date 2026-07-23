"""
YOLO 模型测试工具 —— 独立 GUI

用法：
    python tools/yolo_detect_gui.py
    python tools/yolo_detect_gui.py --model runs/coc_detect/weights/best.pt

功能：
- 选择训练好的模型（.pt）
- 选择单张图片或整个目录进行检测
- 可调置信度阈值
- 图片上叠加检测框和标签
- 检测结果列表（类别、置信度、坐标）
- 上一张/下一张浏览
- 保存检测结果图
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from PyQt5.QtCore import Qt, QPoint, QProcess
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QFileDialog, QMessageBox,
    QSplitter, QListWidget, QShortcut, QSlider,
    QCheckBox, QGridLayout, QStyle,
)
import cv2
import numpy as np

import yolo_ui_theme as theme
from yolo_ui_widgets import SectionHeader, StatusBadge, ToolButton
from core.task_manager import TaskManager
from core.task_protocol import TaskEventType, decode_task_event
from core.runtime_paths import RuntimePaths
from core.worker_commands import build_worker_command
from yolo_runtime_dialog import ensure_ml_runtime


# ---------------------------------------------------------------------------
# cv2 中文路径兼容层
# ---------------------------------------------------------------------------
def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def cv2_imwrite(path, img, ext=None):
    ext = ext or (Path(path).suffix if Path(path).suffix else ".png")
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(str(path))
        return True
    return False


RUNTIME_PATHS = RuntimePaths.from_environment()
PROJECT_DIR = RUNTIME_PATHS.resource_dir
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DETECT_LOG_FILE = RUNTIME_PATHS.logs_dir / "yolo_detect_gui.log"
DETECT_READY_PREFIX = "__YOLO_DETECT_READY__"
DETECT_RESULT_PREFIX = "__YOLO_DETECT_RESULT__"
LOGGER = logging.getLogger("yolo_detect_gui")

# 类别颜色（循环使用，保证不同类别有不同颜色）
BOX_COLORS = [
    (0, 200, 0),     # 绿
    (0, 100, 255),   # 橙
    (255, 50, 50),   # 红
    (200, 0, 200),   # 紫
    (0, 200, 200),   # 黄
    (50, 150, 255),  # 蓝
    (0, 255, 100),   # 青绿
    (150, 100, 0),   # 棕
]


def setup_detect_logging(log_file=None):
    """为模型测试界面单独写日志，方便定位 GUI 中无法弹出的崩溃。"""
    target = Path(log_file) if log_file else DETECT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    log_path = str(target.resolve())
    exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "") == log_path
        for handler in LOGGER.handlers
    )
    if not exists:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        LOGGER.addHandler(handler)


class DetectImageLabel(QLabel):
    """图片显示组件，支持缩放和拖动（复用标注工具的交互逻辑，简化版）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.pixmap = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.panning = False
        self.last_pos = QPoint()
        self.user_zoomed = False
        # 检测结果: [(x1,y1,x2,y2,class_name,conf,color)]
        self.detections = []

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        self.setMinimumSize(300, 200)

    def set_image(self, cv_img):
        """从 OpenCV 图像（BGR）设置显示"""
        if cv_img is None:
            self.pixmap = None
            self.update()
            return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.user_zoomed = False
        self._fit_to_widget()
        self.update()

    def set_detections(self, detections):
        self.detections = detections
        self.update()

    def _fit_to_widget(self):
        if self.pixmap is None:
            return
        pw, ph = self.pixmap.width(), self.pixmap.height()
        ww, wh = self.width(), self.height()
        self.scale = min(ww / pw, wh / ph, 1.0) if pw > 0 and ph > 0 else 1.0
        scaled_w = pw * self.scale
        scaled_h = ph * self.scale
        self.offset = QPoint(int((ww - scaled_w) / 2), int((wh - scaled_h) / 2))
        self.user_zoomed = False
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap is None:
            painter = QPainter(self)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "请选择图片进行检测")
            return

        painter = QPainter(self)
        scaled_w = int(self.pixmap.width() * self.scale)
        scaled_h = int(self.pixmap.height() * self.scale)
        scaled_pixmap = self.pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(self.offset, scaled_pixmap)

        # 绘制检测框
        font = QFont("", 9, QFont.Bold)
        painter.setFont(font)
        for x1, y1, x2, y2, name, conf, color in self.detections:
            sx1 = int(x1 * self.scale + self.offset.x())
            sy1 = int(y1 * self.scale + self.offset.y())
            sx2 = int(x2 * self.scale + self.offset.x())
            sy2 = int(y2 * self.scale + self.offset.y())
            r, g, b = color
            pen = QPen(QColor(r, g, b))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
            # 标签背景
            label = f"{name} {conf:.2f}"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label) + 6
            th = fm.height() + 2
            painter.fillRect(sx1, sy1 - th, tw, th, QColor(r, g, b))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(sx1 + 3, sy1 - 3, label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier):
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()
            self.user_zoomed = True
            self.update()

    def mouseReleaseEvent(self, event):
        if self.panning:
            self.panning = False
            self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        if self.pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        old_scale = self.scale
        new_scale = max(0.1, min(self.scale * factor, 20.0))
        cursor = event.pos()
        img_x = (cursor.x() - self.offset.x()) / old_scale
        img_y = (cursor.y() - self.offset.y()) / old_scale
        self.offset = QPoint(int(cursor.x() - img_x * new_scale), int(cursor.y() - img_y * new_scale))
        self.scale = new_scale
        self.user_zoomed = True
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap and not self.user_zoomed:
            self._fit_to_widget()


class YoloDetectGUI(QMainWindow):
    def __init__(self, model_path=None, runs_dir=None, models_dir=None):
        super().__init__()
        self.models_dir = Path(models_dir) if models_dir else RUNTIME_PATHS.models_dir
        self.workspace_dir = self.models_dir.parent
        self.cache_dir = RUNTIME_PATHS.cache_dir
        setup_detect_logging(RUNTIME_PATHS.logs_dir / "yolo_detect_gui.log")
        self.setWindowTitle("YOLO 模型测试工具")
        self.setGeometry(60, 40, 1400, 850)
        self.setMinimumSize(1000, 680)

        self.model = None
        self.model_path = None
        self.runs_dir = Path(runs_dir) if runs_dir else RUNTIME_PATHS.runs_dir
        self.class_names = {}
        self.image_paths = []
        self.current_idx = 0
        self.current_cv_img = None
        self.conf_threshold = 0.5
        self.save_results = False
        self._detect_process = None
        self._detect_process_model_path = None
        self._detect_output_buffer = ""
        self._detect_worker_ready = False
        self._detect_image_path = None
        self._detect_pending_request = None
        self._detect_active_request = None
        self._detect_request_id = 0
        self._detect_worker_task_id = None
        self._detect_task_manager = TaskManager()
        self._updating_image_combo = False
        self._detect_started_at = None

        self._init_ui()

        # 如果传入了模型路径，自动加载
        if model_path:
            self._load_model(model_path)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.setup_panel = theme.set_panel(QWidget(), "DetectionSetup")
        setup_layout = QGridLayout(self.setup_panel)
        setup_layout.setContentsMargins(14, 12, 14, 12)
        setup_layout.setHorizontalSpacing(8)
        setup_layout.setVerticalSpacing(8)
        setup_layout.setColumnStretch(1, 1)

        setup_layout.addWidget(QLabel("模型"), 0, 0)
        self.model_combo = QComboBox()
        self._scan_models()
        setup_layout.addWidget(self.model_combo, 0, 1)
        self.btn_load_model = ToolButton(
            "加载模型",
            "加载当前选择的模型",
            self.style().standardIcon(QStyle.SP_DialogApplyButton),
        )
        theme.set_button_role(self.btn_load_model, "primary")
        self.btn_load_model.clicked.connect(self._on_load_model)
        setup_layout.addWidget(self.btn_load_model, 0, 2)
        self.btn_browse_model = ToolButton(
            "浏览",
            "选择其他 .pt 模型",
            self.style().standardIcon(QStyle.SP_DialogOpenButton),
        )
        self.btn_browse_model.clicked.connect(self._browse_model)
        setup_layout.addWidget(self.btn_browse_model, 0, 3)

        setup_layout.addWidget(QLabel("图片"), 1, 0)
        self.image_combo = QComboBox()
        self.image_combo.setToolTip("从当前图片目录中直接定位")
        setup_layout.addWidget(self.image_combo, 1, 1)
        self.btn_select_image = ToolButton(
            "单张图片",
            "选择一张图片",
            self.style().standardIcon(QStyle.SP_FileIcon),
        )
        self.btn_select_image.clicked.connect(self._browse_image)
        setup_layout.addWidget(self.btn_select_image, 1, 2)
        self.btn_select_dir = ToolButton(
            "图片目录",
            "选择包含多张图片的目录",
            self.style().standardIcon(QStyle.SP_DirOpenIcon),
        )
        self.btn_select_dir.clicked.connect(self._browse_dir)
        setup_layout.addWidget(self.btn_select_dir, 1, 3)

        setup_layout.addWidget(QLabel("置信度"), 2, 0)
        controls = QWidget()
        controls.setObjectName("DetectionControls")
        conf_row = QHBoxLayout(controls)
        conf_row.setContentsMargins(0, 0, 0, 0)
        conf_row.setSpacing(8)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.setMinimumWidth(160)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.conf_slider, 1)
        self.conf_label = QLabel("0.50")
        self.conf_label.setFixedWidth(40)
        conf_row.addWidget(self.conf_label)
        self.save_check = QCheckBox("自动保存结果图")
        self.save_check.stateChanged.connect(lambda s: setattr(self, 'save_results', s == 2))
        conf_row.addWidget(self.save_check)
        setup_layout.addWidget(controls, 2, 1)

        self.btn_detect = ToolButton(
            "开始检测",
            "使用当前模型检测当前图片 (D)",
            self.style().standardIcon(QStyle.SP_MediaPlay),
        )
        theme.set_button_role(self.btn_detect, "accent")
        self.btn_detect.clicked.connect(self._on_detect)
        setup_layout.addWidget(self.btn_detect, 2, 2, 1, 2)
        main_layout.addWidget(self.setup_panel)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.workspace_panel = theme.set_panel(QWidget(), "DetectionWorkspace")
        self.workspace_panel.setMinimumWidth(560)
        workspace_layout = QVBoxLayout(self.workspace_panel)
        workspace_layout.setContentsMargins(10, 10, 10, 10)
        workspace_layout.setSpacing(8)

        self.image_label = DetectImageLabel(self)
        theme.set_image_canvas(self.image_label)

        toolbar = QWidget()
        toolbar.setObjectName("DetectionToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        self.btn_fit = ToolButton(
            "适应窗口",
            "让图片适应当前画布 (F)",
            self.style().standardIcon(QStyle.SP_DesktopIcon),
        )
        self.btn_fit.clicked.connect(self.image_label._fit_to_widget)
        toolbar_layout.addWidget(self.btn_fit)
        toolbar_layout.addStretch(1)
        self.img_label = QLabel("未选择图片")
        theme.set_text_role(self.img_label, "muted")
        self.img_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        toolbar_layout.addWidget(self.img_label, 1)
        workspace_layout.addWidget(toolbar)
        workspace_layout.addWidget(self.image_label, 1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self.btn_prev = ToolButton(
            "",
            "上一张图片 (P)",
            self.style().standardIcon(QStyle.SP_ArrowLeft),
        )
        self.btn_next = ToolButton(
            "",
            "下一张图片 (N)",
            self.style().standardIcon(QStyle.SP_ArrowRight),
        )
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self.btn_prev)
        self.status_label = StatusBadge("请选择模型和图片", "info")
        self.status_label.setWordWrap(True)
        nav_row.addWidget(self.status_label, 1)
        nav_row.addWidget(self.btn_next)
        workspace_layout.addLayout(nav_row)
        self.main_splitter.addWidget(self.workspace_panel)

        self.inspector_panel = theme.set_panel(QWidget(), "DetectionInspector")
        self.inspector_panel.setMinimumWidth(300)
        self.inspector_panel.setMaximumWidth(430)
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(8)
        inspector_layout.addWidget(SectionHeader("检测结果"))

        self.stats_label = StatusBadge("尚未检测", "info")
        self.stats_label.setWordWrap(True)
        inspector_layout.addWidget(self.stats_label)
        self.elapsed_label = QLabel("本次耗时: --")
        theme.set_text_role(self.elapsed_label, "muted")
        inspector_layout.addWidget(self.elapsed_label)

        self.result_list = QListWidget()
        self.result_list.setFont(QFont("Consolas", 9))
        inspector_layout.addWidget(self.result_list, 1)

        self.btn_save_result = ToolButton(
            "保存结果图",
            "保存带检测框的当前图片",
            self.style().standardIcon(QStyle.SP_DialogSaveButton),
        )
        theme.set_button_role(self.btn_save_result, "primary")
        self.btn_save_result.setEnabled(False)
        self.btn_save_result.clicked.connect(self._save_current)
        inspector_layout.addWidget(self.btn_save_result)
        self.main_splitter.addWidget(self.inspector_panel)

        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([950, 350])
        main_layout.addWidget(self.main_splitter, 1)

        self.image_combo.currentIndexChanged.connect(self._on_image_index_changed)
        self._set_navigation_enabled(True)

        # 快捷键
        QShortcut(QKeySequence("D"), self, self._on_detect)
        QShortcut(QKeySequence("N"), self, self._next_image)
        QShortcut(QKeySequence("P"), self, self._prev_image)
        QShortcut(QKeySequence("F"), self, self.image_label._fit_to_widget)

    def _scan_models(self):
        """扫描训练结果和项目目录下的 .pt 模型文件。"""
        self.model_combo.clear()
        models = []
        # 扫描当前工作区训练结果。
        for runs_dir in [self.runs_dir]:
            if runs_dir.exists():
                for pt in sorted(runs_dir.rglob("best.pt")):
                    if pt not in models:
                        models.append(pt)
                for pt in sorted(runs_dir.rglob("last.pt")):
                    if pt not in models:
                        models.append(pt)
        # 兼容工作区根目录旧模型，并扫描标准 models/。
        for d in [self.workspace_dir, self.models_dir]:
            if d.exists():
                for pt in sorted(d.glob("*.pt")):
                    if pt not in models:
                        models.append(pt)

        models = sorted(models, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        for m in models:
            label = str(m.relative_to(self.workspace_dir)) if m.is_relative_to(self.workspace_dir) else str(m)
            if "best" in m.name:
                label += " (最佳)"
            elif "last" in m.name:
                label += " (末轮)"
            self.model_combo.addItem(label, str(m))

        if not models:
            self.model_combo.addItem("（未找到模型，请点击浏览选择）", "")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", str(self.runs_dir),
            "模型文件 (*.pt);;所有文件 (*)",
        )
        if path:
            self.model_combo.addItem(path, path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
            self._on_load_model()

    def _refresh_image_combo(self):
        self._updating_image_combo = True
        try:
            self.image_combo.clear()
            for image_path in self.image_paths:
                self.image_combo.addItem(image_path.name, str(image_path))
            self.image_combo.setCurrentIndex(
                self.current_idx if self.image_paths else -1
            )
        finally:
            self._updating_image_combo = False
        self._set_navigation_enabled(
            self._detect_active_request is None
            and self._detect_pending_request is None
        )

    def _sync_image_combo(self):
        self._updating_image_combo = True
        try:
            target = self.current_idx if self.image_paths else -1
            if self.image_combo.currentIndex() != target:
                self.image_combo.setCurrentIndex(target)
        finally:
            self._updating_image_combo = False

    def _on_image_index_changed(self, index):
        if self._updating_image_combo:
            return
        if not (0 <= index < len(self.image_paths)):
            self._sync_image_combo()
            return
        if self._detect_active_request is not None or self._detect_pending_request is not None:
            self._sync_image_combo()
            return
        if index == self.current_idx:
            return
        self.current_idx = index
        self._load_and_show()

    def _set_navigation_enabled(self, enabled):
        has_images = bool(self.image_paths)
        self.btn_prev.setEnabled(enabled and has_images and self.current_idx > 0)
        self.btn_next.setEnabled(
            enabled and has_images and self.current_idx < len(self.image_paths) - 1
        )
        self.image_combo.setEnabled(enabled and has_images)
        self.btn_select_image.setEnabled(enabled)
        self.btn_select_dir.setEnabled(enabled)

    def _on_load_model(self):
        path = self.model_combo.currentData()
        if not path:
            return
        self._load_model(path)

    def _load_model(self, path):
        try:
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            if model_path.suffix.lower() != ".pt":
                raise ValueError("请选择 .pt 模型文件")

            # 不在 Qt 主进程中执行 ultralytics/torch 推理，避免 native 崩溃带关整个工具。
            if self.model_path != model_path:
                self._stop_detect_worker()
            self.model = None
            self.model_path = model_path
            self.class_names = {}
            self.status_label.setText(f"模型已选择: {model_path.name}（首次检测会预热）")
            theme.set_status_tone(self.status_label, "success")
            LOGGER.info("模型已选择: %s", model_path)
        except Exception as e:
            LOGGER.exception("模型准备失败")
            QMessageBox.critical(self, "加载失败", f"无法加载模型:\n{e}")
            self.model = None
            self.model_path = None
            self.status_label.setText("模型加载失败")
            theme.set_status_tone(self.status_label, "danger")

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(self.workspace_dir / "dataset" / "images" / "train"),
            f"图片 ({' '.join('*' + e for e in IMAGE_EXTS)});;所有文件 (*)",
        )
        if path:
            self.image_paths = [Path(path)]
            self.current_idx = 0
            self._refresh_image_combo()
            self._load_and_show()

    def _browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择图片目录", str(self.workspace_dir / "dataset" / "images" / "train"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if dir_path:
            d = Path(dir_path)
            self.image_paths = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
            if self.image_paths:
                self.current_idx = 0
                self._refresh_image_combo()
                self._load_and_show()
            else:
                QMessageBox.information(self, "提示", "该目录下未找到图片。")

    def _on_conf_changed(self, val):
        self.conf_threshold = val / 100.0
        self.conf_label.setText(f"{self.conf_threshold:.2f}")

    def _load_and_show(self):
        if not self.image_paths or self.current_idx < 0 or self.current_idx >= len(self.image_paths):
            return
        path = self.image_paths[self.current_idx]
        self._sync_image_combo()
        self.current_cv_img = cv2_imread(path)
        if self.current_cv_img is None:
            self.status_label.setText(f"无法读取: {path.name}")
            theme.set_status_tone(self.status_label, "danger")
            self._set_navigation_enabled(True)
            return
        self.image_label.set_image(self.current_cv_img)
        self.image_label.set_detections([])
        self.result_list.clear()
        self.stats_label.setText("尚未检测")
        theme.set_status_tone(self.stats_label, "info")
        self.elapsed_label.setText("本次耗时: --")
        self.btn_save_result.setEnabled(False)
        self.img_label.setText(f"{self.current_idx + 1}/{len(self.image_paths)} | {path.name}")
        self.status_label.setText(f"已加载: {path.name}（{self.current_cv_img.shape[1]}x{self.current_cv_img.shape[0]}）")
        theme.set_status_tone(self.status_label, "info")
        self._set_navigation_enabled(True)
        # 如果模型已准备，自动检测
        if self.model_path is not None:
            self._on_detect()

    def _on_detect(self):
        if self._detect_active_request is not None:
            self.status_label.setText("正在检测，请稍候...")
            theme.set_status_tone(self.status_label, "info")
            return
        if self.model_path is None:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        if self.current_cv_img is None:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        if not self.image_paths or self.current_idx < 0 or self.current_idx >= len(self.image_paths):
            QMessageBox.warning(self, "提示", "当前图片路径无效，请重新选择图片。")
            return
        worker_program, _ = build_worker_command(
            "detect",
            [],
            resource_dir=PROJECT_DIR,
        )
        if not Path(worker_program).is_file():
            QMessageBox.critical(self, "检测失败", f"缺少检测 Worker:\n{worker_program}")
            return

        self.status_label.setText("检测中...")
        theme.set_status_tone(self.status_label, "info")
        self.stats_label.setText("检测中...")
        theme.set_status_tone(self.stats_label, "info")
        self.elapsed_label.setText("本次耗时: 计算中...")
        self._detect_started_at = time.perf_counter()
        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("检测中...")
        self._set_navigation_enabled(False)

        if not self._ensure_detect_worker():
            self._handle_detect_failure("无法启动检测进程", "")
            return

        self._detect_image_path = self.image_paths[self.current_idx]
        self._detect_pending_request = {
            "image": str(self._detect_image_path),
            "conf": self.conf_threshold,
        }
        if self._detect_worker_ready:
            self._send_detect_request()
        else:
            self.status_label.setText("首次检测：正在加载模型...")
            theme.set_status_tone(self.status_label, "info")

    def _ensure_detect_worker(self):
        if not ensure_ml_runtime(self, "模型测试"):
            return False
        if (
            self._detect_process is not None
            and self._detect_process.state() != QProcess.NotRunning
            and self._detect_process_model_path == self.model_path
        ):
            return True

        self._stop_detect_worker()
        worker_task_id = str(uuid4())
        process = QProcess(self)
        worker_program, worker_args = build_worker_command(
            "detect",
            [
            "--model", str(self.model_path),
            "--serve",
            "--task-id", worker_task_id,
            ],
            resource_dir=PROJECT_DIR,
        )
        process.setProgram(worker_program)
        process.setArguments(worker_args)
        process.setWorkingDirectory(str(PROJECT_DIR))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_detect_process_output)
        process.finished.connect(self._on_detect_process_finished)
        process.errorOccurred.connect(self._on_detect_process_error)

        self._detect_process = process
        self._detect_worker_task_id = worker_task_id
        self._detect_process_model_path = self.model_path
        self._detect_output_buffer = ""
        self._detect_worker_ready = False
        self._detect_task_manager.create_task("detection_worker", task_id=worker_task_id)

        LOGGER.info("启动常驻检测进程: model=%s", self.model_path)
        process.start()
        if not process.waitForStarted(3000):
            LOGGER.error("常驻检测进程启动失败: %s", process.errorString())
            self._stop_detect_worker()
            return False
        return True

    def _send_detect_request(self):
        if self._detect_process is None or self._detect_pending_request is None:
            return
        self._detect_request_id += 1
        request = {
            "cmd": "detect",
            "id": self._detect_request_id,
            "task_id": str(uuid4()),
            **self._detect_pending_request,
        }
        self._detect_active_request = request
        self._detect_pending_request = None
        payload = json.dumps(request, ensure_ascii=False) + "\n"
        self._detect_process.write(payload.encode("utf-8"))
        LOGGER.info("发送检测请求: id=%s image=%s conf=%.4f", request["id"], request["image"], request["conf"])

    def _on_detect_process_output(self):
        process = self._detect_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._detect_output_buffer += text
        while "\n" in self._detect_output_buffer:
            line, self._detect_output_buffer = self._detect_output_buffer.split("\n", 1)
            self._handle_detect_process_line(line.rstrip("\r"))

    def _handle_detect_process_line(self, line):
        line = line.strip()
        if not line:
            return
        try:
            event = decode_task_event(line)
        except ValueError as exc:
            LOGGER.warning("任务事件解析失败: %s", exc)
            return
        if event is not None:
            self._handle_detect_task_event(event)
            return
        if line.startswith(DETECT_READY_PREFIX):
            self._handle_detect_ready(line[len(DETECT_READY_PREFIX):])
            return
        if line.startswith(DETECT_RESULT_PREFIX):
            self._handle_detect_result(line[len(DETECT_RESULT_PREFIX):])
            return
        LOGGER.info("检测进程输出: %s", line)

    def _handle_detect_task_event(self, event):
        try:
            accepted = self._detect_task_manager.accept(event)
        except ValueError:
            LOGGER.warning("忽略不匹配的检测任务事件: %s", event.task_id)
            return
        if not accepted:
            return

        if event.task_id == self._detect_worker_task_id:
            if event.type is TaskEventType.FAILED:
                self._handle_detect_failure(
                    event.message or "检测 worker 启动失败",
                    event.payload.get("traceback", ""),
                    restart_worker=True,
                )
            elif event.type is TaskEventType.CANCELLED and self._detect_process is not None:
                LOGGER.info("检测 worker 已取消")
            return

        active = self._detect_active_request
        if active is None or event.task_id != active.get("task_id"):
            return
        if event.type is TaskEventType.RESULT:
            data = dict(event.payload)
            data.setdefault("id", active.get("id"))
            self._apply_detect_result(data)
            self._finish_detect_request()
        elif event.type is TaskEventType.FAILED:
            self._handle_detect_failure(
                event.message or "检测失败",
                event.payload.get("traceback", ""),
            )

    def _handle_detect_ready(self, payload_text):
        try:
            data = json.loads(payload_text)
        except Exception as e:
            self._handle_detect_failure("检测进程返回异常", str(e), restart_worker=True)
            return

        if not data.get("ok"):
            self._handle_detect_failure(data.get("error", "模型加载失败"), data.get("traceback", ""), restart_worker=True)
            return

        names = data.get("names") or {}
        self.class_names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else {}
        self._detect_worker_ready = True
        LOGGER.info("常驻检测进程已就绪: %s 类", len(self.class_names))
        if self._detect_pending_request is not None:
            self._send_detect_request()
        else:
            self.status_label.setText(f"模型已加载: {Path(self.model_path).name}（{len(self.class_names)} 类）")
            theme.set_status_tone(self.status_label, "success")

    def _handle_detect_result(self, payload_text):
        try:
            data = json.loads(payload_text)
        except Exception as e:
            self._handle_detect_failure("读取检测结果失败", str(e))
            return

        active = self._detect_active_request
        if active is None or data.get("id") != active.get("id"):
            LOGGER.info("忽略过期检测结果: %s", data.get("id"))
            return

        if not data.get("ok"):
            self._handle_detect_failure(data.get("error", "检测失败"), data.get("traceback", ""))
            return

        current_path = self.image_paths[self.current_idx] if self.image_paths and 0 <= self.current_idx < len(self.image_paths) else None
        if self._detect_image_path is not None and current_path != self._detect_image_path:
            LOGGER.info("图片已切换，忽略过期检测结果: %s", self._detect_image_path)
            self.status_label.setText("图片已切换，已忽略上一张检测结果")
            theme.set_status_tone(self.status_label, "warning")
            self._finish_detect_request()
            return

        self._apply_detect_result(data)
        self._finish_detect_request()

    def _on_detect_process_error(self, error):
        if self._detect_process is None:
            return
        LOGGER.error("检测进程错误: %s, %s", error, self._detect_process.errorString())
        if error == QProcess.FailedToStart:
            self._handle_detect_failure("无法启动检测进程", self._detect_process.errorString(), restart_worker=True)

    def _on_detect_process_finished(self, exit_code, exit_status):
        LOGGER.info("检测进程结束: exit_code=%s exit_status=%s", exit_code, exit_status)
        had_request = self._detect_active_request is not None or self._detect_pending_request is not None
        worker_task_id = self._detect_worker_task_id
        if worker_task_id:
            task = self._detect_task_manager.get(worker_task_id)
            if task is not None and task.final_event is None:
                self._detect_task_manager.process_exited(
                    worker_task_id,
                    exit_code,
                    exit_status == QProcess.CrashExit,
                )
        self._detect_process = None
        self._detect_process_model_path = None
        self._detect_worker_task_id = None
        self._detect_worker_ready = False
        if had_request:
            self._handle_detect_failure("检测进程异常退出", "主界面已保留，下一次检测会自动重启检测进程。")

    def _apply_detect_result(self, data):
        detections = []
        result_list_items = []
        names = data.get("names") or {}
        self.class_names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else {}

        for item in data.get("detections", []):
            cls_id = int(item.get("class_id", 0))
            conf = float(item.get("conf", 0.0))
            x1, y1, x2, y2 = [int(v) for v in item.get("xyxy", [0, 0, 0, 0])]
            name = str(item.get("name") or self.class_names.get(cls_id, cls_id))
            color = BOX_COLORS[cls_id % len(BOX_COLORS)]
            detections.append((x1, y1, x2, y2, name, conf, color))
            result_list_items.append(
                f"{name:12s}  {conf:.3f}  ({x1},{y1})-({x2},{y2})"
            )

        self.image_label.set_detections(detections)
        self.result_list.clear()
        if result_list_items:
            self.result_list.addItems(result_list_items)
        else:
            self.result_list.addItem("未检测到目标")

        elapsed_ms = None
        if self._detect_started_at is not None:
            elapsed_ms = max(
                0.0,
                (time.perf_counter() - self._detect_started_at) * 1000.0,
            )
            self.elapsed_label.setText(f"本次耗时: {elapsed_ms:.0f} ms")
        else:
            self.elapsed_label.setText("本次耗时: --")

        from collections import Counter
        cls_counter = Counter(d[4] for d in detections)
        stats = f"检测到 {len(detections)} 个目标"
        if cls_counter:
            top = ", ".join(f"{k}×{v}" for k, v in cls_counter.most_common(5))
            stats += f"\n{top}"
        else:
            stats += "\n可尝试降低置信度，或确认已加载最新训练模型。"
        self.stats_label.setText(stats)
        self.status_label.setText(f"检测完成: {len(detections)} 个目标")
        theme.set_status_tone(self.status_label, "success" if detections else "warning")
        theme.set_status_tone(self.stats_label, "success" if detections else "warning")
        self.btn_save_result.setEnabled(bool(detections))

        if self.save_results and detections:
            self._save_annotated(detections)
        LOGGER.info("检测完成: %s 个目标", len(detections))

    def _handle_detect_failure(self, title, detail, restart_worker=False):
        LOGGER.error("%s\n%s", title, detail)
        self.image_label.set_detections([])
        self.result_list.clear()
        self.result_list.addItem("检测失败，请查看日志")
        self.stats_label.setText("检测失败")
        self.elapsed_label.setText("本次检测失败")
        self.btn_save_result.setEnabled(False)
        self.status_label.setText(f"检测失败: {title}")
        theme.set_status_tone(self.status_label, "danger")
        theme.set_status_tone(self.stats_label, "danger")
        if restart_worker:
            self._stop_detect_worker()
        self._finish_detect_request()

        msg = f"{title}\n\n日志位置:\n{DETECT_LOG_FILE}"
        if detail:
            msg += f"\n\n{detail[:1200]}"
        QMessageBox.critical(self, "检测失败", msg)

    def _finish_detect_request(self):
        self._detect_pending_request = None
        self._detect_active_request = None
        self._detect_image_path = None
        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("开始检测")
        self._set_navigation_enabled(True)
        self._detect_started_at = None

    def _stop_detect_worker(self):
        process = self._detect_process
        worker_task_id = self._detect_worker_task_id
        if worker_task_id:
            self._detect_task_manager.request_cancel(worker_task_id)
        if process is not None:
            try:
                if process.state() != QProcess.NotRunning:
                    process.write(json.dumps({"cmd": "quit"}).encode("utf-8") + b"\n")
                    process.closeWriteChannel()
                    if not process.waitForFinished(800):
                        process.kill()
                        process.waitForFinished(800)
            except Exception:
                LOGGER.exception("停止检测进程失败")
            process.deleteLater()
        if worker_task_id:
            task = self._detect_task_manager.get(worker_task_id)
            if task is not None and task.final_event is None:
                self._detect_task_manager.process_exited(worker_task_id, 0, False)
        self._detect_process = None
        self._detect_process_model_path = None
        self._detect_worker_task_id = None
        self._detect_worker_ready = False
        self._detect_output_buffer = ""
        self._detect_pending_request = None
        self._detect_active_request = None
        self._detect_image_path = None
        if hasattr(self, "btn_detect"):
            self._finish_detect_request()

    def closeEvent(self, event):
        self._stop_detect_worker()
        super().closeEvent(event)

    def _save_annotated(self, detections):
        """保存带检测框的图片"""
        if self.current_cv_img is None:
            return
        img = self.current_cv_img.copy()
        for x1, y1, x2, y2, name, conf, color in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1 - 20), (x1 + len(label) * 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_dir = self.cache_dir / "detect_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"det_{self.image_paths[self.current_idx].name}"
        cv2_imwrite(out_path, img)
        self.status_label.setText(f"已保存: {out_path}")
        theme.set_status_tone(self.status_label, "success")

    def _save_current(self):
        if self.current_cv_img is None or not self.image_label.detections:
            QMessageBox.information(self, "提示", "没有检测结果可保存。")
            return
        self._save_annotated(self.image_label.detections)

    def _prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_and_show()

    def _next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self._load_and_show()


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型测试工具")
    parser.add_argument("--model", default=None, help="模型文件路径")
    parser.add_argument("--runs", default=None, help="训练结果目录")
    parser.add_argument("--models", default=None, help="工作区模型目录")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    theme.apply_app_theme(app)
    window = YoloDetectGUI(
        model_path=args.model,
        runs_dir=args.runs,
        models_dir=args.models,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

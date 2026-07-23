"""
PyQt5 版 YOLO 标注工具

用法：
    python tools/yolo_annotate_gui.py

功能：
- 打开图片目录
- 鼠标框选目标
- 输入标签名
- 缩放（滚轮，向光标位置缩放）+ 拖动平移（空格/中键/Ctrl+左键）
- 保存 YOLO 格式标签
- 上一张/下一张
- 标签管理：删除标签、改名、上移/下移、拖动排序、按名称排序

交互说明：
- 滚轮缩放：向光标位置缩放，光标下的图片区域保持不动
- 空格+拖动：按住空格键后左键拖动可平移查看放大区域
- 中键拖动：直接中键拖动可平移图片（最直观）
- Ctrl+左键：拖动平移图片
- 右键：删除最近的标注框
- 标签列表可拖动排序（DragDrop模式）
- 删除/排序标签会自动更新所有已有标注文件的 class_id 映射
"""

import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QShortcut, QInputDialog, QSplitter,
    QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QStyle
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QTimer, QProcess
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QKeySequence
import cv2
import numpy as np

import yolo_dataset_tools as dataset_tools
import yolo_ui_theme as theme
from yolo_ui_widgets import SectionHeader, StatusBadge, ToolButton
from core.annotation_service import (
    AUTO_DEDUPE_CONTAINMENT_THRESHOLD,
    AUTO_DEDUPE_IOU_THRESHOLD,
    AUTO_DEDUPE_MIN_AREA_RATIO,
    apply_class_changes_atomic,
    build_class_id_mapping,
    dedupe_auto_annotation_candidates as _dedupe_auto_annotation_candidates,
    load_classes_file,
    load_pixel_boxes,
    save_classes_file_atomic,
    save_pixel_boxes_atomic,
)
from core.runtime_paths import RuntimePaths
from core.worker_commands import build_worker_command
from yolo_runtime_dialog import ensure_ml_runtime


# ---------------------------------------------------------------------------
# cv2 中文路径兼容层（Windows 上 cv2.imread/imwrite 不支持 Unicode 路径）
# ---------------------------------------------------------------------------
def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """兼容中文路径的 cv2.imread 替代。使用 np.fromfile + cv2.imdecode。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def cv2_imwrite(path, img, ext=".png", params=None):
    """兼容中文路径的 cv2.imwrite 替代。使用 cv2.imencode + tofile。"""
    ext = Path(path).suffix if Path(path).suffix else ext
    result, buf = cv2.imencode(ext, img, params or [])
    if result:
        buf.tofile(str(path))
        return True
    return False


# 只读程序资源目录；用户数据路径由配置和 RuntimePaths 管理
RUNTIME_PATHS = RuntimePaths.from_environment()
PROJECT_DIR = RUNTIME_PATHS.resource_dir
DETECT_READY_PREFIX = "__YOLO_DETECT_READY__"
DETECT_RESULT_PREFIX = "__YOLO_DETECT_RESULT__"
# 支持的图片扩展名（与 yolo_dataset_tools.IMAGE_EXTS 保持一致）
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class ImageLabel(QLabel):
    """自定义图片显示组件，支持缩放、拖动、框选"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.pixmap = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.dragging = False       # 正在框选标注
        self.panning = False        # 正在拖动平移图片
        self.space_held = False     # 空格键按下（进入拖动模式）
        self.last_pos = QPoint()
        self.start_pos = QPoint()
        self.current_rect = None
        self.boxes = []  # [(x1, y1, x2, y2, class_id)] in original image coords
        self.selected_box_index = -1
        self.edit_mode = None
        self.resize_handle = None
        self.drag_start_original = QPoint()
        self.drag_start_box = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # 需要接收键盘事件
        self.setCursor(Qt.CrossCursor)
        self.user_zoomed = False  # 用户是否手动缩放/平移过（True 后 resize 不再自动 fit）

    def set_image(self, image_path: str):
        self.pixmap = QPixmap(image_path)
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.boxes = []
        self.selected_box_index = -1
        self.edit_mode = None
        self.resize_handle = None
        self.current_rect = None
        self.user_zoomed = False  # 新图片加载时重置标志
        self.update()
        # 首次加载后居中显示
        self._fit_to_widget()

    def set_boxes(self, boxes):
        self.boxes = boxes
        self.selected_box_index = -1
        self.update()

    def select_box(self, index):
        if 0 <= index < len(self.boxes):
            self.selected_box_index = index
        else:
            self.selected_box_index = -1
        self.update()

    def _to_original(self, widget_pos: QPoint) -> QPoint:
        """将窗口坐标转换为原始图片坐标"""
        x = int((widget_pos.x() - self.offset.x()) / self.scale)
        y = int((widget_pos.y() - self.offset.y()) / self.scale)
        return QPoint(x, y)

    def _image_size(self):
        if self.pixmap is None:
            return 0, 0
        return self.pixmap.width(), self.pixmap.height()

    def _clamp_box(self, box):
        w, h = self._image_size()
        x1, y1, x2, y2, class_id = box
        x1 = max(0, min(int(x1), w))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h))
        y2 = max(0, min(int(y2), h))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2, class_id)

    def _box_widget_rect(self, box):
        x1, y1, x2, y2, _ = box
        sx1 = int(x1 * self.scale + self.offset.x())
        sy1 = int(y1 * self.scale + self.offset.y())
        sx2 = int(x2 * self.scale + self.offset.x())
        sy2 = int(y2 * self.scale + self.offset.y())
        return QRect(QPoint(sx1, sy1), QPoint(sx2, sy2)).normalized()

    def _hit_test_box(self, pos):
        handle_size = 8
        edge_size = 6
        handle_hits = []
        edge_hits = []
        for index in range(len(self.boxes) - 1, -1, -1):
            rect = self._box_widget_rect(self.boxes[index])
            area = max(1, rect.width() * rect.height())
            handles = {
                "tl": QRect(rect.left() - handle_size, rect.top() - handle_size, handle_size * 2, handle_size * 2),
                "tr": QRect(rect.right() - handle_size, rect.top() - handle_size, handle_size * 2, handle_size * 2),
                "bl": QRect(rect.left() - handle_size, rect.bottom() - handle_size, handle_size * 2, handle_size * 2),
                "br": QRect(rect.right() - handle_size, rect.bottom() - handle_size, handle_size * 2, handle_size * 2),
            }
            for handle, handle_rect in handles.items():
                if handle_rect.contains(pos):
                    handle_hits.append((area, index, handle))
            if rect.adjusted(-edge_size, -edge_size, edge_size, edge_size).contains(pos):
                near_left = abs(pos.x() - rect.left()) <= edge_size
                near_right = abs(pos.x() - rect.right()) <= edge_size
                near_top = abs(pos.y() - rect.top()) <= edge_size
                near_bottom = abs(pos.y() - rect.bottom()) <= edge_size
                if near_left or near_right or near_top or near_bottom:
                    edge_hits.append((area, index))
        if handle_hits:
            _, index, handle = min(handle_hits, key=lambda item: item[0])
            return index, "resize", handle
        if edge_hits:
            _, index = min(edge_hits, key=lambda item: item[0])
            return index, "move", None
        return -1, None, None

    def _fit_to_widget(self):
        """缩放图片以适应窗口，并将图片居中显示。"""
        if self.pixmap is None:
            return
        pw, ph = self.pixmap.width(), self.pixmap.height()
        ww, wh = self.width(), self.height()
        self.scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
        # 计算居中偏移：图片缩放后的尺寸居中于 widget
        scaled_w = pw * self.scale
        scaled_h = ph * self.scale
        self.offset = QPoint(
            int((ww - scaled_w) / 2),
            int((wh - scaled_h) / 2),
        )
        self.user_zoomed = False  # 重置标志，后续 resize 可自动适应
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap is None:
            return

        painter = QPainter(self)

        # 绘制图片
        scaled_w = int(self.pixmap.width() * self.scale)
        scaled_h = int(self.pixmap.height() * self.scale)
        scaled_pixmap = self.pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(self.offset, scaled_pixmap)

        # 绘制已有框
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)

        for index, (x1, y1, x2, y2, class_id) in enumerate(self.boxes):
            sx1 = int(x1 * self.scale + self.offset.x())
            sy1 = int(y1 * self.scale + self.offset.y())
            sx2 = int(x2 * self.scale + self.offset.x())
            sy2 = int(y2 * self.scale + self.offset.y())
            if index == self.selected_box_index:
                pen = QPen(QColor(255, 210, 0))
                pen.setWidth(3)
                painter.setPen(pen)
            else:
                pen = QPen(QColor(0, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

            class_name = self.parent.get_class_name(class_id)
            painter.drawText(sx1, sy1 - 5, class_name)

            if index == self.selected_box_index:
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setBrush(QColor(255, 210, 0))
                for hx, hy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                    painter.drawRect(hx - 4, hy - 4, 8, 8)
                painter.setBrush(Qt.NoBrush)

        # 绘制当前框
        if self.current_rect:
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

    def mousePressEvent(self, event):
        self.setFocus()  # 确保获得键盘焦点，空格键拖动才生效
        if event.button() == Qt.MiddleButton:
            # 中键拖动平移图片（最直观的交互方式）
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self.space_held or self.panning:
                # 空格+左键拖动平移图片
                self.panning = True
                self.last_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            elif event.modifiers() == Qt.ControlModifier:
                # Ctrl+左键拖动平移图片
                self.panning = True
                self.last_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            else:
                box_index, mode, handle = self._hit_test_box(event.pos())
                if box_index >= 0:
                    self.selected_box_index = box_index
                    self.edit_mode = mode
                    self.resize_handle = handle
                    self.drag_start_original = self._to_original(event.pos())
                    self.drag_start_box = self.boxes[box_index]
                    self.parent.select_box(box_index)
                    self.setCursor(Qt.SizeAllCursor if mode == "move" else Qt.SizeFDiagCursor)
                    self.update()
                    return
                self.selected_box_index = -1
                self.parent.select_box(-1)
                # 普通左键：框选标注
                self.dragging = True
                self.start_pos = event.pos()
                self.current_rect = QRect(self.start_pos, self.start_pos)
                self.update()
        elif event.button() == Qt.RightButton:
            # 删除最近的框
            if self.boxes:
                pos = self._to_original(event.pos())
                closest_idx = -1
                closest_dist = float('inf')
                for i, (x1, y1, x2, y2, _) in enumerate(self.boxes):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dist = (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = i
                if closest_idx >= 0 and closest_dist < 10000:
                    self.boxes.pop(closest_idx)
                    self.selected_box_index = -1
                    self.parent.refresh_box_list()
                    self.update()

    def mouseMoveEvent(self, event):
        if self.edit_mode and 0 <= self.selected_box_index < len(self.boxes) and self.drag_start_box:
            current = self._to_original(event.pos())
            dx = current.x() - self.drag_start_original.x()
            dy = current.y() - self.drag_start_original.y()
            x1, y1, x2, y2, class_id = self.drag_start_box
            if self.edit_mode == "move":
                new_box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy, class_id)
            else:
                if "l" in self.resize_handle:
                    x1 += dx
                if "r" in self.resize_handle:
                    x2 += dx
                if "t" in self.resize_handle:
                    y1 += dy
                if "b" in self.resize_handle:
                    y2 += dy
                new_box = (x1, y1, x2, y2, class_id)
            x1, y1, x2, y2, class_id = self._clamp_box(new_box)
            if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                self.boxes[self.selected_box_index] = (x1, y1, x2, y2, class_id)
            self.update()
        elif self.dragging and self.current_rect:
            self.current_rect = QRect(self.start_pos, event.pos()).normalized()
            self.update()
        elif self.panning:
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()
            self.user_zoomed = True  # 标记用户已手动平移
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self.panning:
            self.panning = False
            if self.space_held:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)
        elif event.button() == Qt.LeftButton and self.edit_mode:
            self.edit_mode = None
            self.resize_handle = None
            self.drag_start_box = None
            self.setCursor(Qt.CrossCursor)
            self.parent.refresh_box_list()
            self.parent.select_box(self.selected_box_index)
            self.update()
        elif event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            if self.current_rect and self.current_rect.width() > 5 and self.current_rect.height() > 5:
                # 转换为原始图片坐标
                p1 = self._to_original(self.current_rect.topLeft())
                p2 = self._to_original(self.current_rect.bottomRight())
                x1, y1, x2, y2 = min(p1.x(), p2.x()), min(p1.y(), p2.y()), max(p1.x(), p2.x()), max(p1.y(), p2.y())

                # 获取标签
                class_name = self.parent.get_current_label()
                if class_name:
                    class_id = self.parent.get_or_create_class_id(class_name)
                    if class_id is not None:
                        self.boxes.append((x1, y1, x2, y2, class_id))
                        self.selected_box_index = len(self.boxes) - 1
                        self.parent.refresh_box_list()
                        self.parent.select_box(self.selected_box_index)
                        self.update()

            self.current_rect = None
            self.update()
        elif event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            if self.space_held:
                self.setCursor(Qt.OpenHandCursor)  # 空格还在，提示可继续拖
            else:
                self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        # 滚轮缩放 —— 向光标位置缩放（光标下的图片区域保持不动）
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15  # 稍快的缩放步进
        old_scale = self.scale
        new_scale = self.scale * factor
        new_scale = max(0.1, min(new_scale, 20.0))

        # 光标在 widget 中的位置
        cursor = event.pos()

        # 光标在原始图片中的位置（缩放前）
        img_x = (cursor.x() - self.offset.x()) / old_scale
        img_y = (cursor.y() - self.offset.y()) / old_scale

        # 新 offset：保持光标下的图片区域位置不变
        self.offset = QPoint(
            int(cursor.x() - img_x * new_scale),
            int(cursor.y() - img_y * new_scale),
        )
        self.scale = new_scale
        self.user_zoomed = True  # 标记用户已手动缩放
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 仅在用户未手动缩放时自动适应窗口，避免打断用户的缩放/平移状态
        if self.pixmap and not self.user_zoomed:
            self._fit_to_widget()

    def keyPressEvent(self, event):
        """空格键按下 → 进入拖动平移模式"""
        if event.key() == Qt.Key_Space and not self.space_held:
            self.space_held = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """空格键释放 → 退出拖动平移模式"""
        if event.key() == Qt.Key_Space and self.space_held:
            self.space_held = False
            self.panning = False
            self.setCursor(Qt.CrossCursor)
            event.accept()
        else:
            super().keyReleaseEvent(event)


class YoloAnnotatorGUI(QMainWindow):
    def __init__(self, dataset_dir=None, initial_image=None):
        super().__init__()
        self.setWindowTitle("YOLO 标注工具")
        self.setGeometry(60, 40, 1500, 900)
        self.setMinimumSize(1100, 720)

        # 支持自定义数据集目录（通过 --dataset 参数传入）
        if dataset_dir:
            base = Path(dataset_dir)
        else:
            config = dataset_tools.load_config()
            base = Path(config.get("dataset_dir") or dataset_tools.get_dataset_dir())
        self.dataset_root = base
        self._initial_image_path = Path(initial_image) if initial_image else None
        initial_split = self._split_for_initial_image(self._initial_image_path)
        self.image_dir = self.dataset_root / "images" / initial_split
        self.label_dir = self.dataset_root / "labels" / initial_split
        self.label_root = self.dataset_root / "labels"
        self.classes_file = self.dataset_root / "classes.txt"
        self.image_paths = []
        self.current_idx = 0
        self.class_names = []
        self.class_to_id = {}
        self.current_label = ""
        self.model_path = None
        self.conf_threshold = 0.5
        self._detect_process = None
        self._detect_process_model_path = None
        self._detect_output_buffer = ""
        self._detect_worker_ready = False
        self._detect_pending_request = None
        self._detect_active_request = None
        self._detect_request_id = 0
        self._detect_image_path = None

        self._load_classes()
        self._init_ui()
        self._load_image_list()

    def _split_for_initial_image(self, image_path):
        if image_path is None:
            return "train"
        try:
            target = image_path.resolve()
        except OSError:
            target = image_path
        for split in ("train", "val", "test", "unlabeled"):
            split_dir = self.dataset_root / "images" / split
            try:
                target.relative_to(split_dir.resolve())
                return split
            except (OSError, ValueError):
                continue
        return "train"

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.main_splitter)

        # 左侧：轻量图片导航，不集中解码缩略图。
        self.navigator_panel = theme.set_panel(QWidget(), "ImageNavigator")
        self.navigator_panel.setMinimumWidth(220)
        self.navigator_panel.setMaximumWidth(320)
        navigator_layout = QVBoxLayout(self.navigator_panel)
        navigator_layout.setContentsMargins(12, 12, 12, 12)
        navigator_layout.setSpacing(8)
        navigator_layout.addWidget(SectionHeader("图片导航"))

        search_layout = QHBoxLayout()
        search_layout.setSpacing(6)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("按文件名搜索")
        search_layout.addWidget(self.search_input, 1)
        self.btn_find_image = ToolButton("查找", "从第一张图片开始查找")
        self.btn_find_next_image = ToolButton(
            "",
            "查找下一个匹配项",
            self.style().standardIcon(QStyle.SP_ArrowForward),
        )
        search_layout.addWidget(self.btn_find_image)
        search_layout.addWidget(self.btn_find_next_image)
        navigator_layout.addLayout(search_layout)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)
        self.image_filter_combo = QComboBox()
        self.image_filter_combo.addItems(["全部", "未标注", "已标注"])
        filter_layout.addWidget(self.image_filter_combo)
        self.navigator_summary_label = QLabel("显示 0 / 0")
        theme.set_text_role(self.navigator_summary_label, "muted")
        filter_layout.addWidget(self.navigator_summary_label, 1, Qt.AlignRight)
        navigator_layout.addLayout(filter_layout)

        self.image_list = QListWidget()
        self.image_list.setUniformItemSizes(True)
        self.image_list.setSpacing(2)
        navigator_layout.addWidget(self.image_list, 1)

        self.btn_open = ToolButton(
            "打开图片目录",
            "选择需要标注的图片目录",
            self.style().standardIcon(QStyle.SP_DirOpenIcon),
        )
        navigator_layout.addWidget(self.btn_open)
        self.main_splitter.addWidget(self.navigator_panel)

        # 中间：画布优先的标注工作区。
        self.workspace_panel = theme.set_panel(QWidget(), "AnnotationWorkspace")
        self.workspace_panel.setMinimumWidth(500)
        workspace_layout = QVBoxLayout(self.workspace_panel)
        workspace_layout.setContentsMargins(10, 10, 10, 10)
        workspace_layout.setSpacing(8)

        toolbar = QWidget()
        toolbar.setObjectName("AnnotationToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        self.btn_save = ToolButton(
            "保存",
            "保存当前图片标注 (S)",
            self.style().standardIcon(QStyle.SP_DialogSaveButton),
        )
        theme.set_button_role(self.btn_save, "primary")
        toolbar_layout.addWidget(self.btn_save)
        self.btn_fit = ToolButton(
            "适应窗口",
            "让图片适应当前画布 (F)",
            self.style().standardIcon(QStyle.SP_DesktopIcon),
        )
        toolbar_layout.addWidget(self.btn_fit)
        toolbar_layout.addStretch(1)
        self.current_image_label = QLabel("未选择图片")
        theme.set_text_role(self.current_image_label, "muted")
        self.current_image_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        toolbar_layout.addWidget(self.current_image_label, 1)
        workspace_layout.addWidget(toolbar)

        self.image_label = ImageLabel(self)
        self.image_label.setMinimumSize(400, 280)
        theme.set_image_canvas(self.image_label)
        workspace_layout.addWidget(self.image_label, 1)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)
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
        self.btn_next_unlabeled = ToolButton(
            "",
            "跳到下一张未标注图片 (U)",
            self.style().standardIcon(QStyle.SP_MediaSkipForward),
        )
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addSpacing(8)
        nav_layout.addWidget(QLabel("跳到"))
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(1, 1)
        self.goto_spin.setFixedWidth(76)
        nav_layout.addWidget(self.goto_spin)
        self.goto_total_label = QLabel("/ 0")
        theme.set_text_role(self.goto_total_label, "muted")
        nav_layout.addWidget(self.goto_total_label)
        self.btn_go_to_image = ToolButton(
            "",
            "跳到指定序号",
            self.style().standardIcon(QStyle.SP_ArrowForward),
        )
        nav_layout.addWidget(self.btn_go_to_image)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.btn_next_unlabeled)
        workspace_layout.addLayout(nav_layout)

        self.status_label = StatusBadge("准备就绪", "info")
        self.status_label.setWordWrap(True)
        workspace_layout.addWidget(self.status_label)
        self.main_splitter.addWidget(self.workspace_panel)

        # 右侧：框、类别和预标注使用独立视图，避免纵向堆叠。
        self.inspector_panel = theme.set_panel(QWidget(), "AnnotationInspector")
        self.inspector_panel.setMinimumWidth(300)
        self.inspector_panel.setMaximumWidth(420)
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(8)
        inspector_layout.addWidget(SectionHeader("标注检查器"))

        self.inspector_tabs = QTabWidget()
        inspector_layout.addWidget(self.inspector_tabs, 1)

        box_tab = QWidget()
        box_layout = QVBoxLayout(box_tab)
        box_layout.setContentsMargins(8, 10, 8, 8)
        box_layout.setSpacing(8)
        self.box_list = QListWidget()
        box_layout.addWidget(self.box_list, 1)
        self.btn_delete_box = ToolButton(
            "删除选中框",
            "删除当前选中的标注框 (Delete)",
            self.style().standardIcon(QStyle.SP_TrashIcon),
        )
        theme.set_button_role(self.btn_delete_box, "danger")
        box_layout.addWidget(self.btn_delete_box)
        self.inspector_tabs.addTab(box_tab, "标注框")

        class_tab = QWidget()
        class_layout = QVBoxLayout(class_tab)
        class_layout.setContentsMargins(8, 10, 8, 8)
        class_layout.setSpacing(8)
        label_row = QHBoxLayout()
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("例如：attack_button")
        label_row.addWidget(self.label_input, 1)
        self.btn_add_label = ToolButton(
            "添加",
            "添加类别",
            self.style().standardIcon(QStyle.SP_DialogApplyButton),
        )
        theme.set_button_role(self.btn_add_label, "primary")
        label_row.addWidget(self.btn_add_label)
        class_layout.addLayout(label_row)

        self.class_list = QListWidget()
        self.class_list.setDragDropMode(QListWidget.InternalMove)
        self.class_list.setDefaultDropAction(Qt.MoveAction)
        self.class_list.setMovement(QListWidget.Free)
        self.class_list.model().rowsMoved.connect(self._on_class_rows_moved)
        class_layout.addWidget(self.class_list, 1)

        class_actions = QHBoxLayout()
        class_actions.setSpacing(6)
        self.btn_delete_class = ToolButton(
            "",
            "删除类别",
            self.style().standardIcon(QStyle.SP_TrashIcon),
        )
        self.btn_rename_class = ToolButton("改名", "重命名选中类别")
        self.btn_move_up = ToolButton(
            "",
            "上移类别",
            self.style().standardIcon(QStyle.SP_ArrowUp),
        )
        self.btn_move_down = ToolButton(
            "",
            "下移类别",
            self.style().standardIcon(QStyle.SP_ArrowDown),
        )
        self.btn_sort_class = ToolButton(
            "排序",
            "按名称排序类别",
            self.style().standardIcon(QStyle.SP_FileDialogListView),
        )
        theme.set_button_role(self.btn_delete_class, "danger")
        class_actions.addWidget(self.btn_delete_class)
        class_actions.addWidget(self.btn_rename_class)
        class_actions.addWidget(self.btn_move_up)
        class_actions.addWidget(self.btn_move_down)
        class_actions.addWidget(self.btn_sort_class)
        class_layout.addLayout(class_actions)
        self.inspector_tabs.addTab(class_tab, "类别")

        auto_tab = QWidget()
        auto_layout = QVBoxLayout(auto_tab)
        auto_layout.setContentsMargins(8, 10, 8, 8)
        auto_layout.setSpacing(8)
        model_label = QLabel("模型")
        model_label.setObjectName("SectionTitle")
        auto_layout.addWidget(model_label)
        self.model_combo = QComboBox()
        auto_layout.addWidget(self.model_combo)

        model_btn_row = QHBoxLayout()
        self.btn_load_model = ToolButton("加载模型", "加载选中的预标注模型")
        self.btn_browse_model = ToolButton(
            "浏览",
            "选择其他模型",
            self.style().standardIcon(QStyle.SP_DialogOpenButton),
        )
        theme.set_button_role(self.btn_load_model, "secondary")
        theme.set_button_role(self.btn_browse_model, "secondary")
        model_btn_row.addWidget(self.btn_load_model)
        model_btn_row.addWidget(self.btn_browse_model)
        auto_layout.addLayout(model_btn_row)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("置信度"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(self.conf_threshold)
        conf_row.addWidget(self.conf_spin)
        auto_layout.addLayout(conf_row)

        self.btn_auto_annotate = ToolButton(
            "预标注当前图",
            "使用当前模型预标注图片",
            self.style().standardIcon(QStyle.SP_MediaPlay),
        )
        theme.set_button_role(self.btn_auto_annotate, "accent")
        auto_layout.addWidget(self.btn_auto_annotate)
        auto_layout.addStretch(1)
        self.inspector_tabs.addTab(auto_tab, "预标注")
        self.main_splitter.addWidget(self.inspector_panel)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([250, 920, 330])

        # 绑定事件
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next_unlabeled.clicked.connect(self.next_unlabeled_image)
        self.btn_save.clicked.connect(self.save_labels)
        self.btn_fit.clicked.connect(self.image_label._fit_to_widget)
        self.btn_open.clicked.connect(self.open_directory)
        self.btn_go_to_image.clicked.connect(self.go_to_image_index)
        self.btn_find_image.clicked.connect(lambda: self.find_image_by_name(from_next=False))
        self.btn_find_next_image.clicked.connect(lambda: self.find_image_by_name(from_next=True))
        self.btn_add_label.clicked.connect(self.add_label_from_input)
        self.btn_delete_box.clicked.connect(self.delete_selected_box)
        self.btn_delete_class.clicked.connect(self.delete_selected_class)
        self.btn_rename_class.clicked.connect(self.rename_selected_class)
        self.btn_move_up.clicked.connect(self.move_class_up)
        self.btn_move_down.clicked.connect(self.move_class_down)
        self.btn_sort_class.clicked.connect(self.sort_classes_by_name)
        self.btn_load_model.clicked.connect(self._on_load_model)
        self.btn_browse_model.clicked.connect(self._browse_model)
        self.btn_auto_annotate.clicked.connect(self._on_auto_annotate)
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.class_list.itemDoubleClicked.connect(self.rename_selected_class)
        self.box_list.itemClicked.connect(self.on_box_selected)
        self.image_list.itemClicked.connect(self.on_image_item_selected)
        self.image_filter_combo.currentTextChanged.connect(
            lambda _text: self._refresh_image_navigator()
        )
        self.search_input.textChanged.connect(
            lambda _text: self._refresh_image_navigator()
        )
        self.label_input.returnPressed.connect(self.add_label_from_input)
        self.search_input.returnPressed.connect(lambda: self.find_image_by_name(from_next=True))
        if self.goto_spin.lineEdit() is not None:
            self.goto_spin.lineEdit().returnPressed.connect(self.go_to_image_index)

        # 快捷键（单键快捷键需防止与标签名输入框冲突）
        QShortcut(QKeySequence("S"), self, self._shortcut_save)
        QShortcut(QKeySequence("N"), self, self._shortcut_next)
        QShortcut(QKeySequence("U"), self, self._shortcut_next_unlabeled)
        QShortcut(QKeySequence("P"), self, self._shortcut_prev)
        QShortcut(QKeySequence("F"), self, self._shortcut_fit)
        QShortcut(QKeySequence("Ctrl+G"), self, self._shortcut_focus_goto)
        QShortcut(QKeySequence("Ctrl+F"), self, self._shortcut_focus_search)
        QShortcut(QKeySequence("Delete"), self, self._shortcut_delete_box)

        self._populate_model_combo()
        self._refresh_class_list()

    def _load_classes(self):
        names, issues = load_classes_file(self.classes_file)
        self.class_names = list(names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self._class_load_issues = issues

    def _save_classes(self, names=None):
        target_names = self.class_names if names is None else list(names)
        try:
            saved_names = save_classes_file_atomic(self.classes_file, target_names)
        except Exception as exc:
            if hasattr(self, "status_label"):
                self.status_label.setText(f"保存类别失败：{exc}")
                QMessageBox.critical(
                    self,
                    "保存类别失败",
                    f"classes.txt 未修改：\n{exc}",
                )
            return False
        self.class_names = list(saved_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self._sync_dataset_config()
        self._refresh_class_list()
        return True

    def _sync_dataset_config(self):
        """同步当前数据集的 data.yaml，并清理旧标签缓存。"""
        try:
            dataset_tools.regenerate_data_yaml(self.dataset_root)
        except Exception as exc:
            self.status_label.setText(f"同步 data.yaml 失败：{exc}")
        self._clear_label_caches()

    def _clear_label_caches(self):
        """标签变化后删除 Ultralytics 生成的缓存，避免训练读到旧标签。"""
        label_root = self.dataset_root / "labels"
        if not label_root.exists():
            return 0
        removed = 0
        for cache_file in label_root.rglob("*.cache"):
            try:
                cache_file.unlink()
                removed += 1
            except Exception:
                pass
        return removed

    def _refresh_class_list(self):
        self.class_list.clear()
        for name in self.class_names:
            self.class_list.addItem(f"{self.class_to_id[name]}: {name}")

    def _load_image_list(self):
        self.image_paths = []
        if self.image_dir.exists():
            self.image_paths = sorted([
                p for p in self.image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ])
        self._update_locator_controls()
        self._refresh_image_navigator()
        if self.image_paths:
            self.load_image(self._initial_image_index())
        else:
            self.status_label.setText(f"未找到图片：{self.image_dir}")
            self.current_image_label.setText("未选择图片")

    def _initial_image_index(self):
        if self._initial_image_path is None:
            return self._last_image_index()
        try:
            target = self._initial_image_path.resolve()
        except OSError:
            target = self._initial_image_path
        for index, image_path in enumerate(self.image_paths):
            try:
                candidate = image_path.resolve()
            except OSError:
                candidate = image_path
            if candidate == target:
                return index
        return self._last_image_index()

    def _update_locator_controls(self):
        total = len(self.image_paths)
        enabled = total > 0
        self.goto_spin.setEnabled(enabled)
        self.btn_go_to_image.setEnabled(enabled)
        self.search_input.setEnabled(enabled)
        self.btn_find_image.setEnabled(enabled)
        self.btn_find_next_image.setEnabled(enabled)
        self.image_filter_combo.setEnabled(enabled)
        self.image_list.setEnabled(enabled)
        self.goto_spin.setRange(1, max(1, total))
        self.goto_total_label.setText(f"/ {total}")
        if not enabled:
            self.goto_spin.setValue(1)

    def _image_navigation_status(self, image_path: Path) -> str:
        return "未标注" if self._is_unlabeled_image(image_path) else "已标注"

    def _refresh_image_navigator(self):
        if not hasattr(self, "image_list"):
            return

        keyword = self.search_input.text().strip().lower()
        status_filter = self.image_filter_combo.currentText() or "全部"
        self._image_item_by_index = {}
        self.image_list.blockSignals(True)
        try:
            self.image_list.clear()
            for index, image_path in enumerate(self.image_paths):
                status = self._image_navigation_status(image_path)
                if keyword and keyword not in image_path.name.lower():
                    continue
                if status_filter != "全部" and status != status_filter:
                    continue

                item = QListWidgetItem(
                    f"{index + 1:05d}  [{status}]  {image_path.name}"
                )
                item.setData(Qt.UserRole, index)
                item.setToolTip(str(image_path))
                item.setSizeHint(QSize(0, 34))
                item.setForeground(
                    QColor("#047857" if status == "已标注" else "#9a6700")
                )
                self.image_list.addItem(item)
                self._image_item_by_index[index] = item

            self.navigator_summary_label.setText(
                f"显示 {self.image_list.count()} / {len(self.image_paths)}"
            )
            self._sync_image_navigator_selection()
        finally:
            self.image_list.blockSignals(False)

    def _sync_image_navigator_selection(self):
        if not hasattr(self, "image_list"):
            return
        item = getattr(self, "_image_item_by_index", {}).get(self.current_idx)
        self.image_list.blockSignals(True)
        try:
            if item is None:
                self.image_list.setCurrentRow(-1)
            else:
                self.image_list.setCurrentItem(item)
                self.image_list.scrollToItem(item)
        finally:
            self.image_list.blockSignals(False)

    def on_image_item_selected(self, item):
        target_index = item.data(Qt.UserRole) if item is not None else None
        if not isinstance(target_index, int) or not (
            0 <= target_index < len(self.image_paths)
        ):
            self._sync_image_navigator_selection()
            return
        if target_index == self.current_idx:
            return
        if not self.save_labels():
            self._sync_image_navigator_selection()
            return
        self.load_image(target_index)

    def _image_dir_key(self):
        try:
            return str(self.image_dir.resolve())
        except Exception:
            return str(self.image_dir)

    def _last_image_index(self):
        try:
            cfg = dataset_tools.load_config()
            last_positions = cfg.get("annotator_last_images", {})
            last_image = last_positions.get(self._image_dir_key())
        except Exception:
            last_image = None
        if not last_image:
            return 0

        try:
            target = str(Path(last_image).resolve()).lower()
        except Exception:
            target = str(last_image).lower()
        for idx, path in enumerate(self.image_paths):
            try:
                current = str(path.resolve()).lower()
            except Exception:
                current = str(path).lower()
            if current == target:
                return idx

        target_name = Path(last_image).name.lower()
        for idx, path in enumerate(self.image_paths):
            if path.name.lower() == target_name:
                return idx
        return 0

    def _save_last_image_position(self, image_path):
        try:
            cfg = dataset_tools.load_config()
            last_positions = cfg.get("annotator_last_images", {})
            if not isinstance(last_positions, dict):
                last_positions = {}
            try:
                saved_path = str(Path(image_path).resolve())
            except Exception:
                saved_path = str(image_path)
            last_positions[self._image_dir_key()] = saved_path
            cfg["annotator_last_images"] = last_positions
            dataset_tools.save_config(cfg)
        except Exception:
            pass

    def _model_search_dirs(self):
        dirs = [PROJECT_DIR]
        try:
            cfg = dataset_tools.load_config()
            for key in ("models_dir", "runs_dir"):
                value = cfg.get(key, "")
                if value:
                    directory = Path(value)
                    if directory not in dirs:
                        dirs.insert(0, directory)
        except Exception:
            pass
        return [d for d in dirs if d.exists()]

    def _populate_model_combo(self):
        self.model_combo.clear()
        models = []
        for d in self._model_search_dirs():
            for pt in sorted(d.glob("*.pt")):
                if pt not in models:
                    models.append(pt)
            for name in ("best.pt", "last.pt"):
                for pt in sorted(d.rglob(name)):
                    if pt not in models:
                        models.append(pt)
        models = sorted(models, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for model in models:
            label = str(model.relative_to(PROJECT_DIR)) if model.is_relative_to(PROJECT_DIR) else str(model)
            if model.name == "best.pt":
                label += " (最佳)"
            elif model.name == "last.pt":
                label += " (末轮)"
            self.model_combo.addItem(label, str(model))
        if not models:
            self.model_combo.addItem("未找到模型，请点浏览选择 .pt", "")
        elif self.model_path is None:
            self._set_model_path(models[0])

    def _browse_model(self):
        search_dirs = self._model_search_dirs()
        start_dir = search_dirs[0] if search_dirs else Path(dataset_tools.get_runs_dir())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            str(start_dir),
            "模型文件 (*.pt);;所有文件 (*)",
        )
        if path:
            self.model_combo.addItem(path, path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
            self._set_model_path(Path(path))

    def _on_load_model(self):
        path = self.model_combo.currentData()
        if path:
            self._set_model_path(Path(path))

    def _set_model_path(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            QMessageBox.warning(self, "模型不存在", f"找不到模型文件:\n{model_path}")
            return
        if model_path.suffix.lower() != ".pt":
            QMessageBox.warning(self, "模型格式不支持", "请选择 .pt 模型文件。")
            return
        if self.model_path != model_path:
            self._stop_detect_worker()
        self.model_path = model_path
        self.status_label.setText(f"预标注模型已选择: {model_path.name}")

    def _on_conf_changed(self, value):
        self.conf_threshold = float(value)

    def _on_auto_annotate(self):
        if not self.image_paths:
            QMessageBox.information(self, "提示", "请先打开图片目录。")
            return
        if self.model_path is None:
            self._on_load_model()
        if self.model_path is None:
            QMessageBox.information(self, "提示", "请先选择并加载一个模型。")
            return
        if self.image_label.boxes:
            reply = QMessageBox.question(
                self,
                "当前图已有标注",
                f"当前图片已有 {len(self.image_label.boxes)} 个框。\n\n是否把模型检测结果追加到当前标注中？\n工具会自动跳过和已有框高度重叠的模型框。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.status_label.setText("已取消预标注，当前标注未改变")
                return

        if not self._ensure_detect_worker():
            QMessageBox.critical(self, "预标注失败", "无法启动检测进程。")
            return

        image_path = self.image_paths[self.current_idx]
        self._detect_image_path = image_path
        self._detect_pending_request = {
            "image": str(image_path),
            "conf": self.conf_threshold,
        }
        self.btn_auto_annotate.setEnabled(False)
        self.btn_auto_annotate.setText("预标注中...")
        self.status_label.setText("模型预标注中...")
        if self._detect_worker_ready:
            self._send_detect_request()
        else:
            self.status_label.setText("首次预标注：正在加载模型...")

    def _ensure_detect_worker(self):
        if not ensure_ml_runtime(self, "自动预标注"):
            return False
        if (
            self._detect_process is not None
            and self._detect_process.state() != QProcess.NotRunning
            and self._detect_process_model_path == self.model_path
        ):
            return True

        self._stop_detect_worker()
        process = QProcess(self)
        worker_program, worker_args = build_worker_command(
            "detect",
            [
            "--model", str(self.model_path),
            "--serve",
            ],
            resource_dir=PROJECT_DIR,
        )
        if not Path(worker_program).is_file():
            QMessageBox.critical(self, "缺少文件", f"找不到检测 Worker:\n{worker_program}")
            return
        process.setProgram(worker_program)
        process.setArguments(worker_args)
        process.setWorkingDirectory(str(PROJECT_DIR))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_detect_process_output)
        process.finished.connect(self._on_detect_process_finished)
        process.errorOccurred.connect(self._on_detect_process_error)

        self._detect_process = process
        self._detect_process_model_path = self.model_path
        self._detect_output_buffer = ""
        self._detect_worker_ready = False
        process.start()
        if not process.waitForStarted(3000):
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
            **self._detect_pending_request,
        }
        self._detect_active_request = request
        self._detect_pending_request = None
        payload = json.dumps(request, ensure_ascii=False) + "\n"
        self._detect_process.write(payload.encode("utf-8"))

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
        if line.startswith(DETECT_READY_PREFIX):
            self._handle_detect_ready(line[len(DETECT_READY_PREFIX):])
        elif line.startswith(DETECT_RESULT_PREFIX):
            self._handle_detect_result(line[len(DETECT_RESULT_PREFIX):])

    def _handle_detect_ready(self, payload_text):
        try:
            data = json.loads(payload_text)
        except Exception as exc:
            self._handle_detect_failure(f"检测进程返回异常: {exc}")
            return
        if not data.get("ok"):
            self._handle_detect_failure(data.get("error", "模型加载失败"))
            return
        self._detect_worker_ready = True
        if self._detect_pending_request is not None:
            self._send_detect_request()
        else:
            self.status_label.setText(f"模型已加载: {Path(self.model_path).name}")

    def _handle_detect_result(self, payload_text):
        try:
            data = json.loads(payload_text)
        except Exception as exc:
            self._handle_detect_failure(f"读取检测结果失败: {exc}")
            return
        active = self._detect_active_request
        if active is None or data.get("id") != active.get("id"):
            return
        if not data.get("ok"):
            self._handle_detect_failure(data.get("error", "检测失败"))
            return
        current_path = self.image_paths[self.current_idx] if self.image_paths else None
        if current_path != self._detect_image_path:
            self._finish_detect_request()
            self.status_label.setText("图片已切换，已忽略过期预标注结果")
            return
        self._apply_auto_annotations(data)
        self._finish_detect_request()

    def _apply_auto_annotations(self, data):
        candidates = []
        for item in data.get("detections", []):
            name = str(item.get("name") or item.get("class_id", "")).strip()
            if not name:
                continue
            xyxy = item.get("xyxy", [])
            if len(xyxy) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
                conf = float(item.get("conf", 0.0))
            except (TypeError, ValueError):
                continue
            box = self.image_label._clamp_box((x1, y1, x2, y2, -1))
            if abs(box[2] - box[0]) < 4 or abs(box[3] - box[1]) < 4:
                continue
            candidates.append({
                "name": name,
                "box": box[:4],
                "conf": conf,
            })

        kept_candidates, skipped_existing, skipped_duplicate = _dedupe_auto_annotation_candidates(
            candidates,
            self.image_label.boxes,
        )

        new_class_names = self.class_names[:]
        for candidate in kept_candidates:
            name = candidate["name"]
            if name not in new_class_names:
                new_class_names.append(name)
        if new_class_names != self.class_names and not self._save_classes(new_class_names):
            return

        new_boxes = []
        for candidate in kept_candidates:
            class_id = self.class_to_id[candidate["name"]]
            x1, y1, x2, y2 = candidate["box"]
            box = (x1, y1, x2, y2, class_id)
            new_boxes.append(box)
        if new_boxes:
            self.image_label.boxes.extend(new_boxes)
            self.image_label.selected_box_index = len(self.image_label.boxes) - 1
        self.refresh_box_list()
        self.select_box(self.image_label.selected_box_index)
        self.image_label.update()
        skipped_total = skipped_existing + skipped_duplicate
        if new_boxes:
            message = f"已预标注 {len(new_boxes)} 个框"
            if skipped_total:
                message += f"，已跳过重复框 {skipped_total} 个"
            self.status_label.setText(message + "，请检查调整后保存")
        elif skipped_total:
            self.status_label.setText(f"没有新增框，已跳过重复框 {skipped_total} 个")
        else:
            self.status_label.setText("模型没有检测到目标，可降低置信度后重试")

    def _handle_detect_failure(self, message):
        self._finish_detect_request()
        QMessageBox.critical(self, "预标注失败", str(message))
        self.status_label.setText(f"预标注失败: {message}")

    def _finish_detect_request(self):
        self._detect_pending_request = None
        self._detect_active_request = None
        self._detect_image_path = None
        if hasattr(self, "btn_auto_annotate"):
            self.btn_auto_annotate.setEnabled(True)
            self.btn_auto_annotate.setText("预标注当前图")

    def _on_detect_process_error(self, error):
        if error == QProcess.FailedToStart:
            self._handle_detect_failure("检测进程启动失败")

    def _on_detect_process_finished(self, exit_code, exit_status):
        had_request = self._detect_active_request is not None or self._detect_pending_request is not None
        self._detect_process = None
        self._detect_process_model_path = None
        self._detect_worker_ready = False
        if had_request:
            self._handle_detect_failure("检测进程异常退出，下一次会自动重启")

    def _stop_detect_worker(self):
        process = self._detect_process
        self._detect_process = None
        self._detect_process_model_path = None
        self._detect_worker_ready = False
        self._detect_output_buffer = ""
        self._detect_pending_request = None
        self._detect_active_request = None
        self._detect_image_path = None
        if process is not None:
            try:
                if process.state() != QProcess.NotRunning:
                    process.write(json.dumps({"cmd": "quit"}).encode("utf-8") + b"\n")
                    process.closeWriteChannel()
                    if not process.waitForFinished(800):
                        process.kill()
                        process.waitForFinished(800)
            except Exception:
                pass
            process.deleteLater()
        self._finish_detect_request()

    def load_image(self, idx):
        if not self.image_paths or idx < 0 or idx >= len(self.image_paths):
            return

        self.current_idx = idx
        image_path = self.image_paths[idx]
        self.image_label.set_image(str(image_path))
        self.image_label._fit_to_widget()
        label_issues = self._load_labels(image_path)
        self._update_locator_controls()
        self.goto_spin.setValue(idx + 1)
        self._save_last_image_position(image_path)
        self.current_image_label.setText(
            f"{idx + 1} / {len(self.image_paths)}   {image_path.name}"
        )
        message = f"{idx + 1}/{len(self.image_paths)} | {image_path.name}"
        class_issues = getattr(self, "_class_load_issues", ())
        if class_issues:
            message += " | classes.txt 读取失败"
        if label_issues:
            message += f" | 标签问题 {len(label_issues)} 处，已加载有效框"
        self.status_label.setText(message)
        self._sync_image_navigator_selection()

    def _load_labels(self, image_path: Path):
        self.image_label.boxes = []
        label_path = self._label_path_for_image(image_path)
        if not label_path.exists():
            self.image_label.update()
            self.refresh_box_list()
            return ()

        img = cv2_imread(image_path)
        if img is None:
            return ()
        h, w = img.shape[:2]
        boxes, issues = load_pixel_boxes(
            label_path,
            image_size=(w, h),
            class_count=len(self.class_names) if self.class_names else None,
        )
        self.image_label.boxes = list(boxes)

        self.image_label.update()
        self.refresh_box_list()
        return issues

    def _label_path_for_image(self, image_path: Path):
        return self.label_dir / (image_path.stem + ".txt")

    def _is_unlabeled_image(self, image_path: Path):
        label_path = self._label_path_for_image(image_path)
        if not label_path.exists():
            return True
        try:
            return label_path.read_text(encoding="utf-8").strip() == ""
        except Exception:
            return True

    def save_labels(self):
        if not self.image_paths:
            return True

        image_path = self.image_paths[self.current_idx]
        img = cv2_imread(image_path)
        if img is None:
            QMessageBox.warning(self, "警告", f"无法读取图片：{image_path}")
            return False

        h, w = img.shape[:2]
        label_path = self._label_path_for_image(image_path)
        try:
            saved_boxes = save_pixel_boxes_atomic(
                label_path,
                self.image_label.boxes,
                image_size=(w, h),
            )
        except Exception as exc:
            self.status_label.setText(f"保存失败，原标签未修改：{exc}")
            QMessageBox.critical(
                self,
                "保存标注失败",
                f"当前标签没有写入，请勿继续翻页：\n{exc}",
            )
            return False

        self.image_label.boxes = list(saved_boxes)
        self._clear_label_caches()
        self.status_label.setText(f"已保存：{label_path}")
        self._refresh_image_navigator()
        return True

    def next_image(self):
        if not self.save_labels():
            return
        if self.current_idx < len(self.image_paths) - 1:
            self.load_image(self.current_idx + 1)

    def prev_image(self):
        if not self.save_labels():
            return
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_unlabeled_image(self):
        if not self.image_paths:
            return
        if not self.save_labels():
            return
        total = len(self.image_paths)
        for step in range(1, total):
            idx = (self.current_idx + step) % total
            if self._is_unlabeled_image(self.image_paths[idx]):
                self.load_image(idx)
                self.status_label.setText(
                    f"已跳到未标注图片: {idx + 1}/{total} | {self.image_paths[idx].name}"
                )
                return
        self.status_label.setText("没有找到其他未标注图片")

    def go_to_image_index(self):
        if not self.image_paths:
            return
        target_idx = self.goto_spin.value() - 1
        if target_idx == self.current_idx:
            image_path = self.image_paths[self.current_idx]
            self.status_label.setText(f"已在目标图片: {target_idx + 1}/{len(self.image_paths)} | {image_path.name}")
            return
        if 0 <= target_idx < len(self.image_paths):
            if not self.save_labels():
                return
            self.load_image(target_idx)
            image_path = self.image_paths[target_idx]
            self.status_label.setText(f"已跳转: {target_idx + 1}/{len(self.image_paths)} | {image_path.name}")

    def find_image_by_name(self, from_next=False):
        if not self.image_paths:
            return
        keyword = self.search_input.text().strip().lower()
        if not keyword:
            self.status_label.setText("请输入要搜索的文件名片段")
            return

        total = len(self.image_paths)
        start_idx = (self.current_idx + 1) % total if from_next else 0
        for step in range(total):
            idx = (start_idx + step) % total
            if keyword in self.image_paths[idx].name.lower():
                if idx != self.current_idx:
                    if not self.save_labels():
                        return
                    self.load_image(idx)
                self.status_label.setText(
                    f"已找到: {idx + 1}/{total} | {self.image_paths[idx].name}"
                )
                return
        self.status_label.setText(f"未找到文件名包含「{keyword}」的图片")

    def prompt_open_directory_on_empty(self):
        """如果图片目录为空，启动时提示选择目录"""
        if not self.image_paths:
            reply = QMessageBox.question(
                self,
                "未找到图片",
                f"默认图片目录为空或不存在：\n{self.image_dir}\n\n是否选择其他目录？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.open_directory()

    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录", str(self.image_dir))
        if dir_path:
            self.image_dir = Path(dir_path)
            if self.image_dir.parent.name.lower() == "images":
                # 标准 YOLO 结构：dataset/images/{split} -> dataset/labels/{split}
                self.dataset_root = self.image_dir.parent.parent
                self.label_dir = self.dataset_root / "labels" / self.image_dir.name
                self.label_root = self.dataset_root / "labels"
                self.classes_file = self.dataset_root / "classes.txt"
                self._load_classes()
                self._refresh_class_list()
            else:
                # 非标准目录只在该目录下写 labels/classes.txt，避免误把旧数据集当成写入目标。
                self.dataset_root = self.image_dir
                self.label_dir = self.image_dir / "labels"
                self.label_root = self.label_dir
                self.classes_file = self.image_dir / "classes.txt"
                self._load_classes()
                self._refresh_class_list()
            self._load_image_list()

    def get_current_label(self) -> str:
        return self.label_input.text().strip()

    def get_or_create_class_id(self, name: str) -> int | None:
        if name not in self.class_to_id:
            new_names = self.class_names + [name]
            if not self._save_classes(new_names):
                return None
        return self.class_to_id[name]

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def add_label_from_input(self):
        name = self.label_input.text().strip()
        if name:
            class_id = self.get_or_create_class_id(name)
            if class_id is None:
                return
            self.label_input.clear()
            self.status_label.setText(f"已添加标签：{name}")

    def on_class_selected(self, item):
        text = item.text()
        name = text.split(": ", 1)[1]
        self.label_input.setText(name)

    def refresh_box_list(self):
        selected = self.image_label.selected_box_index
        self.box_list.clear()
        for i, (x1, y1, x2, y2, class_id) in enumerate(self.image_label.boxes):
            name = self.get_class_name(class_id)
            self.box_list.addItem(f"{i}: {name} ({x1},{y1})-({x2},{y2})")
        if 0 <= selected < self.box_list.count():
            self.box_list.setCurrentRow(selected)

    def select_box(self, index):
        self.image_label.select_box(index)
        if 0 <= index < self.box_list.count():
            self.box_list.setCurrentRow(index)
        else:
            self.box_list.clearSelection()

    def on_box_selected(self, item):
        row = self.box_list.row(item)
        self.image_label.select_box(row)

    def delete_selected_box(self):
        row = self.box_list.currentRow()
        if row >= 0 and row < len(self.image_label.boxes):
            self.image_label.boxes.pop(row)
            self.image_label.selected_box_index = -1
            self.refresh_box_list()
            self.image_label.update()

    # ---------------------------------------------------------------
    # 快捷键包装：当焦点在文本输入框时不触发单键快捷键
    # ---------------------------------------------------------------
    def _is_text_widget_focused(self):
        """检查当前焦点是否在文本输入控件上（QLineEdit/QTextEdit）。
        若是，则单键快捷键不应触发，以免与正常打字冲突。
        """
        from PyQt5.QtWidgets import QLineEdit, QTextEdit
        widget = QApplication.focusWidget()
        return isinstance(widget, (QLineEdit, QTextEdit))

    def _shortcut_save(self):
        if not self._is_text_widget_focused():
            self.save_labels()

    def _shortcut_next(self):
        if not self._is_text_widget_focused():
            self.next_image()

    def _shortcut_next_unlabeled(self):
        if not self._is_text_widget_focused():
            self.next_unlabeled_image()

    def _shortcut_prev(self):
        if not self._is_text_widget_focused():
            self.prev_image()

    def _shortcut_fit(self):
        if not self._is_text_widget_focused():
            self.image_label._fit_to_widget()

    def _shortcut_focus_goto(self):
        if not self.image_paths:
            return
        self.goto_spin.setFocus()
        self.goto_spin.selectAll()

    def _shortcut_focus_search(self):
        if not self.image_paths:
            return
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _shortcut_delete_box(self):
        if not self._is_text_widget_focused():
            self.delete_selected_box()

    # ---------------------------------------------------------------
    # 标签管理：删除、上移、下移、排序
    # ---------------------------------------------------------------
    def _build_id_mapping(self, old_names, new_names):
        """构建 old_id → new_id 的映射表，用于更新所有标签文件。"""
        return build_class_id_mapping(old_names, new_names)

    def _iter_label_files(self):
        """遍历当前标签根目录下需要批量更新的 YOLO txt 文件。"""
        if not self.label_root.exists():
            return []
        files = []
        direct_txt = [
            p for p in self.label_root.glob("*.txt")
            if p.is_file() and p.name != "classes.txt"
        ]
        files.extend(direct_txt)
        for split_dir in self.label_root.iterdir():
            if not split_dir.is_dir() or split_dir.name == "unlabeled":
                continue
            files.extend([
                p for p in split_dir.glob("*.txt")
                if p.is_file() and p.name != "classes.txt"
            ])
        return sorted(set(files))

    def _backup_label_state(self, reason):
        """标签结构变化前自动备份，降低批量改名/排序/删除的风险。"""
        try:
            backup_dir = dataset_tools.backup_dataset_state(self.dataset_root, reason=reason)
            self.status_label.setText(f"已自动备份：{backup_dir}")
        except Exception as exc:
            QMessageBox.warning(self, "备份失败", f"自动备份失败，已取消本次操作：\n{exc}")
            return False
        return True

    def _apply_class_reorder(
        self,
        new_names,
        backup_reason="class_reorder",
        deleted_ids=(),
    ):
        """应用新的标签顺序：保存 classes.txt、更新所有标签文件、刷新界面。"""
        old_names = self.class_names[:]
        if old_names == new_names:
            return
        if backup_reason and not self._backup_label_state(backup_reason):
            self._refresh_class_list()
            return False
        id_mapping = self._build_id_mapping(old_names, new_names)
        try:
            updated_files = apply_class_changes_atomic(
                classes_path=self.classes_file,
                new_names=new_names,
                label_paths=self._iter_label_files(),
                id_mapping=id_mapping,
                deleted_ids=deleted_ids,
            )
        except Exception as exc:
            self._refresh_class_list()
            self.status_label.setText(f"标签顺序更新失败，原文件已保留：{exc}")
            QMessageBox.critical(
                self,
                "更新标签失败",
                f"类别和标签文件没有完成更新：\n{exc}",
            )
            return False

        self.class_names = list(new_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self._sync_dataset_config()
        self._refresh_class_list()

        # 更新当前图片上的 boxes 的 class_id
        deleted = set(deleted_ids)
        new_boxes = []
        for x1, y1, x2, y2, old_id in self.image_label.boxes:
            if old_id in deleted:
                continue
            new_id = id_mapping.get(old_id, old_id)
            new_boxes.append((x1, y1, x2, y2, new_id))
        self.image_label.boxes = new_boxes
        self.refresh_box_list()
        self.image_label.update()

        msg = f"标签顺序已更新"
        if updated_files > 0:
            msg += f"，已同步修改 {updated_files} 个标签文件"
        self.status_label.setText(msg)
        return True

    def delete_selected_class(self):
        """删除选中的标签，自动更新所有标签文件的 class_id 映射。"""
        row = self.class_list.currentRow()
        if row < 0:
            self.status_label.setText("请先在标签列表中选择要删除的标签")
            return
        name_to_delete = self.class_names[row]

        # 确认对话框：说明后果
        reply = QMessageBox.warning(
            self, "删除标签",
            f"确定要删除标签「{name_to_delete}」？\n\n"
            f"删除后：\n"
            f"  • 所有标注文件中该类别的框将一并删除\n"
            f"  • 其他标签的编号（class_id）可能发生变化\n"
            f"  • 数据集中所有 .txt 标签文件将自动更新\n\n"
            f"此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if not self._backup_label_state(f"delete_class_{name_to_delete}"):
            return

        deleted_id = self.class_to_id[name_to_delete]
        new_names = [n for n in self.class_names if n != name_to_delete]
        if not self._apply_class_reorder(
            new_names,
            backup_reason=None,
            deleted_ids={deleted_id},
        ):
            return

        self.refresh_box_list()
        self.image_label.update()
        self.status_label.setText(f"已删除标签「{name_to_delete}」并更新所有标注文件")

    def rename_selected_class(self):
        """改名为选中标签——弹出输入框，仅改名不改 class_id，无需更新标注文件。"""
        row = self.class_list.currentRow()
        if row < 0:
            self.status_label.setText("请先在标签列表中选择要改名的标签")
            return
        old_name = self.class_names[row]

        new_name, ok = QInputDialog.getText(
            self, "改名",
            f"将标签「{old_name}」改名为：",
            text=old_name,  # 默认填入原名方便修改
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()

        # 检查是否重名
        if new_name in self.class_names and new_name != old_name:
            QMessageBox.warning(self, "重名", f"标签「{new_name}」已存在，不能重复！")
            return
        if not self._backup_label_state(f"rename_class_{old_name}"):
            return

        # 仅改名，class_id 不变，不需要更新标注文件中的 class_id
        new_names = self.class_names[:]
        new_names[row] = new_name
        if not self._save_classes(new_names):
            return
        self.refresh_box_list()
        self.image_label.update()
        self.status_label.setText(f"标签已改名：「{old_name}」→「{new_name}」")

    def move_class_up(self):
        """将选中的标签上移一位。"""
        row = self.class_list.currentRow()
        if row <= 0:
            return
        new_names = self.class_names[:]
        new_names[row], new_names[row - 1] = new_names[row - 1], new_names[row]
        self._apply_class_reorder(new_names)
        self.class_list.setCurrentRow(row - 1)

    def move_class_down(self):
        """将选中的标签下移一位。"""
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self.class_names) - 1:
            return
        new_names = self.class_names[:]
        new_names[row], new_names[row + 1] = new_names[row + 1], new_names[row]
        self._apply_class_reorder(new_names)
        self.class_list.setCurrentRow(row + 1)

    def sort_classes_by_name(self):
        """按名称（拼音序）排序标签，自动更新所有标签文件。"""
        if len(self.class_names) <= 1:
            return
        reply = QMessageBox.question(
            self, "排序标签",
            f"按名称排序将改变标签编号（class_id），\n"
            f"数据集中所有 .txt 标签文件将自动更新。\n\n"
            f"确定排序？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        new_names = sorted(self.class_names)  # Python sorted 对中文按 Unicode 顺序
        self._apply_class_reorder(new_names)

    def _on_class_rows_moved(self, parent, start, end, dest, row):
        """QListWidget 内部拖动排序回调——读取列表新顺序并更新。"""
        # 从 QListWidget 的当前显示顺序重建 class_names
        new_names = []
        for i in range(self.class_list.count()):
            text = self.class_list.item(i).text()
            name = text.split(": ", 1)[1]
            new_names.append(name)
        self._apply_class_reorder(new_names)

    def closeEvent(self, event):
        """关闭窗口前自动保存当前图片的标注，防止数据丢失。"""
        if self.image_paths and not self.save_labels():
            reply = QMessageBox.question(
                self,
                "标注尚未保存",
                "当前标注保存失败，确定仍要关闭窗口吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self._stop_detect_worker()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO 标注工具")
    parser.add_argument("--dataset", default=None, help="数据集目录路径（含 images/labels/classes.txt）")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    theme.apply_app_theme(app)
    window = YoloAnnotatorGUI(dataset_dir=args.dataset)
    window.show()

    # 如果启动时图片目录为空，自动提示选择目录
    QTimer.singleShot(100, window.prompt_open_directory_on_empty)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

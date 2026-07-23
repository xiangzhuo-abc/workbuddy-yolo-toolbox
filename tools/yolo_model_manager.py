"""官方目标检测模型管理窗口。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import yolo_ui_theme as theme
from core.model_registry import (
    MODEL_CATALOG,
    DownloadableModel,
    ModelDownloadCancelled,
    ModelDownloadError,
    download_model,
    installed_models,
    validate_model_file,
)
from core.runtime_paths import RuntimePaths
from yolo_ui_widgets import SectionHeader, StatusBadge, ToolButton


class ModelDownloadThread(QThread):
    """在后台下载单个模型，避免阻塞 Qt 主线程。"""

    progress_changed = pyqtSignal(int, int)
    def __init__(self, model: DownloadableModel, models_dir: Path, parent=None):
        super().__init__(parent)
        self.model = model
        self.models_dir = Path(models_dir)
        self.cancel_event = threading.Event()
        self.result = None
        self.error_message = ""
        self.cancelled = False

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            self.result = download_model(
                self.model,
                self.models_dir,
                overwrite=True,
                progress=lambda current, total: self.progress_changed.emit(
                    current, total
                ),
                cancel_event=self.cancel_event,
            )
        except ModelDownloadCancelled as exc:
            self.error_message = str(exc)
            self.cancelled = True
        except ModelDownloadError as exc:
            self.error_message = str(exc)
        except Exception as exc:  # pragma: no cover - 防止后台线程静默退出
            self.error_message = f"下载模型失败: {exc}"


class ModelManagerDialog(QDialog):
    """选择、下载和管理官方目标检测模型。"""

    def __init__(self, models_dir=None, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelManagerDialog")
        self.setWindowTitle("模型管理")
        self.resize(820, 620)
        self.setMinimumSize(720, 520)
        self.models_dir = (
            Path(models_dir)
            if models_dir
            else RuntimePaths.from_environment().models_dir
        )
        self._thread: ModelDownloadThread | None = None
        self._close_when_finished = False
        self._build_ui()
        self._refresh_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(theme.SPACING["md"])

        header = QHBoxLayout()
        header.addWidget(
            SectionHeader(
                "模型管理",
                "只提供官方 YOLOv8 / YOLO11 目标检测权重",
            ),
            1,
        )
        self.status_badge = StatusBadge("就绪", "info")
        self.status_badge.setAlignment(Qt.AlignCenter)
        header.addWidget(self.status_badge, 0, Qt.AlignTop)
        layout.addLayout(header)

        self.models_table = QTableWidget(0, 5)
        self.models_table.setHorizontalHeaderLabels(
            ["系列", "尺寸", "文件名", "大小", "状态"]
        )
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.models_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models_table.setAlternatingRowColors(True)
        self.models_table.verticalHeader().setVisible(False)
        header_view = self.models_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.models_table.itemSelectionChanged.connect(self._refresh_selection)
        layout.addWidget(self.models_table, 1)

        self.detail_label = QLabel("请选择一个模型。")
        self.detail_label.setWordWrap(True)
        theme.set_text_role(self.detail_label, "hint")
        layout.addWidget(self.detail_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        self.download_button = ToolButton(
            "下载选中模型",
            "从官方 Ultralytics 地址下载选中的目标检测模型",
        )
        theme.set_button_role(self.download_button, "primary")
        self.download_button.clicked.connect(self._download_selected)
        action_row.addWidget(self.download_button)

        self.refresh_button = ToolButton("刷新状态", "重新扫描 models 目录")
        self.refresh_button.clicked.connect(self._refresh_table)
        action_row.addWidget(self.refresh_button)

        self.cancel_button = ToolButton("取消下载", "取消当前模型下载")
        theme.set_button_role(self.cancel_button, "warning")
        self.cancel_button.clicked.connect(self._cancel_download)
        self.cancel_button.setVisible(False)
        action_row.addWidget(self.cancel_button)

        self.open_dir_button = ToolButton("打开模型目录", "打开 models 目录")
        self.open_dir_button.clicked.connect(self._open_models_dir)
        action_row.addWidget(self.open_dir_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)
        layout.addWidget(self.button_box)

    def _selected_model(self):
        row = self.models_table.currentRow()
        if row < 0:
            return None
        return self.models_table.item(row, 0).data(Qt.UserRole)

    def _refresh_table(self):
        if self._thread is not None:
            return
        states = installed_models(self.models_dir)
        selected_name = None
        selected = self._selected_model()
        if selected is not None:
            selected_name = selected.filename
        self.models_table.setRowCount(0)
        for row, model in enumerate(MODEL_CATALOG):
            self.models_table.insertRow(row)
            state = states.get(model.filename, False)
            path = self.models_dir / model.filename
            if state:
                state_text = "已安装"
            elif path.exists():
                state_text = "文件异常"
            else:
                state_text = "未下载"
            values = [
                model.series,
                model.size,
                model.filename,
                self._format_size(model.expected_size),
                state_text,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, model)
                self.models_table.setItem(row, column, item)
        target_row = 0
        if selected_name:
            for row, model in enumerate(MODEL_CATALOG):
                if model.filename == selected_name:
                    target_row = row
                    break
        if self.models_table.rowCount():
            self.models_table.selectRow(target_row)
        self._refresh_selection()

    def _refresh_selection(self):
        model = self._selected_model()
        if model is None:
            self.download_button.setEnabled(False)
            self.detail_label.setText("请选择一个模型。")
            return
        path = self.models_dir / model.filename
        installed = validate_model_file(path, model)
        if installed:
            state = "已安装，可直接在训练、模型测试或预标注中选择。"
        elif path.exists():
            state = "文件存在但校验失败，需要重新下载。"
        else:
            state = "尚未下载。"
        self.detail_label.setText(
            f"{model.label} · {model.task} · {self._format_size(model.expected_size)}\n"
            f"保存位置：{path}\n状态：{state}"
        )
        self.download_button.setEnabled(self._thread is None)
        self.download_button.setText("重新下载" if path.exists() else "下载选中模型")

    def _download_selected(self):
        model = self._selected_model()
        if model is None or self._thread is not None:
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        target = self.models_dir / model.filename
        if target.exists():
            reply = QMessageBox.question(
                self,
                "确认覆盖",
                f"模型文件已经存在：\n{target}\n\n是否重新下载并覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._thread = ModelDownloadThread(model, self.models_dir, self)
        self._thread.progress_changed.connect(self._on_progress)
        self._thread.finished.connect(self._on_download_finished)
        self._set_running(True)
        self.status_badge.setText(f"正在下载 {model.filename}")
        self.status_badge.set_tone("info")
        self._thread.start()

    def _on_progress(self, current, total):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        self.detail_label.setText(
            f"正在下载 {current / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
        )

    def _on_download_finished(self):
        thread = self._thread
        if thread is None:
            return
        result = thread.result
        message = thread.error_message
        cancelled = thread.cancelled
        self._thread = None
        self._set_running(False)
        thread.deleteLater()

        if result is not None:
            self.status_badge.setText("下载完成")
            self.status_badge.set_tone("success")
            self._refresh_table()
            self.detail_label.setText(f"模型已保存：{result.path}")
        else:
            self.status_badge.setText("已取消" if cancelled else "下载失败")
            self.status_badge.set_tone("warning" if cancelled else "danger")
            self.detail_label.setText(message or "下载未完成。")
            if not cancelled and not self._close_when_finished:
                QMessageBox.warning(self, "模型下载失败", message or "下载未完成。")

        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _set_running(self, running):
        self.models_table.setEnabled(not running)
        self.download_button.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.open_dir_button.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        self.progress_bar.setVisible(running)
        self.button_box.button(QDialogButtonBox.Close).setEnabled(not running)

    def _cancel_download(self):
        if self._thread is None:
            return
        self._thread.cancel()
        self.cancel_button.setEnabled(False)
        self.status_badge.setText("正在取消")
        self.status_badge.set_tone("warning")

    @staticmethod
    def _format_size(size):
        return f"{size / 1024 / 1024:.1f} MB"

    def _open_models_dir(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.models_dir))
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", f"无法打开模型目录：{exc}")

    def closeEvent(self, event):
        if self._thread is not None:
            if self._close_when_finished:
                event.ignore()
                return
            reply = QMessageBox.question(
                self,
                "下载进行中",
                "模型正在下载，是否取消并关闭窗口？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._close_when_finished = True
            self._cancel_download()
            self.detail_label.setText("正在取消下载，完成后将自动关闭窗口。")
            event.ignore()
            return
        event.accept()

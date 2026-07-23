from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import yolo_ui_theme as theme
from core.dataset_quality import DatasetQualityReport, DatasetQualityScanner
from core.issues import IssueSeverity
from core.paths import ProjectPaths
from yolo_ui_widgets import SectionHeader, StatusBadge


PROJECT_DIR = Path(__file__).resolve().parent.parent


class SortTableWidgetItem(QTableWidgetItem):
    def __init__(self, text, sort_value=None):
        super().__init__(str(text))
        self.sort_value = sort_value if sort_value is not None else str(text)

    def __lt__(self, other):
        if isinstance(other, SortTableWidgetItem):
            return self.sort_value < other.sort_value
        return super().__lt__(other)


class DatasetQualityWorker(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress_changed = pyqtSignal(int, int)

    def __init__(self, scanner: DatasetQualityScanner, parent=None):
        super().__init__(parent)
        self.scanner = scanner

    def run(self):
        try:
            report = self.scanner.scan(
                progress=lambda current, total: self.progress_changed.emit(
                    current, total
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(report)


class DatasetQualityDialog(QDialog):
    locate_requested = pyqtSignal(str)
    optimize_split_requested = pyqtSignal(str)

    def __init__(self, dataset_dir, parent=None, auto_start=True):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.report: DatasetQualityReport | None = None
        self._worker: DatasetQualityWorker | None = None
        self.setWindowTitle("数据质量检查")
        self.resize(1040, 700)
        self.setMinimumSize(780, 560)
        self._build_ui()
        if auto_start:
            self.start_scan()

    @staticmethod
    def _configure_table(table: QTableWidget):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(theme.SPACING["md"])

        header = QHBoxLayout()
        header.addWidget(
            SectionHeader("数据质量检查", str(self.dataset_dir)),
            1,
        )
        self.summary_badge = StatusBadge("等待扫描", "info")
        self.summary_badge.setAlignment(Qt.AlignCenter)
        header.addWidget(self.summary_badge, 0, Qt.AlignTop)
        layout.addLayout(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("QualityTabs")
        layout.addWidget(self.tabs, 1)

        issues_tab = QWidget()
        issues_layout = QVBoxLayout(issues_tab)
        issues_layout.setContentsMargins(8, 10, 8, 8)
        self.issues_table = QTableWidget(0, 5)
        self.issues_table.setHorizontalHeaderLabels(
            ["级别", "代码", "问题", "建议", "路径"]
        )
        self._configure_table(self.issues_table)
        self.issues_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.issues_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.issues_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        issues_layout.addWidget(self.issues_table)
        self.tabs.addTab(issues_tab, "问题")

        classes_tab = QWidget()
        classes_layout = QVBoxLayout(classes_tab)
        classes_layout.setContentsMargins(8, 10, 8, 8)
        self.classes_table = QTableWidget(0, 10)
        self.classes_table.setHorizontalHeaderLabels(
            [
                "ID",
                "类别",
                "总图片",
                "总框",
                "train 图",
                "train 框",
                "val 图",
                "val 框",
                "test 图",
                "test 框",
            ]
        )
        self._configure_table(self.classes_table)
        self.classes_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        for column in (0, 2, 3, 4, 5, 6, 7, 8, 9):
            self.classes_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        classes_layout.addWidget(self.classes_table)
        self.tabs.addTab(classes_tab, "类别分布")

        duplicates_tab = QWidget()
        duplicates_layout = QVBoxLayout(duplicates_tab)
        duplicates_layout.setContentsMargins(8, 10, 8, 8)
        self.duplicates_table = QTableWidget(0, 4)
        self.duplicates_table.setHorizontalHeaderLabels(
            ["状态", "数量", "分组", "文件"]
        )
        self._configure_table(self.duplicates_table)
        self.duplicates_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.duplicates_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.duplicates_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        duplicates_layout.addWidget(self.duplicates_table)
        self.tabs.addTab(duplicates_tab, "重复图片")

        actions = QHBoxLayout()
        self.rescan_button = QPushButton("重新检查")
        theme.set_button_role(self.rescan_button, "secondary")
        self.optimize_button = QPushButton("优化划分")
        theme.set_button_role(self.optimize_button, "secondary")
        self.optimize_button.setEnabled(False)
        self.locate_button = QPushButton("在标注工具定位")
        theme.set_button_role(self.locate_button, "primary")
        self.locate_button.setEnabled(False)
        actions.addWidget(self.rescan_button)
        actions.addWidget(self.optimize_button)
        actions.addStretch(1)
        actions.addWidget(self.locate_button)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.button_box.button(QDialogButtonBox.Close).setText("关闭")
        theme.set_button_role(
            self.button_box.button(QDialogButtonBox.Close), "secondary"
        )
        actions.addWidget(self.button_box)
        layout.addLayout(actions)

        self.rescan_button.clicked.connect(self.start_scan)
        self.optimize_button.clicked.connect(
            lambda: self.optimize_split_requested.emit("repair")
        )
        self.locate_button.clicked.connect(self._emit_current_location)
        self.button_box.rejected.connect(self.reject)
        self.issues_table.currentCellChanged.connect(self._update_action_state)
        self.duplicates_table.currentCellChanged.connect(self._update_action_state)
        self.tabs.currentChanged.connect(self._update_action_state)
        self.issues_table.cellDoubleClicked.connect(
            lambda _row, _column: self._emit_current_location()
        )
        self.duplicates_table.cellDoubleClicked.connect(
            lambda _row, _column: self._emit_current_location()
        )

    def start_scan(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self.report = None
        self.summary_badge.setText("正在扫描")
        self.summary_badge.set_tone("info")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.rescan_button.setEnabled(False)
        self.locate_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self._clear_tables()

        paths = ProjectPaths.from_project_dir(
            PROJECT_DIR,
            dataset_dir=self.dataset_dir,
        )
        worker = DatasetQualityWorker(
            DatasetQualityScanner(paths),
            QApplication.instance(),
        )
        worker.progress_changed.connect(self._on_progress)
        worker.result_ready.connect(self._apply_report)
        worker.failed.connect(self._show_failure)
        worker.finished.connect(self._finish_scan)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_progress(self, current, total):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)

    def _finish_scan(self):
        self._worker = None
        self.progress_bar.setVisible(False)
        self.rescan_button.setEnabled(True)
        self._update_action_state()

    def _show_failure(self, message):
        self.summary_badge.setText(f"扫描失败：{message}")
        self.summary_badge.set_tone("danger")

    def _clear_tables(self):
        self.issues_table.setRowCount(0)
        self.classes_table.setRowCount(0)
        self.duplicates_table.setRowCount(0)

    @staticmethod
    def _set_item(
        table,
        row,
        column,
        text,
        image_path=None,
        sort_value=None,
    ):
        item = SortTableWidgetItem(text, sort_value)
        item.setToolTip(str(text))
        if image_path is not None:
            item.setData(Qt.UserRole, str(image_path))
        table.setItem(row, column, item)
        return item

    def _apply_report(self, report: DatasetQualityReport):
        self.report = report
        if report.error_count:
            tone = "danger"
        elif report.warning_count:
            tone = "warning"
        else:
            tone = "success"
        self.summary_badge.setText(
            f"图片 {report.image_count} | 错误 {report.error_count} | "
            f"警告 {report.warning_count} | 提示 {report.info_count}"
        )
        self.summary_badge.set_tone(tone)
        self._populate_issues(report)
        self._populate_classes(report)
        self._populate_duplicates(report)
        fixable_codes = {"class.missing_in_val"}
        self.optimize_button.setEnabled(
            not report.error_count
            and not report.duplicate_groups
            and any(
                finding.issue.code in fixable_codes
                for finding in report.findings
            )
        )
        self._update_action_state()

    def _populate_issues(self, report):
        labels = {
            IssueSeverity.ERROR: "错误",
            IssueSeverity.WARNING: "警告",
            IssueSeverity.INFO: "提示",
        }
        self.issues_table.setSortingEnabled(False)
        self.issues_table.setRowCount(len(report.findings))
        for row, finding in enumerate(report.findings):
            issue = finding.issue
            values = (
                labels[issue.severity],
                issue.code,
                issue.message,
                issue.suggested_action,
                str(issue.path) if issue.path is not None else "",
            )
            for column, value in enumerate(values):
                self._set_item(
                    self.issues_table,
                    row,
                    column,
                    value,
                    finding.image_path,
                    {
                        IssueSeverity.ERROR: 0,
                        IssueSeverity.WARNING: 1,
                        IssueSeverity.INFO: 2,
                    }[issue.severity]
                    if column == 0
                    else None,
                )
        self.issues_table.setSortingEnabled(True)
        self.issues_table.sortItems(0, Qt.AscendingOrder)

    def _populate_classes(self, report):
        self.classes_table.setSortingEnabled(False)
        self.classes_table.setRowCount(len(report.class_distributions))
        for row, item in enumerate(report.class_distributions):
            values = (
                item.class_id,
                item.name,
                item.image_count,
                item.box_count,
                item.train_images,
                item.train_boxes,
                item.val_images,
                item.val_boxes,
                item.test_images,
                item.test_boxes,
            )
            for column, value in enumerate(values):
                self._set_item(
                    self.classes_table,
                    row,
                    column,
                    value,
                    sort_value=value if column != 1 else None,
                )
        self.classes_table.setSortingEnabled(True)
        self.classes_table.sortItems(0, Qt.AscendingOrder)

    def _populate_duplicates(self, report):
        self.duplicates_table.setSortingEnabled(False)
        self.duplicates_table.setRowCount(len(report.duplicate_groups))
        for row, group in enumerate(report.duplicate_groups):
            values = (
                "跨分组" if group.cross_split else "同组重复",
                len(group.image_paths),
                ", ".join(group.splits),
                "\n".join(path.name for path in group.image_paths),
            )
            for column, value in enumerate(values):
                self._set_item(
                    self.duplicates_table,
                    row,
                    column,
                    value,
                    group.image_paths[0],
                    (0 if group.cross_split else 1) if column == 0 else None,
                )
        self.duplicates_table.resizeRowsToContents()
        self.duplicates_table.setSortingEnabled(True)
        self.duplicates_table.sortItems(0, Qt.AscendingOrder)

    def _current_image_path(self):
        if self.tabs.currentIndex() == 0:
            table = self.issues_table
        elif self.tabs.currentIndex() == 2:
            table = self.duplicates_table
        else:
            return None
        row = table.currentRow()
        if row < 0 or table.item(row, 0) is None:
            return None
        return table.item(row, 0).data(Qt.UserRole)

    def _update_action_state(self, *_args):
        self.locate_button.setEnabled(bool(self._current_image_path()))

    def _emit_current_location(self):
        image_path = self._current_image_path()
        if image_path:
            self.locate_requested.emit(str(image_path))

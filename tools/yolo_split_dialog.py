from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dataset_scanner import IMAGE_SUFFIXES
from core.dataset_split import (
    ClassCoveragePolicy,
    SplitMode,
    SplitPlan,
    SplitPlanner,
    SplitPolicy,
)
from core.dataset_split_executor import SplitExecutor
from core.paths import ProjectPaths
import yolo_ui_theme as theme
from yolo_ui_widgets import SectionHeader, StatusBadge


PROJECT_DIR = Path(__file__).resolve().parent.parent


class SplitPlanWorker(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, planner: SplitPlanner, policy: SplitPolicy, parent=None):
        super().__init__(parent)
        self.planner = planner
        self.policy = policy

    def run(self):
        try:
            plan = self.planner.plan(self.policy)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(plan)


class SplitExecuteWorker(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, executor: SplitExecutor, plan: SplitPlan, parent=None):
        super().__init__(parent)
        self.executor = executor
        self.plan = plan

    def run(self):
        try:
            result = self.executor.apply(self.plan)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(result)


class DatasetSplitDialog(QDialog):
    split_completed = pyqtSignal(object)

    def __init__(
        self,
        dataset_dir,
        parent=None,
        initial_mode=SplitMode.REPAIR,
        auto_start=False,
    ):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.plan: SplitPlan | None = None
        self._worker: QThread | None = None
        self.setWindowTitle("数据集智能划分")
        self.resize(1080, 720)
        self.setMinimumSize(820, 600)
        self._build_ui(initial_mode)
        self._load_observed_ratios()
        if auto_start:
            self.start_preview()

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

    def _build_ui(self, initial_mode: SplitMode) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(theme.SPACING["md"])

        header = QHBoxLayout()
        header.addWidget(
            SectionHeader("数据集智能划分", str(self.dataset_dir)),
            1,
        )
        self.status_badge = StatusBadge("等待设置", "info")
        header.addWidget(self.status_badge, 0, Qt.AlignTop)
        layout.addLayout(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_settings_page(initial_mode))
        self.pages.addWidget(self._build_preview_page())
        layout.addWidget(self.pages, 1)

        actions = QHBoxLayout()
        self.back_button = QPushButton("返回设置")
        theme.set_button_role(self.back_button, "secondary")
        self.back_button.setVisible(False)
        self.preview_button = QPushButton("生成预览")
        theme.set_button_role(self.preview_button, "primary")
        self.execute_button = QPushButton("执行方案")
        theme.set_button_role(self.execute_button, "warning")
        self.execute_button.setEnabled(False)
        self.execute_button.setVisible(False)
        actions.addWidget(self.back_button)
        actions.addStretch(1)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.execute_button)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.button_box.button(QDialogButtonBox.Close).setText("关闭")
        theme.set_button_role(
            self.button_box.button(QDialogButtonBox.Close), "secondary"
        )
        actions.addWidget(self.button_box)
        layout.addLayout(actions)

        self.preview_button.clicked.connect(self.start_preview)
        self.back_button.clicked.connect(self._show_settings)
        self.execute_button.clicked.connect(self._confirm_execute)
        self.button_box.rejected.connect(self.reject)

    def _build_settings_page(self, initial_mode: SplitMode) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(theme.SPACING["lg"])

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("模式"))
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.repair_mode_button = QPushButton("最小修复")
        self.full_mode_button = QPushButton("智能重划分")
        for button in (self.repair_mode_button, self.full_mode_button):
            button.setCheckable(True)
            button.setMinimumWidth(120)
            theme.set_button_role(button, "secondary")
            self.mode_group.addButton(button)
            mode_row.addWidget(button)
        if SplitMode(initial_mode) is SplitMode.FULL:
            self.full_mode_button.setChecked(True)
        else:
            self.repair_mode_button.setChecked(True)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(theme.SPACING["md"])
        grid.setVerticalSpacing(theme.SPACING["sm"])
        self.train_spin = self._ratio_spin(0.80)
        self.val_spin = self._ratio_spin(0.15)
        self.test_spin = self._ratio_spin(0.05)
        for row, (label, widget) in enumerate(
            (
                ("训练集比例", self.train_spin),
                ("验证集比例", self.val_spin),
                ("测试集比例", self.test_spin),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        grid.addWidget(QLabel("随机种子"), 3, 0)
        grid.addWidget(self.seed_spin, 3, 1)

        self.min_train_spin = QSpinBox()
        self.min_train_spin.setRange(1, 20)
        self.min_train_spin.setValue(5)
        grid.addWidget(QLabel("最低训练图片数"), 4, 0)
        grid.addWidget(self.min_train_spin, 4, 1)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)

        self.ratio_status = StatusBadge("比例合计 1.00", "success")
        layout.addWidget(self.ratio_status, 0, Qt.AlignLeft)
        layout.addStretch(1)
        for spin in (self.train_spin, self.val_spin, self.test_spin):
            spin.valueChanged.connect(self._update_ratio_status)
        return page

    @staticmethod
    def _ratio_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    def _build_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.preview_tabs = QTabWidget()
        layout.addWidget(self.preview_tabs)

        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        self.summary_table = QTableWidget(0, 4)
        self.summary_table.setHorizontalHeaderLabels(
            ["分组", "当前", "目标", "计划"]
        )
        self._configure_table(self.summary_table)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        summary_layout.addWidget(self.summary_table)
        self.risk_label = QLabel()
        self.risk_label.setWordWrap(True)
        theme.set_text_role(self.risk_label, "muted")
        summary_layout.addWidget(self.risk_label)
        self.preview_tabs.addTab(summary_page, "摘要")

        classes_page = QWidget()
        classes_layout = QVBoxLayout(classes_page)
        self.classes_table = QTableWidget(0, 10)
        self.classes_table.setHorizontalHeaderLabels(
            [
                "ID",
                "类别",
                "总图片",
                "当前 train",
                "当前 val",
                "当前 test",
                "计划 train",
                "计划 val",
                "计划 test",
                "状态",
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
        self.preview_tabs.addTab(classes_page, "类别覆盖")

        moves_page = QWidget()
        moves_layout = QVBoxLayout(moves_page)
        self.moves_table = QTableWidget(0, 4)
        self.moves_table.setHorizontalHeaderLabels(
            ["图片", "源分组", "目标分组", "类别 ID"]
        )
        self._configure_table(self.moves_table)
        self.moves_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for column in (1, 2, 3):
            self.moves_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        moves_layout.addWidget(self.moves_table)
        self.preview_tabs.addTab(moves_page, "移动清单")
        return page

    def _load_observed_ratios(self) -> None:
        counts = []
        for split in ("train", "val", "test"):
            directory = self.dataset_dir / "images" / split
            count = (
                sum(
                    1
                    for path in directory.rglob("*")
                    if path.is_file()
                    and path.suffix.casefold() in IMAGE_SUFFIXES
                )
                if directory.is_dir()
                else 0
            )
            counts.append(count)
        total = sum(counts)
        if total == 0 or sum(count > 0 for count in counts) < 2:
            return
        val_ratio = round(counts[1] / total, 2)
        test_ratio = round(counts[2] / total, 2)
        train_ratio = round(1.0 - val_ratio - test_ratio, 2)
        self.train_spin.setValue(train_ratio)
        self.val_spin.setValue(val_ratio)
        self.test_spin.setValue(test_ratio)

    def selected_mode(self) -> SplitMode:
        return (
            SplitMode.FULL
            if self.full_mode_button.isChecked()
            else SplitMode.REPAIR
        )

    def _update_ratio_status(self) -> None:
        total = (
            self.train_spin.value()
            + self.val_spin.value()
            + self.test_spin.value()
        )
        valid = abs(total - 1.0) <= 0.001 and self.train_spin.value() > 0
        self.ratio_status.setText(f"比例合计 {total:.2f}")
        self.ratio_status.set_tone("success" if valid else "danger")
        self.preview_button.setEnabled(valid and not self._is_busy())

    def _policy(self) -> SplitPolicy:
        return SplitPolicy(
            train_ratio=self.train_spin.value(),
            val_ratio=self.val_spin.value(),
            test_ratio=self.test_spin.value(),
            seed=self.seed_spin.value(),
            mode=self.selected_mode(),
            coverage=ClassCoveragePolicy(self.min_train_spin.value()),
        )

    def _paths(self) -> ProjectPaths:
        return ProjectPaths.from_project_dir(
            PROJECT_DIR,
            dataset_dir=self.dataset_dir,
        )

    def _is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self.progress_bar.setVisible(busy)
        self.preview_button.setEnabled(not busy)
        self.back_button.setEnabled(not busy)
        self.execute_button.setEnabled(
            not busy
            and self.plan is not None
            and self.plan.is_executable
            and bool(self.plan.moves)
        )
        if text:
            self.status_badge.setText(text)
            self.status_badge.set_tone("info")

    def start_preview(self) -> None:
        if self._is_busy():
            return
        try:
            policy = self._policy()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self.plan = None
        self.execute_button.setEnabled(False)
        self._set_busy(True, "正在生成预览")
        worker = SplitPlanWorker(
            SplitPlanner(self._paths()),
            policy,
            QApplication.instance(),
        )
        worker.result_ready.connect(self._apply_plan)
        worker.failed.connect(self._show_failure)
        worker.finished.connect(self._finish_worker)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    @staticmethod
    def _set_item(table, row, column, value) -> None:
        item = QTableWidgetItem(str(value))
        item.setToolTip(str(value))
        table.setItem(row, column, item)

    def _apply_plan(self, plan) -> None:
        self.plan = plan
        self.pages.setCurrentIndex(1)
        self.back_button.setVisible(True)
        self.preview_button.setText("重新生成")
        self.execute_button.setVisible(True)

        current = dict(plan.current_counts)
        target = dict(plan.target_counts)
        planned = dict(plan.planned_counts)
        self.summary_table.setRowCount(3)
        for row, split in enumerate(("train", "val", "test")):
            for column, value in enumerate(
                (split, current.get(split, 0), target.get(split, 0), planned.get(split, 0))
            ):
                self._set_item(self.summary_table, row, column, value)

        self.classes_table.setRowCount(len(plan.class_coverages))
        for row, coverage in enumerate(plan.class_coverages):
            values = (
                coverage.class_id,
                coverage.name,
                coverage.total_images,
                coverage.current_train,
                coverage.current_val,
                coverage.current_test,
                coverage.planned_train,
                coverage.planned_val,
                coverage.planned_test,
                "满足" if coverage.requirements_met else "受限",
            )
            for column, value in enumerate(values):
                self._set_item(self.classes_table, row, column, value)

        self.moves_table.setRowCount(len(plan.moves))
        for row, move in enumerate(plan.moves):
            values = (
                move.image_path.name,
                move.source_split,
                move.target_split,
                ", ".join(str(item) for item in move.class_ids) or "负样本",
            )
            for column, value in enumerate(values):
                self._set_item(self.moves_table, row, column, value)

        messages = [risk.message for risk in plan.risks]
        messages.extend(issue.message for issue in plan.blocking_issues)
        self.risk_label.setText("\n".join(messages) if messages else "无未解决风险")
        if plan.blocking_issues:
            self.status_badge.setText(
                f"阻断问题 {len(plan.blocking_issues)} | 移动 {len(plan.moves)}"
            )
            self.status_badge.set_tone("danger")
        elif plan.risks:
            self.status_badge.setText(
                f"移动 {len(plan.moves)} | 风险 {len(plan.risks)}"
            )
            self.status_badge.set_tone("warning")
        elif plan.moves:
            self.status_badge.setText(f"移动 {len(plan.moves)} | 可执行")
            self.status_badge.set_tone("success")
        else:
            self.status_badge.setText("无需调整")
            self.status_badge.set_tone("success")
        self.execute_button.setEnabled(
            plan.is_executable and bool(plan.moves)
        )

    def _show_settings(self) -> None:
        if self._is_busy():
            return
        self.pages.setCurrentIndex(0)
        self.back_button.setVisible(False)
        self.execute_button.setVisible(False)
        self.preview_button.setText("生成预览")
        self.status_badge.setText("等待设置")
        self.status_badge.set_tone("info")

    def _confirm_execute(self) -> None:
        if self.plan is None or not self.execute_button.isEnabled():
            return
        reply = QMessageBox.question(
            self,
            "确认执行划分方案",
            f"数据集：{self.dataset_dir}\n"
            f"移动图片和标签：{len(self.plan.moves)} 组\n"
            "执行前会创建完整分组备份。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._set_busy(True, "正在执行方案")
        worker = SplitExecuteWorker(
            SplitExecutor(self._paths()),
            self.plan,
            QApplication.instance(),
        )
        worker.result_ready.connect(self._show_success)
        worker.failed.connect(self._show_failure)
        worker.finished.connect(self._finish_worker)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _show_success(self, result) -> None:
        self.plan = None
        self.execute_button.setEnabled(False)
        self.status_badge.setText(f"已完成 | 移动 {result.moved_pairs}")
        self.status_badge.set_tone("success")
        self.risk_label.setText(f"备份：{result.backup_dir}")
        self.split_completed.emit(result)

    def _show_failure(self, message: str) -> None:
        self.plan = None
        self.execute_button.setEnabled(False)
        self.status_badge.setText("操作失败")
        self.status_badge.set_tone("danger")
        self.risk_label.setText(str(message))

    def _finish_worker(self) -> None:
        self._worker = None
        self.progress_bar.setVisible(False)
        self.preview_button.setEnabled(True)
        if self.plan is not None:
            self.execute_button.setEnabled(
                self.plan.is_executable and bool(self.plan.moves)
            )

    def closeEvent(self, event) -> None:
        if self._is_busy():
            QMessageBox.information(self, "任务进行中", "请等待当前任务完成。")
            event.ignore()
            return
        super().closeEvent(event)

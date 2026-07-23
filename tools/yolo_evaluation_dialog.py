"""模型评估与对比对话框。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QProcess, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

import yolo_ui_theme as theme
from yolo_dataset_tools import detect_devices
from core.model_evaluation import (
    EvaluationSession,
    load_evaluation_session,
)
from core.task_protocol import TaskEventType, decode_task_event
from core.runtime_paths import RuntimePaths
from core.worker_commands import build_worker_command
from yolo_ui_widgets import SectionHeader, StatusBadge
from yolo_runtime_dialog import ensure_ml_runtime


RUNTIME_PATHS = RuntimePaths.from_environment()
PROJECT_DIR = RUNTIME_PATHS.resource_dir
def _format_metric(value: float) -> str:
    return f"{float(value):.3f}"


class ModelEvaluationDialog(QDialog):
    """负责配置和展示评估任务，不自行执行模型推理。"""

    session_ready = pyqtSignal(object)

    def __init__(self, data_yaml=None, runs_dir=None, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelEvaluationDialog")
        self.setWindowTitle("模型评估")
        self.resize(1040, 720)
        self.setMinimumSize(820, 600)
        self._data_yaml = Path(data_yaml) if data_yaml else None
        self._runs_dir = Path(runs_dir) if runs_dir else RUNTIME_PATHS.runs_dir
        self._process: QProcess | None = None
        self._task_id = ""
        self._buffer = ""
        self._session: EvaluationSession | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(theme.SPACING["md"])

        header = QHBoxLayout()
        header.addWidget(
            SectionHeader("模型评估", "验证模型、对比基准并查看逐类别退化"),
            1,
        )
        self.summary_badge = StatusBadge("等待评估", "info")
        self.summary_badge.setAlignment(Qt.AlignCenter)
        header.addWidget(self.summary_badge, 0, Qt.AlignTop)
        layout.addLayout(header)

        setup = theme.set_panel(QWidget(), "EvaluationSetup")
        setup_layout = QGridLayout(setup)
        setup_layout.setContentsMargins(14, 12, 14, 12)
        setup_layout.setHorizontalSpacing(theme.SPACING["sm"])
        setup_layout.setVerticalSpacing(theme.SPACING["sm"])
        setup_layout.setColumnStretch(1, 1)
        setup_layout.setColumnStretch(4, 1)
        setup_layout.addWidget(QLabel("候选模型"), 0, 0)
        self.candidate_edit = QLineEdit()
        self.candidate_edit.setPlaceholderText("选择待评估的 .pt 模型")
        setup_layout.addWidget(self.candidate_edit, 0, 1, 1, 3)
        candidate_button = QPushButton("浏览")
        candidate_button.clicked.connect(self._browse_candidate)
        theme.set_button_role(candidate_button, "secondary")
        setup_layout.addWidget(candidate_button, 0, 4)

        setup_layout.addWidget(QLabel("基准模型"), 1, 0)
        self.baseline_edit = QLineEdit()
        self.baseline_edit.setPlaceholderText("可选，用于与候选模型比较")
        setup_layout.addWidget(self.baseline_edit, 1, 1, 1, 3)
        baseline_button = QPushButton("浏览")
        baseline_button.clicked.connect(self._browse_baseline)
        theme.set_button_role(baseline_button, "secondary")
        setup_layout.addWidget(baseline_button, 1, 4)

        setup_layout.addWidget(QLabel("数据配置"), 2, 0)
        self.data_edit = QLineEdit(str(self._data_yaml or ""))
        self.data_edit.setReadOnly(True)
        setup_layout.addWidget(self.data_edit, 2, 1, 1, 3)
        data_button = QPushButton("浏览")
        data_button.clicked.connect(self._browse_data)
        theme.set_button_role(data_button, "secondary")
        setup_layout.addWidget(data_button, 2, 4)

        setup_layout.addWidget(QLabel("评估分组"), 3, 0)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["test", "val"])
        setup_layout.addWidget(self.split_combo, 3, 1)
        setup_layout.addWidget(QLabel("图像尺寸"), 3, 2)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSuffix(" px")
        setup_layout.addWidget(self.imgsz_spin, 3, 3)
        setup_layout.addWidget(QLabel("设备"), 3, 4)
        self.device_combo = QComboBox()
        for label, value in detect_devices():
            self.device_combo.addItem(label, value)
        self.device_combo.setToolTip(
            "选择评估使用的计算设备；对比模型必须使用相同设备"
        )
        setup_layout.addWidget(self.device_combo, 3, 5)
        layout.addWidget(setup)

        metric_panel = theme.set_panel(QWidget(), "EvaluationMetrics")
        metric_layout = QGridLayout(metric_panel)
        metric_layout.setContentsMargins(14, 10, 14, 10)
        metric_layout.addWidget(QLabel("Precision"), 0, 0)
        metric_layout.addWidget(QLabel("Recall"), 0, 1)
        metric_layout.addWidget(QLabel("mAP50"), 0, 2)
        metric_layout.addWidget(QLabel("mAP50-95"), 0, 3)
        self.precision_value = QLabel("-")
        self.recall_value = QLabel("-")
        self.map50_value = QLabel("-")
        self.map50_95_value = QLabel("-")
        for index, label in enumerate(
            (
                self.precision_value,
                self.recall_value,
                self.map50_value,
                self.map50_95_value,
            )
        ):
            label.setObjectName("EvaluationMetricValue")
            metric_layout.addWidget(label, 1, index)
        self.sample_badge = StatusBadge("等待数据", "info")
        metric_layout.addWidget(self.sample_badge, 0, 4, 2, 1)
        layout.addWidget(metric_panel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.metrics_table = QTableWidget(0, 8)
        self.metrics_table.setHorizontalHeaderLabels(
            [
                "ID",
                "类别",
                "实例数",
                "Precision",
                "Recall",
                "mAP50",
                "mAP50-95",
                "相对变化",
            ]
        )
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        for column in (0, 2, 3, 4, 5, 6, 7):
            self.metrics_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        layout.addWidget(self.metrics_table, 1)

        footer = QHBoxLayout()
        self.detail_label = QLabel("尚未生成评估报告")
        self.detail_label.setWordWrap(True)
        theme.set_text_role(self.detail_label, "hint")
        footer.addWidget(self.detail_label, 1)
        self.open_output_button = QPushButton("打开产物目录")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output)
        theme.set_button_role(self.open_output_button, "secondary")
        footer.addWidget(self.open_output_button)
        self.start_button = QPushButton("开始评估")
        self.start_button.clicked.connect(self.start_evaluation)
        theme.set_button_role(self.start_button, "primary")
        footer.addWidget(self.start_button)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_evaluation)
        theme.set_button_role(self.cancel_button, "secondary")
        footer.addWidget(self.cancel_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        theme.set_button_role(close_button, "secondary")
        footer.addWidget(close_button)
        layout.addLayout(footer)

    def _browse_model(self, target: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型",
            str(self._runs_dir),
            "PyTorch 模型 (*.pt);;所有文件 (*)",
        )
        if path:
            target.setText(path)

    def _browse_candidate(self):
        self._browse_model(self.candidate_edit)

    def _browse_baseline(self):
        self._browse_model(self.baseline_edit)

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据集配置",
            str(self._data_yaml.parent if self._data_yaml else self.dataset_dir),
            "YAML 配置文件 (*.yaml *.yml);;所有文件 (*)",
        )
        if path:
            self._data_yaml = Path(path)
            self.data_edit.setText(path)

    @property
    def dataset_dir(self):
        return self._data_yaml.parent if self._data_yaml else RUNTIME_PATHS.dataset_dir

    def _set_running(self, running: bool):
        for widget in (
            self.candidate_edit,
            self.baseline_edit,
            self.data_edit,
            self.split_combo,
            self.imgsz_spin,
            self.device_combo,
        ):
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.progress_bar.setVisible(running)

    def _set_summary(self, text: str, tone: str):
        self.summary_badge.setText(text)
        self.summary_badge.set_tone(tone)

    def apply_session(self, session: EvaluationSession):
        self._session = session
        candidate = session.candidate
        metrics = candidate.metrics
        self.precision_value.setText(_format_metric(metrics.precision))
        self.recall_value.setText(_format_metric(metrics.recall))
        self.map50_value.setText(_format_metric(metrics.map50))
        self.map50_95_value.setText(_format_metric(metrics.map50_95))
        if session.comparison is None:
            self._set_summary("无基准", "info")
            self.sample_badge.setText(
                f"{metrics.image_count} 张图片 / {metrics.target_count} 个目标"
            )
            self.sample_badge.set_tone("info")
        elif session.comparison.low_sample:
            self._set_summary("仅供参考", "warning")
            self.sample_badge.setText(
                f"低样本：{metrics.image_count} 张图片 / {metrics.target_count} 个目标"
            )
            self.sample_badge.set_tone("warning")
        else:
            tone = {
                "推荐候选": "success",
                "保持基准": "danger",
                "基本持平": "warning",
            }.get(session.comparison.verdict, "info")
            self._set_summary(session.comparison.verdict, tone)
            self.sample_badge.setText(
                f"{metrics.image_count} 张图片 / {metrics.target_count} 个目标"
            )
            self.sample_badge.set_tone("success")

        deltas = {
            item.class_id: item
            for item in (session.comparison.class_deltas if session.comparison else ())
        }
        rows = list(metrics.classes)
        rows.sort(
            key=lambda item: (
                deltas.get(item.class_id).map50_95
                if item.class_id in deltas
                else 0.0,
                item.class_id,
            )
        )
        self.metrics_table.setRowCount(0)
        for row, item in enumerate(rows):
            self.metrics_table.insertRow(row)
            delta = deltas.get(item.class_id)
            class_label = item.name
            if item.instances == 0:
                class_label = f"{class_label}（无样本）"
            elif item.instances < 5:
                class_label = f"{class_label}（低样本）"
            if item.instances == 0:
                metric_values = ["-", "-", "-", "-", "-"]
            else:
                metric_values = [
                    _format_metric(item.precision),
                    _format_metric(item.recall),
                    _format_metric(item.map50),
                    _format_metric(item.map50_95),
                    _format_metric(delta.map50_95) if delta else "-",
                ]
            values = [item.class_id, class_label, item.instances, *metric_values]
            for column, value in enumerate(values):
                self.metrics_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.detail_label.setText(
            f"报告：{candidate.output_dir}\n"
            f"结论：{session.comparison.reason if session.comparison else '未选择基准模型'}"
        )
        self.open_output_button.setEnabled(candidate.output_dir.exists())
        self.session_ready.emit(session)

    def _validate_inputs(self):
        candidate = Path(self.candidate_edit.text().strip())
        data = Path(self.data_edit.text().strip())
        if not candidate.is_file():
            raise ValueError("候选模型不存在，请先选择 .pt 文件")
        if not data.is_file():
            raise ValueError("数据配置不存在，请先选择 data.yaml")
        return candidate, data

    def start_evaluation(self):
        if self._process is not None:
            return
        if not ensure_ml_runtime(self, "模型评估"):
            return
        try:
            candidate, data = self._validate_inputs()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self._task_id = os.urandom(8).hex()
        output = (
            self._runs_dir
            / "evaluations"
            / f"{self._task_id}-{candidate.stem}"
            / "evaluation.json"
        )
        args = [
            "--task-id", self._task_id,
            "--model", str(candidate),
            "--data", str(data),
            "--split", self.split_combo.currentText(),
            "--imgsz", str(self.imgsz_spin.value()),
            "--device", str(self.device_combo.currentData() or ""),
            "--output", str(output),
        ]
        baseline = self.baseline_edit.text().strip()
        if baseline:
            args.extend(["--baseline", baseline])
        worker_program, worker_args = build_worker_command(
            "evaluate",
            args,
            resource_dir=PROJECT_DIR,
        )
        if not Path(worker_program).is_file():
            QMessageBox.critical(self, "文件缺失", f"找不到评估 Worker：\n{worker_program}")
            return
        process = QProcess(self)
        process.setProgram(worker_program)
        process.setArguments(worker_args)
        process.setWorkingDirectory(str(PROJECT_DIR))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_process_output)
        process.finished.connect(self._on_process_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        self._buffer = ""
        self._set_running(True)
        self._set_summary("评估中", "info")
        process.start()
        if not process.waitForStarted(3000):
            self._finish_process(False, f"无法启动评估进程：{process.errorString()}")

    def _handle_line(self, line: str):
        event = decode_task_event(line)
        if event is None:
            return
        if event.type is TaskEventType.PROGRESS and event.progress is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(event.progress * 100))
        elif event.type is TaskEventType.FAILED:
            self._set_summary("评估失败", "danger")
            self.detail_label.setText(event.message)
        elif event.type is TaskEventType.CANCELLED:
            self._set_summary("已取消", "warning")
            self.detail_label.setText(event.message)
        elif event.type is TaskEventType.RESULT:
            output = event.payload.get("output")
            if output and Path(output).is_file():
                self.apply_session(load_evaluation_session(Path(output)))

    def _on_process_output(self):
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            try:
                self._handle_line(line.strip())
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self.detail_label.setText(f"评估事件解析失败：{exc}")

    def _on_process_error(self, error):
        if error == QProcess.FailedToStart:
            self._finish_process(False, f"无法启动评估进程：{self._process.errorString()}")

    def _on_process_finished(self, exit_code, exit_status):
        if self._buffer.strip():
            self._handle_line(self._buffer.strip())
            self._buffer = ""
        success = exit_code == 0 and exit_status != QProcess.CrashExit
        if not success and self.summary_badge.text() == "评估中":
            self._set_summary("评估失败", "danger")
        self._finish_process(success, "评估完成" if success else f"评估进程退出码 {exit_code}")

    def _finish_process(self, success: bool, message: str):
        process = self._process
        self._process = None
        self._set_running(False)
        if not success and self.summary_badge.text() == "评估中":
            self._set_summary("评估失败", "danger")
        if process is not None:
            process.deleteLater()
        if not self.detail_label.text() or self.detail_label.text() == "尚未生成评估报告":
            self.detail_label.setText(message)

    def cancel_evaluation(self):
        process = self._process
        if process is None:
            return
        process.terminate()
        if not process.waitForFinished(1500):
            process.kill()
            process.waitForFinished(1500)
        self._set_summary("已取消", "warning")
        self._finish_process(False, "评估任务已取消")

    def _open_output(self):
        if self._session is None:
            return
        path = self._session.candidate.output_dir
        if not path.exists():
            QMessageBox.information(self, "目录不存在", f"评估产物目录不存在：\n{path}")
            return
        try:
            os.startfile(str(path))
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", f"无法打开评估产物目录：\n{exc}")

    def closeEvent(self, event):
        if self._process is not None:
            self.cancel_evaluation()
        event.accept()

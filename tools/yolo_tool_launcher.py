"""YOLO 数据标注工具箱 —— 统一启动器 GUI

合并原 "打开YOLO标注工具.bat" 与 "准备YOLO数据集.bat" 为单一入口，
提供桌面菜单界面让用户选择执行：
  1) 数据集准备 —— 导入图片、可选格式转换、目录规范化
  2) 数据集划分 —— 可配置 train/val/test 比例
  3) 启动标注工具 —— 以子进程启动原 yolo_annotate_gui.py
  4) 格式校验   —— 扫描 YOLO 标签格式合规性
  5) 数据集统计 —— 输出类别分布与标注数量摘要
  6) 查看日志   —— 打开日志文件

依赖：PyQt5, opencv-python, numpy（缺任一则无法启动部分功能）
"""

from __future__ import annotations

import os
import sys
import subprocess
import logging
import socket
import urllib.request
import webbrowser
from multiprocessing import freeze_support
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QProcess
from PyQt5.QtGui import QFont, QColor, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QCheckBox,
    QGroupBox, QGridLayout, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QSplitter, QProgressBar, QSizePolicy,
    QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QScrollArea, QStyle, QTabWidget,
)

# 同目录下导入后端模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolo_dataset_tools as tools  # noqa: E402
import yolo_ui_theme as theme  # noqa: E402
from yolo_ui_widgets import PathBar, SectionHeader, StatusBadge, ToolButton  # noqa: E402
from core.task_manager import TaskManager  # noqa: E402
from core.task_protocol import TaskEvent, TaskEventType, decode_task_event  # noqa: E402
from core.diagnostics import (  # noqa: E402
    build_diagnostic_archive,
    install_global_exception_hook,
)
from core.runtime_paths import RuntimePaths  # noqa: E402
from core.single_instance import SingleInstanceGuard, instance_name_for_state  # noqa: E402
from core.version import (  # noqa: E402
    DISPLAY_VERSION,
    LICENSE_ID,
    PRODUCT_NAME,
    PUBLISHER_NAME,
    get_source_url,
)
from core.worker_commands import build_worker_command  # noqa: E402
from yolo_runtime_dialog import (  # noqa: E402
    RuntimeManagerDialog,
    ensure_ml_runtime,
    runtime_status_summary,
)

PROJECT_DIR = tools.get_project_dir()
DATASET_DIR = tools.get_dataset_dir()
LOG_FILE = tools.get_logs_dir() / "yolo_tool.log"
logger = logging.getLogger("yolo_tool")


def _workspace_is_writable(path: Path) -> bool:
    if not path.is_dir():
        return False
    probe = path / f".workbuddy-write-test-{os.getpid()}"
    try:
        probe.write_text("ok", encoding="ascii")
        return True
    except OSError:
        return False
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass


def prepare_initial_workspace(config: dict, parent=None) -> dict | None:
    """确保首次启动选择了可写工作区；取消时不修改配置。"""
    prepared = dict(config or {})
    configured = str(prepared.get("workspace_dir") or "").strip()
    workspace = Path(configured) if configured else None
    if workspace is not None and _workspace_is_writable(workspace):
        return prepared

    runtime = tools.get_runtime_paths()
    start_dir = runtime.workspace_dir.parent
    if workspace is not None and workspace.parent.exists():
        start_dir = workspace.parent
    selected = QFileDialog.getExistingDirectory(
        parent,
        "选择 YOLO 工作区",
        str(start_dir),
        QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
    )
    if not selected:
        return None

    workspace = Path(selected).resolve()
    if not _workspace_is_writable(workspace):
        QMessageBox.warning(parent, "工作区不可写", f"无法写入所选目录：\n{workspace}")
        return None

    prepared.update(
        {
            "workspace_dir": str(workspace),
            "dataset_dir": str((workspace / "dataset").resolve()),
            "runs_dir": str((workspace / "runs").resolve()),
            "models_dir": str((workspace / "models").resolve()),
        }
    )
    tools.save_config(prepared)
    return prepared


def _activate_existing_window(window):
    if window.isMinimized():
        window.showNormal()
    else:
        window.show()
    window.raise_()
    window.activateWindow()


# ---------------------------------------------------------------------------
# 工作线程：在后台运行后端函数，通过信号实时推送日志
# ---------------------------------------------------------------------------
class Worker(QThread):
    log_signal = pyqtSignal(str, str)   # (level, message)
    done_signal = pyqtSignal(bool, str) # (success, summary)
    event_signal = pyqtSignal(object)

    def __init__(self, func, args=(), kwargs=None, task_id=None, parent=None):
        super().__init__(parent)
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.task_id = task_id or str(uuid4())

    def run(self):
        try:
            self.event_signal.emit(TaskEvent(self.task_id, TaskEventType.STARTED, "background"))

            def emit(level, msg):
                self.log_signal.emit(level, msg)
                self.event_signal.emit(
                    TaskEvent(
                        self.task_id,
                        TaskEventType.LOG,
                        "background",
                        message=str(msg),
                        payload={"level": str(level or "info")},
                    )
                )

            result = self.func(*self.args, emit=emit, **self.kwargs)
            summary = "完成" if result else "失败（请查看日志）"
            self.event_signal.emit(
                TaskEvent(
                    self.task_id,
                    TaskEventType.RESULT if result else TaskEventType.FAILED,
                    "background",
                    message=summary,
                    payload={"success": bool(result)},
                )
            )
            self.done_signal.emit(bool(result), summary)
        except BaseException as e:
            self.log_signal.emit("error", f"执行异常: {e}")
            logger.exception("worker error")
            self.event_signal.emit(
                TaskEvent(
                    self.task_id,
                    TaskEventType.CANCELLED if isinstance(e, KeyboardInterrupt) else TaskEventType.FAILED,
                    "background",
                    message=f"异常: {e}",
                )
            )
            self.done_signal.emit(False, f"异常: {e}")


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class YoloToolLauncher(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.setWindowTitle("YOLO 数据标注工具箱")
        self.setGeometry(60, 40, 1200, 820)
        self.setMinimumSize(960, 680)

        self.worker = None       # 当前运行的工作线程
        self.train_process = None  # 当前训练子进程
        self.task_manager = TaskManager()
        self._worker_task_id = None
        self._worker_ui_finished = False
        self._train_task_id = None
        self._train_cancel_requested = False
        self._train_output_buffer = ""
        self._train_result = None
        self._train_finished = False
        self.tensorboard_process = None
        self.tensorboard_url = ""
        self._tensorboard_task_id = None
        self._tensorboard_output_buffer = ""
        self._tensorboard_log_path = None
        self.child_windows = []   # 标注/检测等子窗口引用，防止被提前回收

        # 初始化日志
        tools.setup_logging(log_file=str(LOG_FILE), console=False)

        # 依赖检测
        self.missing_deps, self.dep_info = tools.check_dependencies(
            include_ml=not bool(getattr(sys, "frozen", False))
        )

        # 读取配置（数据集位置 / 结果位置）
        self.config = dict(config) if config is not None else tools.load_config()
        self.dataset_dir = Path(self.config["dataset_dir"])
        self.runs_dir = Path(self.config["runs_dir"])
        self.workspace_dir = Path(
            self.config.get("workspace_dir") or self.dataset_dir.parent
        )
        self.models_dir = Path(
            self.config.get("models_dir") or self.workspace_dir / "models"
        )
        self.runtime_paths = RuntimePaths.from_environment(
            workspace_dir=self.workspace_dir
        )
        self.default_source = self.config.get("source_dir", "")

        self._build_ui()
        self._refresh_dep_status()

        # 启动时自动输出一条统计概览
        self._append_log("info", f"项目目录: {PROJECT_DIR}")
        self._append_log("info", f"数据集位置: {self.dataset_dir}")
        self._append_log("info", f"结果位置: {self.runs_dir}")
        self._append_log("info", f"Python: {sys.executable}")
        if self.missing_deps:
            self._append_log("error", f"缺失依赖: {self.missing_deps}，部分功能不可用。")

    # ---------------------------------------------------------------
    # UI 构建
    # ---------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(
            theme.SPACING["lg"],
            theme.SPACING["md"],
            theme.SPACING["lg"],
            theme.SPACING["lg"],
        )
        main_layout.setSpacing(theme.SPACING["md"])

        # 顶部项目栏
        self.header_panel = theme.set_panel(QWidget(), "AppHeader")
        header_layout = QHBoxLayout(self.header_panel)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title_label = QLabel("YOLO 数据标注工具箱")
        title_label.setObjectName("PageTitle")
        title_layout.addWidget(title_label)
        subtitle = QLabel("项目工作台")
        subtitle.setObjectName("PageSubtitle")
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)

        self.dep_label = StatusBadge("正在检查环境", "info")
        self.dep_label.setWordWrap(True)
        self.dep_label.setMaximumWidth(620)
        header_layout.addWidget(self.dep_label, 1)
        main_layout.addWidget(self.header_panel)

        # 项目路径栏
        path_panel = theme.set_panel(QWidget(), "Panel")
        path_layout = QHBoxLayout(path_panel)
        path_layout.setContentsMargins(14, 10, 14, 10)
        path_layout.setSpacing(16)

        self.dataset_path_bar = PathBar("数据集", str(self.dataset_dir))
        self.dataset_path_bar.browse_clicked.connect(self._browse_dataset_dir)
        self.dataset_edit = self.dataset_path_bar.line_edit
        path_layout.addWidget(self.dataset_path_bar, 1)

        self.runs_path_bar = PathBar("训练结果", str(self.runs_dir))
        self.runs_path_bar.browse_clicked.connect(self._browse_runs_dir)
        self.runs_edit = self.runs_path_bar.line_edit
        path_layout.addWidget(self.runs_path_bar, 1)
        main_layout.addWidget(path_panel)

        # 主区域：工作流导航与运行控制台
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.main_splitter, 1)

        self.workflow_panel = theme.set_panel(QWidget(), "WorkflowNav")
        self.workflow_panel.setMinimumWidth(240)
        self.workflow_panel.setMaximumWidth(330)
        workflow_layout = QVBoxLayout(self.workflow_panel)
        workflow_layout.setContentsMargins(12, 12, 12, 12)
        workflow_layout.setSpacing(10)
        workflow_layout.addWidget(SectionHeader("工作流程"))

        workflow_scroll = QScrollArea()
        workflow_scroll.setWidgetResizable(True)
        workflow_scroll.setFrameShape(QScrollArea.NoFrame)
        workflow_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        workflow_scroll.setObjectName("WorkflowScroll")
        workflow_content = QWidget()
        workflow_content.setObjectName("WorkflowContent")
        button_layout = QVBoxLayout(workflow_content)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)

        action_groups = [
            ("数据流程", [
                ("01", "数据集准备", "导入图片到数据集（支持格式转换与目录规范化）", self._on_prepare, "primary"),
                ("02", "启动标注工具", "打开 YOLO 标注界面进行框选标注", self._on_launch_annotator, "accent"),
                ("03", "数据集划分", "按可配置比例划分 train / val / test", self._on_split, "secondary"),
                ("04", "格式校验", "校验 YOLO 标签文件合规性（类别ID、归一化坐标）", self._on_validate, "secondary"),
                ("05", "数据质量检查", "检查内容重复、跨分组泄漏和类别分布", self._on_quality_check, "secondary"),
            ]),
            ("训练评估", [
                ("06", "模型训练", "配置参数并启动 YOLO 训练（后台运行，实时日志）", self._on_train, "primary"),
                ("07", "模型测试", "用训练好的模型检测图片，可视化结果", self._on_launch_detect, "accent"),
                ("08", "模型评估", "在验证集或测试集评估模型并对比基准", self._on_model_evaluation, "accent"),
                ("09", "TensorBoard", "打开 TensorBoard 查看训练曲线和指标", self._on_tensorboard, "secondary"),
            ]),
            ("维护工具", [
                ("10", "数据集统计", "输出类别分布、标注数量等摘要信息", self._on_stats, "secondary"),
                ("11", "环境体检", "检查依赖、路径权限、数据集概要和 CUDA 状态", self._on_health_check, "secondary"),
                ("13", "备份恢复", "查看自动备份，并恢复 labels、classes.txt 和 data.yaml", self._on_backup_restore, "warning"),
                ("12", "发布前自检", "检查依赖清单、发布包隔离规则和运行环境", self._on_preflight_check, "secondary"),
                ("14", "查看日志", "打开日志文件查看详细记录", self._on_view_log, "secondary"),
                ("15", "模型管理", "下载和管理官方 YOLO 目标检测模型", self._on_model_manager, "secondary"),
                ("18", "运行环境管理", "安装、切换和修复 CPU 或 GPU 模型环境", self._on_runtime_manager, "secondary"),
                ("16", "导出诊断", "导出版本、路径状态和日志摘要", self._on_export_diagnostics, "secondary"),
                ("17", "关于", "查看版本、许可证、第三方声明和源码", self._on_about, "secondary"),
            ]),
        ]

        icon_map = {
            "数据集准备": QStyle.SP_DialogOpenButton,
            "数据集划分": QStyle.SP_FileDialogListView,
            "启动标注工具": QStyle.SP_FileDialogDetailedView,
            "格式校验": QStyle.SP_DialogApplyButton,
            "数据质量检查": QStyle.SP_FileDialogInfoView,
            "数据集统计": QStyle.SP_FileDialogInfoView,
            "环境体检": QStyle.SP_MessageBoxInformation,
            "发布前自检": QStyle.SP_DialogApplyButton,
            "备份恢复": QStyle.SP_DriveHDIcon,
            "查看日志": QStyle.SP_FileIcon,
            "模型训练": QStyle.SP_MediaPlay,
            "模型测试": QStyle.SP_ComputerIcon,
            "TensorBoard": QStyle.SP_DesktopIcon,
            "模型评估": QStyle.SP_DialogApplyButton,
            "模型管理": QStyle.SP_DriveNetIcon,
            "运行环境管理": QStyle.SP_ComputerIcon,
            "导出诊断": QStyle.SP_DialogSaveButton,
            "关于": QStyle.SP_MessageBoxInformation,
        }

        self.func_buttons = {}
        for title, actions in action_groups:
            group_title = QLabel(title)
            group_title.setProperty("textRole", "hint")
            group_title.setContentsMargins(4, 7, 4, 2)
            button_layout.addWidget(group_title)
            for num, name, desc, callback, role in actions:
                btn = QPushButton(f"{num}  {name}")
                btn.setToolTip(desc)
                btn.setIcon(self.style().standardIcon(icon_map[name]))
                btn.setIconSize(QSize(18, 18))
                theme.set_button_role(btn, "navigation")
                btn.setProperty("actionRole", role)
                btn.clicked.connect(callback)
                button_layout.addWidget(btn)
                self.func_buttons[name] = btn
        button_layout.addStretch(1)
        workflow_scroll.setWidget(workflow_content)
        workflow_layout.addWidget(workflow_scroll, 1)
        self.main_splitter.addWidget(self.workflow_panel)

        self.console_panel = theme.set_panel(QWidget(), "ConsolePanel")
        console_layout = QVBoxLayout(self.console_panel)
        console_layout.setContentsMargins(14, 12, 14, 14)
        console_layout.setSpacing(10)

        console_header = QHBoxLayout()
        console_header.addWidget(SectionHeader("运行控制台", "当前任务、进度与详细日志"), 1)
        self.stop_train_button = QPushButton("停止训练")
        self.stop_train_button.setObjectName("StopTrainingButton")
        self.stop_train_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaStop)
        )
        self.stop_train_button.setToolTip("停止当前模型训练")
        self.stop_train_button.setEnabled(False)
        theme.set_button_role(self.stop_train_button, "danger")
        self.stop_train_button.clicked.connect(self._on_stop_train_clicked)
        console_header.addWidget(self.stop_train_button, 0, Qt.AlignTop)
        clear_log_button = ToolButton(
            "",
            "清空当前日志",
            self.style().standardIcon(QStyle.SP_DialogResetButton),
        )
        clear_log_button.clicked.connect(self._clear_log)
        console_header.addWidget(clear_log_button, 0, Qt.AlignTop)
        console_layout.addLayout(console_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_text.setPlaceholderText("任务日志将在这里显示")
        console_layout.addWidget(self.log_text, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        console_layout.addWidget(self.progress)

        self.main_splitter.addWidget(self.console_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([280, 900])

    # ---------------------------------------------------------------
    # 路径选择
    # ---------------------------------------------------------------
    def _browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择数据集存放位置", str(self.dataset_dir),
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if dir_path:
            ok, msg, suggested = tools.diagnose_dataset_dir_path(dir_path)
            if not ok:
                QMessageBox.warning(
                    self,
                    "已修正数据集目录",
                    f"{msg}\n\n将改用:\n{suggested}",
                )
                dir_path = str(suggested)
            self.dataset_dir = Path(dir_path)
            self.dataset_edit.setText(str(self.dataset_dir))
            self.config["dataset_dir"] = str(self.dataset_dir)
            tools.save_config(self.config)
            # 自动在新位置生成/更新 data.yaml
            tools.regenerate_data_yaml(self.dataset_dir, emit=lambda lvl, msg: self._append_log(lvl, msg))
            self._append_log("info", f"数据集位置已更改为: {self.dataset_dir}")

    def _browse_runs_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择训练结果存放位置", str(self.runs_dir),
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if dir_path:
            self.runs_dir = Path(dir_path)
            self.runs_edit.setText(str(self.runs_dir))
            self.config["runs_dir"] = str(self.runs_dir)
            tools.save_config(self.config)
            self._append_log("info", f"结果位置已更改为: {self.runs_dir}")

    # ---------------------------------------------------------------
    # 依赖状态刷新
    # ---------------------------------------------------------------
    def _refresh_dep_status(self):
        parts = []
        for name, status in self.dep_info.items():
            if status == "已安装":
                parts.append(f"{name}: 正常")
            else:
                parts.append(f"{name}: 缺失")
        text = "依赖状态: " + " | ".join(parts)
        if self.missing_deps:
            text += f"  —— 缺失 {self.missing_deps} 的功能将不可用"
            theme.set_status_tone(self.dep_label, "danger")
        else:
            theme.set_status_tone(self.dep_label, "success")
        runtime_text, runtime_tone = runtime_status_summary()
        text += f"  |  {runtime_text}"
        if not self.missing_deps and runtime_tone != "success":
            theme.set_status_tone(self.dep_label, runtime_tone)
        self.dep_label.setText(text)

    # ---------------------------------------------------------------
    # 日志输出
    # ---------------------------------------------------------------
    def _append_log(self, level, msg):
        color_map = {
            "info": "#333333",
            "debug": "#888888",
            "warning": "#cc8800",
            "error": "#cc0000",
        }
        color = color_map.get(level, "#333333")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f'<span style="color:{color}">[{ts}] {msg}</span>')
        # 自动滚动到底
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def _clear_log(self):
        self.log_text.clear()

    # ---------------------------------------------------------------
    # 按钮启/禁用控制
    # ---------------------------------------------------------------
    def _set_buttons_enabled(self, enabled):
        for btn in self.func_buttons.values():
            btn.setEnabled(enabled)

    def _set_train_control_state(self, running, stopping=False):
        """同步训练停止按钮的空闲、运行和停止中状态。"""
        self.stop_train_button.setText("正在停止" if stopping else "停止训练")
        self.stop_train_button.setEnabled(bool(running and not stopping))

    def _on_stop_train_clicked(self):
        if self.train_process is None:
            self._set_train_control_state(False)
            return
        reply = QMessageBox.question(
            self,
            "停止训练",
            "确定停止当前训练吗？\n\n"
            "最近一个已完成 epoch 的权重会保留，当前未完成 epoch 可能不会保存。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._append_log("warning", "正在停止训练，请稍候...")
        self._stop_train_process()

    def _start_worker(self, func, args=(), kwargs=None):
        """启动工作线程，期间禁用所有功能按钮。"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "有任务正在执行中，请等待完成后再操作。")
            return
        if self.train_process is not None:
            QMessageBox.warning(self, "提示", "训练正在执行中，请等待完成后再操作。")
            return
        self._set_buttons_enabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # 无限滚动
        self._worker_task_id = str(uuid4())
        self.task_manager.create_task("background", task_id=self._worker_task_id)
        self.worker = Worker(func, args, kwargs, task_id=self._worker_task_id, parent=self)
        self.worker.log_signal.connect(lambda lvl, msg: self._append_log(lvl, msg))
        self.worker.event_signal.connect(self._handle_background_event)
        self.worker.done_signal.connect(self._on_worker_done)
        self.worker.finished.connect(self._on_worker_thread_finished)
        self._worker_ui_finished = False
        self.worker.start()

    def _handle_background_event(self, event):
        if not isinstance(event, TaskEvent):
            return
        try:
            self.task_manager.accept(event)
        except ValueError:
            return

    def _handle_task_event_line(self, line):
        try:
            event = decode_task_event(line)
        except ValueError as exc:
            self._append_log("warning", f"任务事件解析失败: {exc}")
            return None
        if event is None:
            return None
        try:
            self.task_manager.accept(event)
        except ValueError:
            return event
        if event.type is TaskEventType.LOG:
            self._append_log(event.payload.get("level", "info"), event.message)
        elif event.type is TaskEventType.STARTED:
            self._append_log("info", event.message or "任务已启动")
        elif event.type is TaskEventType.PROGRESS:
            if event.progress is not None:
                self.progress.setRange(0, 100)
                self.progress.setValue(int(event.progress * 100))
            if event.message:
                self._append_log("info", event.message)
        elif event.type is TaskEventType.FAILED:
            self._append_log("error", event.message)
        elif event.type is TaskEventType.CANCELLED:
            self._append_log("warning", event.message or "任务已取消")
        return event

    def _on_worker_done(self, success, summary):
        task_id = self._worker_task_id
        task = self.task_manager.get(task_id) if task_id else None
        if task is not None and task.final_event is None:
            self.task_manager.accept(
                TaskEvent(
                    task_id,
                    TaskEventType.RESULT if success else TaskEventType.FAILED,
                    "background",
                    message=summary,
                    payload={"success": bool(success)},
                )
            )
        if self._worker_ui_finished:
            return
        self._worker_ui_finished = True
        self._set_buttons_enabled(True)
        self.progress.setVisible(False)
        self._append_log("info", f"任务结束: {summary}")

    def _on_worker_thread_finished(self):
        task_id = self._worker_task_id
        task = self.task_manager.get(task_id) if task_id else None
        if task is not None and task.final_event is None:
            final_event = self.task_manager.process_exited(task_id, 1, True)
            if not self._worker_ui_finished:
                self._worker_ui_finished = True
                self._set_buttons_enabled(True)
                self.progress.setVisible(False)
                self._append_log("error", f"后台任务异常退出: {final_event.message}")
        worker = self.worker
        self.worker = None
        self._worker_task_id = None
        if worker is not None:
            worker.deleteLater()

    def _start_train_process(self, *, model_path, data_yaml, epochs, batch,
                             imgsz, device, project, name, resume):
        """用独立 Python 子进程运行训练，避免 torch 崩溃带关 GUI。"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "有任务正在执行中，请等待完成后再训练。")
            return
        if self.train_process is not None:
            QMessageBox.warning(self, "提示", "训练正在执行中，请等待完成后再操作。")
            return
        task_id = str(uuid4())
        args = [
            "--model", str(model_path),
            "--data", str(data_yaml),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--project", str(project),
            "--name", str(name),
            "--task-id", task_id,
        ]
        if device:
            args.extend(["--device", str(device)])
        if resume:
            args.append("--resume")
        worker_program, worker_args = build_worker_command(
            "train",
            args,
            resource_dir=PROJECT_DIR,
        )
        if not Path(worker_program).is_file():
            QMessageBox.critical(self, "文件缺失", f"找不到训练 Worker:\n{worker_program}")
            return

        self._set_buttons_enabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._train_output_buffer = ""
        self._train_result = None
        self._train_finished = False
        self._train_cancel_requested = False
        self._train_task_id = task_id

        process = QProcess(self)
        process.setProgram(worker_program)
        process.setArguments(worker_args)
        process.setWorkingDirectory(str(PROJECT_DIR))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_train_process_output)
        process.finished.connect(self._on_train_process_finished)
        process.errorOccurred.connect(self._on_train_process_error)
        self.train_process = process
        self._set_train_control_state(True)
        self.task_manager.create_task(
            "training",
            task_id=task_id,
            cancel_callback=lambda: self._terminate_train_process(process),
        )

        self._append_log("info", "训练已切换为独立进程运行，主界面会保持可用。")
        logger.info("启动训练子进程: %s %s", worker_program, " ".join(worker_args))
        process.start()
        if not process.waitForStarted(3000) and self.train_process is process:
            self._ensure_train_final_event(
                TaskEventType.FAILED,
                f"无法启动训练进程: {process.errorString()}",
            )
            self._finish_train_process(False, f"无法启动训练进程: {process.errorString()}")

    def _on_train_process_output(self):
        process = self.train_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._train_output_buffer += text
        while "\n" in self._train_output_buffer:
            line, self._train_output_buffer = self._train_output_buffer.split("\n", 1)
            self._handle_train_process_line(line.rstrip("\r"))

    def _handle_train_process_line(self, line):
        line = line.strip()
        if not line:
            return
        event = self._handle_task_event_line(line)
        if event is not None:
            if event.type is TaskEventType.RESULT:
                self._train_result = {"success": True, "summary": event.message}
            elif event.type is TaskEventType.FAILED:
                self._train_result = {"success": False, "summary": event.message}
            elif event.type is TaskEventType.CANCELLED:
                self._train_result = {"cancelled": True, "summary": event.message}
            return
        log_prefix = "__YOLO_TOOL_LOG__"
        result_prefix = "__YOLO_TOOL_RESULT__"
        if line.startswith(log_prefix) or line.startswith(result_prefix):
            import json
            prefix = log_prefix if line.startswith(log_prefix) else result_prefix
            try:
                payload = json.loads(line[len(prefix):])
            except Exception:
                self._append_log("warning", line)
                return
            if prefix == result_prefix:
                self._train_result = payload
            else:
                self._append_log(payload.get("level", "info"), payload.get("message", ""))
            return
        self._append_log("info", line)

    def _on_train_process_error(self, error):
        process = self.train_process
        if process is None:
            return
        logger.error("训练进程错误: %s, %s", error, process.errorString())
        if error == QProcess.FailedToStart:
            message = f"无法启动训练进程: {process.errorString()}"
            self._ensure_train_final_event(TaskEventType.FAILED, message)
            self._finish_train_process(False, message)

    def _on_train_process_finished(self, exit_code, exit_status):
        if self._train_finished:
            return
        if self._train_output_buffer.strip():
            self._handle_train_process_line(self._train_output_buffer.strip())
            self._train_output_buffer = ""

        task = self.task_manager.get(self._train_task_id) if self._train_task_id else None
        if task is not None:
            final_event = task.final_event
            if final_event is None:
                final_event = self.task_manager.process_exited(
                    self._train_task_id,
                    exit_code,
                    exit_status == QProcess.CrashExit,
                )
            success = final_event.type is TaskEventType.RESULT
            summary = final_event.message or ("完成" if success else "失败")
        elif self._train_result:
            success = bool(self._train_result.get("success"))
            summary = self._train_result.get("summary") or ("完成" if success else "失败")
        else:
            success = exit_code == 0 and exit_status != QProcess.CrashExit
            if exit_status == QProcess.CrashExit:
                summary = "训练进程异常退出，主界面已保留，请查看日志"
            else:
                summary = f"训练进程退出码 {exit_code}"
        self._finish_train_process(success, summary)

    def _ensure_train_final_event(self, event_type, message, payload=None):
        task_id = self._train_task_id
        if not task_id:
            return None
        task = self.task_manager.get(task_id)
        if task is None or task.final_event is not None:
            return task.final_event if task is not None else None
        event = TaskEvent(
            task_id,
            event_type,
            "training",
            message=message,
            payload=payload or {},
        )
        self.task_manager.accept(event)
        return event

    def _finish_train_process(self, success, summary):
        if self._train_finished:
            return
        cancelled = self._train_cancel_requested
        self._train_finished = True
        process = self.train_process
        self.train_process = None
        self._train_task_id = None
        self._set_buttons_enabled(True)
        self._set_train_control_state(False)
        self.progress.setVisible(False)
        level = "warning" if cancelled else ("info" if success else "error")
        self._append_log(level, f"训练任务结束: {summary}")
        logger.info("训练任务结束: success=%s summary=%s", success, summary)
        self._train_cancel_requested = False
        if process is not None:
            process.deleteLater()

    def _stop_train_process(self):
        process = self.train_process
        if process is None:
            self._set_train_control_state(False)
            return
        task_id = self._train_task_id
        self._train_cancel_requested = True
        self._set_train_control_state(True, stopping=True)
        cancel_dispatched = False
        if task_id:
            cancel_dispatched = self.task_manager.request_cancel(task_id)
        if not cancel_dispatched:
            self._terminate_train_process(process)
        if task_id:
            task = self.task_manager.get(task_id)
            if task is not None and task.final_event is None:
                final_event = self.task_manager.process_exited(task_id, -1, True)
                if self.train_process is process:
                    self._finish_train_process(False, final_event.message)

    def _terminate_train_process(self, process):
        """请求训练进程退出，必要时强制结束；该方法可被任务取消回调调用。"""
        try:
            if process.state() != QProcess.NotRunning:
                process.terminate()
                if not process.waitForFinished(3000):
                    process.kill()
                    process.waitForFinished(1500)
        except Exception:
            logger.exception("停止训练进程失败")

    def _get_python_exe(self):
        """返回可用的 Python 解释器路径。"""
        return Path(sys.executable)

    def _ensure_python_available(self):
        """源码模式兼容接口；冻结模式不要求目标机安装 Python。"""
        runtime = tools.get_runtime_paths()
        if runtime.frozen:
            return Path(sys.executable)
        python_exe = self._get_python_exe()
        if not python_exe.exists():
            QMessageBox.critical(self, "Python 不存在", f"找不到 Python 解释器:\n{python_exe}")
            self._append_log("error", f"找不到 Python 解释器: {python_exe}")
            return None
        return python_exe

    # ---------------------------------------------------------------
    # 功能 1：数据集准备
    # ---------------------------------------------------------------
    def _on_prepare(self):
        dlg = PrepareDialog(self, self.default_source, self.dataset_dir)
        if dlg.exec_() != QDialog.Accepted:
            return
        source = dlg.source_edit.text().strip()
        split = dlg.split_combo.currentText()
        clean = dlg.clean_check.isChecked()
        convert = dlg.convert_combo.currentText() if dlg.convert_check.isChecked() else None

        if not source:
            self._append_log("error", "源目录路径为空，请指定截图目录。")
            return
        if not Path(source).exists():
            self._append_log("error", f"源路径不存在: {source}")
            return
        ok, msg = tools.diagnose_prepare_source(source, self.dataset_dir)
        if not ok:
            self._append_log("error", msg)
            QMessageBox.warning(self, "源目录选择不安全", msg)
            return
        self.default_source = source
        self.config["source_dir"] = source
        tools.save_config(self.config)
        if clean:
            reply = QMessageBox.warning(
                self,
                "确认清空目标目录",
                f"即将先清空当前数据集 {split} 子集中的图片和标签，然后重新导入。\n\n"
                f"数据集位置:\n{self.dataset_dir}\n\n"
                f"继续前会自动备份 labels、classes.txt 和 data.yaml。\n\n"
                f"是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self._append_log("info", "已取消清空导入。")
                return

        self._append_log("info", f"开始数据集准备: 源={source}, 目标={split}, 格式转换={convert}")
        self._start_worker(
            tools.prepare_dataset,
            args=(source, split, clean, convert),
            kwargs=dict(dataset_dir=str(self.dataset_dir)),
        )

    # ---------------------------------------------------------------
    # 功能 2：数据集划分
    # ---------------------------------------------------------------
    def _on_split(self):
        self._open_split_dialog("repair")

    def _open_split_dialog(self, mode="repair"):
        self._append_log("info", f"打开数据集智能划分: 模式={mode}")
        try:
            from yolo_split_dialog import DatasetSplitDialog

            window = DatasetSplitDialog(
                self.dataset_dir,
                parent=self,
                initial_mode=mode,
            )
            window.split_completed.connect(
                lambda result: self._append_log(
                    "info",
                    f"智能划分完成: 移动 {result.moved_pairs} 张，"
                    f"备份={result.backup_dir}",
                )
            )
            self._show_child_window(window, "数据集智能划分")
        except Exception as exc:
            self._append_log("error", f"数据集智能划分启动失败: {exc}")
            QMessageBox.critical(
                self,
                "启动失败",
                f"无法打开数据集智能划分:\n{exc}",
            )

    # ---------------------------------------------------------------
    # 功能 3：启动标注工具
    # ---------------------------------------------------------------
    def _on_launch_annotator(self):
        # 检查 PyQt5 和 cv2
        if "PyQt5" in self.missing_deps:
            QMessageBox.critical(self, "依赖缺失", "标注工具需要 PyQt5，请先安装。\n\npip install PyQt5")
            return
        if "opencv-python" in self.missing_deps:
            QMessageBox.critical(self, "依赖缺失", "标注工具需要 opencv-python，请先安装。\n\npip install opencv-python")
            return

        self._append_log("info", "启动 YOLO 标注工具...")
        self._open_annotator_window()

    def _open_annotator_window(self, initial_image=None):
        if "PyQt5" in self.missing_deps or "opencv-python" in self.missing_deps:
            QMessageBox.critical(
                self,
                "依赖缺失",
                "标注工具需要 PyQt5 和 opencv-python，请先完成环境体检。",
            )
            return
        try:
            from yolo_annotate_gui import YoloAnnotatorGUI
            window = YoloAnnotatorGUI(
                dataset_dir=str(self.dataset_dir),
                initial_image=initial_image,
            )
            self._show_child_window(window, "标注工具")
            self._append_log("info", f"标注数据将保存到: {self.dataset_dir}")
        except Exception as e:
            self._append_log("error", f"启动失败: {e}")
            QMessageBox.critical(self, "启动失败", f"无法启动标注工具:\n{e}")

    def _show_child_window(self, window, name):
        """显示子窗口并保存引用，适配后续打包为单进程程序。"""
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.child_windows.append(window)
        window.destroyed.connect(lambda _=None, w=window: self._remove_child_window(w))
        window.show()
        window.raise_()
        window.activateWindow()
        self._append_log("info", f"{name}窗口已打开。")

    def _remove_child_window(self, window):
        if window in self.child_windows:
            self.child_windows.remove(window)

    # ---------------------------------------------------------------
    # 功能 4：格式校验
    # ---------------------------------------------------------------
    def _on_validate(self):
        self._append_log("info", "开始 YOLO 格式校验...")
        self._start_worker(tools.validate_labels, kwargs=dict(dataset_dir=str(self.dataset_dir)))

    # ---------------------------------------------------------------
    # 功能 5：数据质量检查
    # ---------------------------------------------------------------
    def _on_quality_check(self):
        self._append_log("info", "打开数据质量检查...")
        try:
            from yolo_quality_dialog import DatasetQualityDialog

            window = DatasetQualityDialog(self.dataset_dir, parent=self)
            window.locate_requested.connect(self._on_quality_locate)
            window.optimize_split_requested.connect(self._on_quality_optimize)
            self._show_child_window(window, "数据质量检查")
        except Exception as exc:
            self._append_log("error", f"数据质量检查启动失败: {exc}")
            QMessageBox.critical(
                self,
                "启动失败",
                f"无法打开数据质量检查:\n{exc}",
            )

    def _on_quality_locate(self, image_path):
        self._append_log("info", f"从质量报告定位图片: {image_path}")
        self._open_annotator_window(initial_image=image_path)

    def _on_quality_optimize(self, mode):
        self._append_log("info", "从质量报告打开划分优化。")
        self._open_split_dialog(mode)

    # ---------------------------------------------------------------
    # 数据集统计
    # ---------------------------------------------------------------
    def _on_stats(self):
        self._append_log("info", "开始数据集统计...")
        self._start_worker(tools.dataset_stats, kwargs=dict(dataset_dir=str(self.dataset_dir)))

    # ---------------------------------------------------------------
    # 功能 10：环境体检
    # ---------------------------------------------------------------
    def _on_health_check(self):
        report = tools.collect_environment_report(
            dataset_dir=str(self.dataset_dir),
            runs_dir=str(self.runs_dir),
        )
        level_label = {
            "info": "[信息]",
            "warning": "[警告]",
            "error": "[错误]",
        }
        lines = [f"{level_label.get(level, '[信息]')} {msg}" for level, msg in report["lines"]]
        text = "\n".join(lines)

        self._append_log(
            "info",
            f"环境体检完成: 错误 {report['errors']} 项，警告 {report['warnings']} 项",
        )
        for level, msg in report["lines"]:
            if level in {"warning", "error"}:
                self._append_log(level, msg)

        dlg = QDialog(self)
        dlg.setWindowTitle("环境体检")
        dlg.resize(760, 560)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        summary = QLabel(f"错误 {report['errors']} 项，警告 {report['warnings']} 项")
        tone = "danger" if report["errors"] else ("warning" if report["warnings"] else "success")
        theme.set_status_tone(summary, tone)
        layout.addWidget(summary)

        detail = QTextEdit()
        detail.setReadOnly(True)
        detail.setPlainText(text)
        detail.setFont(QFont("Consolas", 9))
        layout.addWidget(detail, 1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.button(QDialogButtonBox.Close).setText("关闭")
        theme.set_button_role(btn_box.button(QDialogButtonBox.Close), "secondary")
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        dlg.exec_()

    # ---------------------------------------------------------------
    # 功能 11：发布前自检
    # ---------------------------------------------------------------
    def _on_preflight_check(self):
        self._append_log("info", "开始发布前自检...")

        def run(emit=None):
            import preflight_check

            report = preflight_check.run_preflight(
                dataset_dir=str(self.dataset_dir),
                runs_dir=str(self.runs_dir),
                include_pip=True,
                include_env=True,
                emit=emit,
            )
            return report["errors"] == 0

        self._start_worker(run)

    # ---------------------------------------------------------------
    # 功能 12：备份恢复
    # ---------------------------------------------------------------
    def _on_backup_restore(self):
        dlg = BackupRestoreDialog(self, self.dataset_dir)
        if dlg.exec_() != QDialog.Accepted:
            return
        backup_dir = dlg.selected_backup_dir
        backup_info = dlg.selected_backup
        if not backup_dir:
            return

        full_restore = bool(
            backup_info and backup_info.get("has_split_manifest")
        )
        scope_text = (
            "图片分组、标签和配置"
            if full_restore
            else "标签和配置"
        )
        impact_text = (
            "恢复会移动已知图片回原分组；备份后新增文件将保留。"
            if full_restore
            else "恢复不会复制或修改图片文件。"
        )

        reply = QMessageBox.warning(
            self,
            "确认恢复备份",
            f"即将从以下备份恢复{scope_text}：\n\n{backup_dir}\n\n"
            f"恢复前会先自动备份当前 labels、classes.txt 和 data.yaml。\n"
            f"{impact_text}\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self._append_log("info", "已取消备份恢复。")
            return

        self._append_log("info", f"开始恢复备份: {backup_dir}")
        if full_restore:
            def restore_full(emit=None):
                from core.dataset_split_executor import SplitRecoveryService
                from core.paths import ProjectPaths

                paths = ProjectPaths.from_runtime_paths(
                    tools.get_runtime_paths(self.workspace_dir),
                    dataset_dir=self.dataset_dir,
                    runs_dir=self.runs_dir,
                    models_dir=self.models_dir,
                )
                result = SplitRecoveryService(paths).restore_backup(backup_dir)
                if emit:
                    emit("info", result.message)
                    if result.unknown_files:
                        emit(
                            "warning",
                            f"保留备份后新增文件: {result.unknown_files} 个",
                        )
                return result.success

            self._start_worker(restore_full)
        else:
            self._start_worker(
                tools.restore_dataset_backup,
                args=(str(backup_dir),),
                kwargs=dict(dataset_dir=str(self.dataset_dir)),
            )

    # ---------------------------------------------------------------
    # 功能 6：模型训练
    # ---------------------------------------------------------------
    def _on_train(self):
        if not ensure_ml_runtime(self, "模型训练"):
            return
        # 检查 ultralytics
        if "ultralytics" in self.missing_deps:
            QMessageBox.critical(
                self, "依赖缺失",
                "训练需要 ultralytics，请先安装。\n\npip install ultralytics",
            )
            return

        # 扫描可用模型
        models = tools.find_pretrained_models(
            project_dir=self.workspace_dir,
            models_dir=self.models_dir,
        )
        if not models:
            QMessageBox.warning(
                self, "无可用模型",
                "未在以下位置找到 .pt 模型文件：\n"
                f"- {self.workspace_dir}\n"
                f"- {self.models_dir}\n\n"
                "请先打开「模型管理」下载官方预训练模型。",
            )
            return

        dlg = TrainDialog(self, models, dataset_dir=self.dataset_dir, runs_dir=str(self.runs_dir))
        if dlg.exec_() != QDialog.Accepted:
            return

        model_path = dlg.model_combo.currentData()
        data_yaml = dlg.data_combo.currentData()
        epochs = dlg.epochs_spin.value()
        batch = dlg.batch_spin.value()
        imgsz = dlg.imgsz_spin.value()
        device = dlg.device_combo.currentData()  # 用 data 取实际值，不再是显示文本
        name = dlg.name_edit.text().strip() or "coc_detect"
        resume = dlg.resume_check.isChecked()

        self._append_log("info", f"开始训练: 模型={Path(model_path).name}, 数据={Path(data_yaml).name}, epochs={epochs}, batch={batch}")
        self._start_train_process(
            model_path=model_path,
            data_yaml=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(self.runs_dir),
            name=name,
            resume=resume,
        )

    # ---------------------------------------------------------------
    # 功能 15：模型管理
    # ---------------------------------------------------------------
    def _on_model_manager(self):
        self._append_log("info", "打开模型管理...")
        try:
            from yolo_model_manager import ModelManagerDialog

            window = ModelManagerDialog(
                models_dir=self.models_dir,
                parent=self,
            )
            self._show_child_window(window, "模型管理")
        except Exception as exc:
            self._append_log("error", f"模型管理启动失败: {exc}")
            QMessageBox.critical(self, "启动失败", f"无法打开模型管理：\n{exc}")

    def _on_runtime_manager(self):
        try:
            dialog = RuntimeManagerDialog(self)
            dialog.exec_()
            self._refresh_dep_status()
        except Exception as exc:
            self._append_log("error", f"运行环境管理启动失败: {exc}")
            QMessageBox.critical(self, "启动失败", f"无法打开运行环境管理：\n{exc}")

    # ---------------------------------------------------------------
    # 功能 7：模型测试
    # ---------------------------------------------------------------
    def _on_launch_detect(self):
        if not ensure_ml_runtime(self, "模型测试"):
            return
        if "ultralytics" in self.missing_deps:
            QMessageBox.critical(
                self, "依赖缺失",
                "模型测试需要 ultralytics，请先安装。\n\npip install ultralytics",
            )
            return

        self._append_log("info", "启动模型测试工具...")
        try:
            from yolo_detect_gui import YoloDetectGUI
            window = YoloDetectGUI(
                runs_dir=str(self.runs_dir),
                models_dir=str(self.models_dir),
            )
            self._show_child_window(window, "模型测试")
        except Exception as e:
            self._append_log("error", f"启动失败: {e}")
            QMessageBox.critical(self, "启动失败", f"无法启动模型测试:\n{e}")

    # ---------------------------------------------------------------
    # 功能 8：模型评估
    # ---------------------------------------------------------------
    def _on_model_evaluation(self):
        if not ensure_ml_runtime(self, "模型评估"):
            return
        if "ultralytics" in self.missing_deps:
            QMessageBox.critical(
                self,
                "依赖缺失",
                "模型评估需要 ultralytics，请先安装。\n\npip install ultralytics",
            )
            return

        data_yaml = self.dataset_dir / "data.yaml"
        configs = tools.find_data_configs(self.dataset_dir)
        if configs:
            data_yaml = configs[0]
        self._append_log("info", "打开模型评估...")
        try:
            from yolo_evaluation_dialog import ModelEvaluationDialog

            window = ModelEvaluationDialog(
                data_yaml=data_yaml,
                runs_dir=self.runs_dir,
                parent=self,
            )
            self._show_child_window(window, "模型评估")
        except Exception as exc:
            self._append_log("error", f"模型评估启动失败: {exc}")
            QMessageBox.critical(
                self,
                "启动失败",
                f"无法打开模型评估:\n{exc}",
            )

    # ---------------------------------------------------------------
    # 功能 9：TensorBoard
    # ---------------------------------------------------------------
    def _is_port_available(self, host, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return True
        except OSError:
            return False

    def _find_tensorboard_port(self, host="localhost", start_port=6006, max_tries=20):
        for port in range(start_port, start_port + max_tries):
            if self._is_port_available(host, port):
                return port
        return None

    def _on_tensorboard(self):
        runs_dir = self.runs_dir
        if not runs_dir.exists() or not any(runs_dir.iterdir()):
            QMessageBox.information(
                self, "无训练记录",
                "尚未找到训练结果目录（runs/）。\n请先完成一次训练后再查看 TensorBoard。",
            )
            return
        if not ensure_ml_runtime(self, "TensorBoard"):
            return

        worker_program, worker_args = build_worker_command(
            "tensorboard",
            [],
            resource_dir=PROJECT_DIR,
        )
        if not Path(worker_program).is_file():
            QMessageBox.critical(self, "文件缺失", f"找不到 TensorBoard Worker:\n{worker_program}")
            self._append_log("error", f"TensorBoard Worker 不存在: {worker_program}")
            return

        if (
            self.tensorboard_process
            and self.tensorboard_process.state() != QProcess.NotRunning
            and self.tensorboard_url
        ):
            if self._is_tensorboard_ready(self.tensorboard_url):
                try:
                    webbrowser.open(self.tensorboard_url)
                    self._append_log("info", f"TensorBoard 已在运行，已重新打开浏览器: {self.tensorboard_url}")
                except Exception as exc:
                    self._append_log("warning", f"浏览器打开失败，请手动访问 {self.tensorboard_url}；原因: {exc}")
                return

        host = "127.0.0.1"
        existing_url = f"http://{host}:6006"
        if self._is_tensorboard_ready(existing_url):
            self.tensorboard_url = existing_url
            try:
                webbrowser.open(existing_url)
                self._append_log("info", f"检测到 TensorBoard 已在运行，已打开浏览器: {existing_url}")
            except Exception as exc:
                self._append_log("warning", f"浏览器打开失败，请手动访问 {existing_url}；原因: {exc}")
            return

        port = self._find_tensorboard_port(host=host, start_port=6006)
        if port is None:
            QMessageBox.warning(self, "端口不可用", "未找到可用端口：6006-6025。请关闭占用端口的程序后重试。")
            self._append_log("error", "未找到可用 TensorBoard 端口：6006-6025")
            return
        if port != 6006:
            self._append_log("warning", f"端口 6006 已被占用，改用 {port}。")
        url = f"http://{host}:{port}"
        tb_log_path = tools.get_logs_dir() / "tensorboard.log"
        tb_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._append_log("info", f"启动 TensorBoard: {url}")
        try:
            task_id = str(uuid4())
            proc = QProcess(self)
            tensorboard_args = [
                "--logdir", str(runs_dir),
                "--host", host,
                "--port", str(port),
                "--task-id", task_id,
            ]
            worker_program, worker_args = build_worker_command(
                "tensorboard",
                tensorboard_args,
                resource_dir=PROJECT_DIR,
            )
            proc.setProgram(worker_program)
            proc.setArguments(worker_args)
            proc.setWorkingDirectory(str(PROJECT_DIR))
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(self._on_tensorboard_process_output)
            proc.finished.connect(self._on_tensorboard_process_finished)
            proc.errorOccurred.connect(self._on_tensorboard_process_error)
            self.tensorboard_process = proc
            self.tensorboard_url = url
            self._tensorboard_task_id = task_id
            self._tensorboard_output_buffer = ""
            self._tensorboard_log_path = tb_log_path
            self.task_manager.create_task(
                "tensorboard",
                task_id=task_id,
                cancel_callback=lambda: self._terminate_tensorboard_process(proc),
            )
            proc.start()
            if not proc.waitForStarted(3000) and self.tensorboard_process is proc:
                self._finish_tensorboard_task(
                    TaskEventType.FAILED,
                    f"无法启动 TensorBoard: {proc.errorString()}",
                    {"url": url},
                )
                self._terminate_tensorboard_process(proc)
                self.tensorboard_process = None
                self._tensorboard_task_id = None
                QMessageBox.critical(self, "启动失败", f"无法启动 TensorBoard:\n{proc.errorString()}")
        except Exception as e:
            self._append_log("error", f"启动失败: {e}")
            QMessageBox.critical(self, "启动失败", f"无法启动 TensorBoard:\n{e}")

    def _is_tensorboard_ready(self, url):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return 200 <= response.status < 500
        except Exception:
            return False

    def _read_tensorboard_log_tail(self, log_path, max_chars=1200):
        try:
            if log_path and Path(log_path).exists():
                return Path(log_path).read_text(encoding="utf-8", errors="replace")[-max_chars:]
        except Exception:
            pass
        return ""

    def _on_tensorboard_process_output(self):
        process = self.tensorboard_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._tensorboard_output_buffer += text
        while "\n" in self._tensorboard_output_buffer:
            line, self._tensorboard_output_buffer = self._tensorboard_output_buffer.split("\n", 1)
            self._handle_tensorboard_process_line(line.rstrip("\r"))

    def _handle_tensorboard_process_line(self, line):
        line = line.strip()
        if not line:
            return
        try:
            event = decode_task_event(line)
        except ValueError as exc:
            self._append_log("warning", f"TensorBoard 事件解析失败: {exc}")
            return
        if event is None:
            self._append_log("info", line)
            if self._tensorboard_log_path:
                try:
                    with open(self._tensorboard_log_path, "a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                except OSError:
                    pass
            return
        if event.task_id != self._tensorboard_task_id:
            return
        try:
            accepted = self.task_manager.accept(event)
        except ValueError:
            return
        if not accepted:
            return
        if event.type is TaskEventType.LOG:
            self._append_log(event.payload.get("level", "info"), event.message)
        elif event.type is TaskEventType.STARTED:
            self._append_log("info", event.message or "TensorBoard 启动中")
        elif event.type is TaskEventType.RESULT:
            self.tensorboard_url = str(event.payload.get("url") or self.tensorboard_url)
            self._append_log("info", event.message or f"TensorBoard 已就绪: {self.tensorboard_url}")
            try:
                webbrowser.open(self.tensorboard_url)
            except Exception as exc:
                self._append_log("warning", f"浏览器打开失败，请手动访问 {self.tensorboard_url}；原因: {exc}")
        elif event.type is TaskEventType.FAILED:
            self._append_log("error", event.message or "TensorBoard 启动失败")
            QMessageBox.warning(self, "TensorBoard", event.message or "TensorBoard 启动失败")
        elif event.type is TaskEventType.CANCELLED:
            self._append_log("warning", event.message or "TensorBoard 已取消")

    def _on_tensorboard_process_error(self, error):
        process = self.tensorboard_process
        if process is None:
            return
        if error == QProcess.FailedToStart:
            message = f"无法启动 TensorBoard: {process.errorString()}"
            self._finish_tensorboard_task(TaskEventType.FAILED, message)
            self._append_log("error", message)

    def _on_tensorboard_process_finished(self, exit_code, exit_status):
        if self._tensorboard_output_buffer.strip():
            self._handle_tensorboard_process_line(self._tensorboard_output_buffer.strip())
            self._tensorboard_output_buffer = ""
        task_id = self._tensorboard_task_id
        if task_id:
            task = self.task_manager.get(task_id)
            if task is not None and task.final_event is None:
                final = self.task_manager.process_exited(
                    task_id,
                    exit_code,
                    exit_status == QProcess.CrashExit,
                )
                self._append_log("error", final.message)
        process = self.tensorboard_process
        self.tensorboard_process = None
        self._tensorboard_task_id = None
        if process is not None:
            process.deleteLater()

    def _finish_tensorboard_task(self, event_type, message, payload=None):
        task_id = self._tensorboard_task_id
        if not task_id:
            return
        task = self.task_manager.get(task_id)
        if task is None or task.final_event is not None:
            return
        self.task_manager.accept(
            TaskEvent(
                task_id,
                event_type,
                "tensorboard",
                message=message,
                payload=payload or {},
            )
        )

    def _terminate_tensorboard_process(self, process):
        """结束 TensorBoard 启动包装进程及其子进程，避免留下后台服务。"""
        if process is None:
            return
        if isinstance(process, QProcess):
            if process.state() == QProcess.NotRunning:
                return
            try:
                if os.name == "nt" and process.processId():
                    result = subprocess.run(
                        ["taskkill", "/PID", str(process.processId()), "/T", "/F"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=5,
                    )
                    if result.returncode != 0 and process.state() != QProcess.NotRunning:
                        process.terminate()
                else:
                    process.terminate()
                if process.state() != QProcess.NotRunning and not process.waitForFinished(3000):
                    process.kill()
                    process.waitForFinished(3000)
            except Exception:
                logger.exception("清理 TensorBoard Qt 进程失败")
            return
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                )
                if result.returncode != 0 and process.poll() is None:
                    process.terminate()
            else:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        except Exception:
            logger.exception("清理 TensorBoard 进程失败")

    # ---------------------------------------------------------------
    # 功能 8：查看日志
    # ---------------------------------------------------------------
    def _on_view_log(self):
        if LOG_FILE.exists():
            try:
                os.startfile(str(LOG_FILE))  # Windows: 用默认编辑器打开
            except Exception as e:
                self._append_log("error", f"打开日志失败: {e}")
                QMessageBox.warning(self, "打开失败", f"无法打开日志文件:\n{LOG_FILE}\n{e}")
        else:
            self._append_log("warning", "日志文件尚未生成（无操作记录）。")
            QMessageBox.information(self, "提示", "日志文件尚未生成，执行任意操作后才会创建。")

    def _on_export_diagnostics(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default_path = Path.home() / "Documents" / f"YOLO工具箱-诊断-{timestamp}.zip"
        selected, _file_filter = QFileDialog.getSaveFileName(
            self,
            "导出诊断包",
            str(default_path),
            "ZIP 压缩包 (*.zip)",
        )
        if not selected:
            return
        try:
            output = build_diagnostic_archive(
                Path(selected),
                runtime_paths=self.runtime_paths,
                include_user_data=False,
            )
        except Exception as exc:
            logger.exception("导出诊断包失败")
            self._append_log("error", f"导出诊断包失败: {exc}")
            QMessageBox.warning(self, "导出失败", f"无法生成诊断包：\n{exc}")
            return
        self._append_log("info", f"诊断包已导出: {output}")
        QMessageBox.information(self, "导出完成", f"诊断包已生成：\n{output}")

    def _on_about(self):
        AboutDialog(self, self.runtime_paths).exec_()

    # ---------------------------------------------------------------
    # 关闭事件
    # ---------------------------------------------------------------
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "任务进行中",
                "有任务正在执行，是否强制退出？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            if self._worker_task_id:
                self.task_manager.request_cancel(self._worker_task_id)
            self.worker.terminate()
            self.worker.wait(2000)
            if self._worker_task_id:
                task = self.task_manager.get(self._worker_task_id)
                if task is not None and task.final_event is None:
                    self.task_manager.process_exited(self._worker_task_id, -1, True)
        if self.train_process is not None:
            self._stop_train_process()
        if self.tensorboard_process is not None:
            try:
                if self._tensorboard_task_id:
                    self.task_manager.request_cancel(self._tensorboard_task_id)
                self._terminate_tensorboard_process(self.tensorboard_process)
                self._append_log("info", "TensorBoard 进程已清理")
            except Exception as exc:
                logger.exception("清理 TensorBoard 失败")
                self._append_log("warning", f"清理 TensorBoard 失败: {exc}")
            finally:
                self.tensorboard_process = None
                self._tensorboard_task_id = None
        event.accept()


class AboutDialog(QDialog):
    def __init__(self, parent, runtime_paths: RuntimePaths):
        super().__init__(parent)
        self.runtime_paths = runtime_paths
        self.source_url = get_source_url()
        self.setWindowTitle(f"关于 {PRODUCT_NAME}")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        title = QLabel(PRODUCT_NAME)
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        details = QLabel(
            f"版本 {DISPLAY_VERSION}\n"
            f"发布者 {PUBLISHER_NAME}\n"
            f"许可证 {LICENSE_ID}"
        )
        details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(details)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        license_button = QPushButton("许可证")
        notices_button = QPushButton("第三方声明")
        source_button = QPushButton("源码")
        for button in (license_button, notices_button, source_button):
            theme.set_button_role(button, "secondary")
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        license_button.clicked.connect(
            lambda: self._open_local_document("LICENSE", "许可证")
        )
        notices_button.clicked.connect(
            lambda: self._open_local_document(
                "THIRD_PARTY_NOTICES.md",
                "第三方声明",
            )
        )
        source_button.clicked.connect(self._open_source)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open_local_document(self, filename: str, title: str) -> None:
        path = self.runtime_paths.install_dir / filename
        if not path.is_file():
            QMessageBox.warning(self, "文件缺失", f"未找到{title}：\n{path}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                webbrowser.open(path.resolve().as_uri())
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", f"无法打开{title}：\n{exc}")

    def _open_source(self) -> None:
        if not self.source_url:
            QMessageBox.information(self, "源码地址", "当前开发构建未配置公开源码地址。")
            return
        webbrowser.open(self.source_url)


# ---------------------------------------------------------------------------
# 数据集准备对话框（使用 QDialog，避免 QMessageBox 无 setCustomWidget 的崩溃）
# ---------------------------------------------------------------------------
class PrepareDialog(QDialog):
    def __init__(self, parent, default_source, dataset_dir):
        super().__init__(parent)
        self.setWindowTitle("数据集准备")
        self.setMinimumWidth(500)
        self.dataset_dir = Path(dataset_dir)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # 源目录
        src_row = QHBoxLayout()
        src_row.setSpacing(8)
        src_row.addWidget(QLabel("截图目录:"))
        self.source_edit = QLineEdit(default_source)
        self.source_edit.setMinimumWidth(350)
        self.source_edit.textChanged.connect(self._update_preview)
        src_row.addWidget(self.source_edit)
        browse_btn = QPushButton("浏览...")
        theme.set_button_role(browse_btn, "secondary")
        browse_btn.clicked.connect(self._browse_source)
        src_row.addWidget(browse_btn)
        layout.addLayout(src_row)

        # 图片计数预览（让用户确认目录中确实有图片）
        self.preview_label = QLabel("")
        self.preview_label.setFont(QFont("", 9))
        self.preview_label.setWordWrap(True)
        theme.set_status_tone(self.preview_label, "info")
        layout.addWidget(self.preview_label)

        # 目标 split
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("导入到:"))
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test"])
        self.split_combo.setCurrentText("train")
        split_row.addWidget(self.split_combo)
        split_row.addStretch()
        layout.addLayout(split_row)

        # 清空选项
        self.clean_check = QCheckBox("清空目标目录后再导入（原文件将被删除）")
        layout.addWidget(self.clean_check)

        # 格式转换
        fmt_row = QHBoxLayout()
        self.convert_check = QCheckBox("格式转换:")
        self.convert_combo = QComboBox()
        self.convert_combo.addItems(["jpg", "png"])
        self.convert_combo.setEnabled(False)
        self.convert_check.toggled.connect(self.convert_combo.setEnabled)
        fmt_row.addWidget(self.convert_check)
        fmt_row.addWidget(self.convert_combo)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        # 提示
        tip = QLabel("保留原 prepare_dataset.py 核心逻辑：将截图复制到 dataset/images/{split}。")
        tip.setFont(QFont("", 8))
        tip.setWordWrap(True)
        theme.set_text_role(tip, "hint")
        layout.addWidget(tip)

        # Ok / Cancel 按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("开始导入")
        theme.set_button_role(btn_box.button(QDialogButtonBox.Ok), "primary")
        theme.set_button_role(btn_box.button(QDialogButtonBox.Cancel), "secondary")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # 初始化预览
        self._update_preview()

    def _browse_source(self):
        # 使用 Qt 内置对话框（DontUseNativeDialog），避免 Windows 原生对话框
        # 在中文路径下显示为空的问题
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择截图目录", self.source_edit.text(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if dir_path:
            self.source_edit.setText(dir_path)

    def _update_preview(self):
        """实时显示源目录中找到的图片数量，让用户确认路径正确。"""
        src = self.source_edit.text().strip()
        if not src:
            self.preview_label.setText("")
            theme.set_status_tone(self.preview_label, "info")
            return
        src_path = Path(src)
        if not src_path.exists():
            self.preview_label.setText(f"路径不存在: {src}")
            theme.set_status_tone(self.preview_label, "danger")
            return
        if not src_path.is_dir():
            self.preview_label.setText(f"不是目录: {src}")
            theme.set_status_tone(self.preview_label, "danger")
            return
        ok, msg = tools.diagnose_prepare_source(src_path, self.dataset_dir)
        if not ok:
            self.preview_label.setText(msg)
            theme.set_status_tone(self.preview_label, "danger")
            return
        n = len(tools.iter_images(src_path))
        if n > 0:
            self.preview_label.setText(f"找到 {n} 张图片 ✓")
            theme.set_status_tone(self.preview_label, "success")
        else:
            self.preview_label.setText("未找到图片（目录为空或格式不支持）")
            theme.set_status_tone(self.preview_label, "warning")


# ---------------------------------------------------------------------------
# 备份恢复对话框
# ---------------------------------------------------------------------------
class BackupRestoreDialog(QDialog):
    def __init__(self, parent, dataset_dir):
        super().__init__(parent)
        self.setWindowTitle("备份恢复")
        self.resize(780, 520)
        self.dataset_dir = Path(dataset_dir)
        self.selected_backup_dir = None
        self.selected_backup = None
        self.backups = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.summary_label = QLabel()
        theme.set_status_tone(self.summary_label, "info")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["时间", "原因", "标签文件", "恢复范围", "包含内容", "备份目录"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_button_state)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.refresh_btn = QPushButton("刷新")
        theme.set_button_role(self.refresh_btn, "secondary")
        self.refresh_btn.clicked.connect(self._load_backups)
        btn_row.addWidget(self.refresh_btn)

        self.open_btn = QPushButton("打开目录")
        theme.set_button_role(self.open_btn, "secondary")
        self.open_btn.clicked.connect(self._open_backup_dir)
        btn_row.addWidget(self.open_btn)

        self.restore_btn = QPushButton("恢复选中")
        theme.set_button_role(self.restore_btn, "warning")
        self.restore_btn.clicked.connect(self._accept_restore)
        btn_row.addWidget(self.restore_btn)

        close_btn = QPushButton("关闭")
        theme.set_button_role(close_btn, "secondary")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._load_backups()

    def _load_backups(self):
        self.backups = tools.list_dataset_backups(self.dataset_dir)
        self.table.setRowCount(0)
        for row, backup in enumerate(self.backups):
            self.table.insertRow(row)
            content = []
            if backup["has_labels"]:
                content.append("labels")
            if backup["has_classes"]:
                content.append("classes.txt")
            if backup["has_data_yaml"]:
                content.append("data.yaml")
            values = [
                backup["time"],
                backup["reason"],
                str(backup["label_files"]),
                backup["restore_scope"],
                " / ".join(content) if content else "无",
                str(backup["path"]),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setData(Qt.UserRole, str(backup["path"]))
                self.table.setItem(row, col, item)

        if self.backups:
            self.table.selectRow(0)
            self.summary_label.setText(f"找到 {len(self.backups)} 个备份。")
            theme.set_status_tone(self.summary_label, "success")
        else:
            self.summary_label.setText(f"尚未找到备份目录: {self.dataset_dir / 'backups'}")
            theme.set_status_tone(self.summary_label, "warning")
        self._update_button_state()

    def _selected_backup(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return Path(value) if value else None

    def _open_backup_dir(self):
        path = self._selected_backup() or (self.dataset_dir / "backups")
        if not path.exists():
            QMessageBox.information(self, "无备份目录", f"目录不存在:\n{path}")
            return
        try:
            os.startfile(str(path))
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", f"无法打开目录:\n{path}\n{exc}")

    def _accept_restore(self):
        path = self._selected_backup()
        if path is None:
            QMessageBox.information(self, "未选择备份", "请先选择一个备份。")
            return
        self.selected_backup_dir = path
        self.selected_backup = next(
            (
                backup
                for backup in self.backups
                if Path(backup["path"]).resolve() == path.resolve()
            ),
            None,
        )
        self.accept()

    def _update_button_state(self):
        has_selection = self._selected_backup() is not None
        self.restore_btn.setEnabled(has_selection)
        self.open_btn.setEnabled(bool(self.backups) or (self.dataset_dir / "backups").exists())


# ---------------------------------------------------------------------------
# 模型训练对话框
# ---------------------------------------------------------------------------
class TrainDialog(QDialog):
    def __init__(self, parent, models, dataset_dir=None, runs_dir=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO 模型训练")
        self.resize(760, 620)
        self.setMinimumSize(680, 560)
        self._dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
        self._runs_dir = runs_dir or ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(theme.SPACING["md"])
        layout.addWidget(SectionHeader("模型训练"))

        # 训练来源始终可见，便于在切换参数页时核对模型与数据配置。
        self.source_panel = theme.set_panel(QWidget(), "TrainingSource")
        source_layout = QGridLayout(self.source_panel)
        source_layout.setContentsMargins(14, 12, 14, 12)
        source_layout.setHorizontalSpacing(theme.SPACING["sm"])
        source_layout.setVerticalSpacing(theme.SPACING["sm"])
        source_layout.setColumnStretch(1, 1)

        source_header = SectionHeader("训练来源")
        source_layout.addWidget(source_header, 0, 0, 1, 2)
        self.data_status_badge = StatusBadge("正在检查", "info")
        self.data_status_badge.setAlignment(Qt.AlignCenter)
        source_layout.addWidget(self.data_status_badge, 0, 2, 1, 1, Qt.AlignRight)

        model_label = QLabel("预训练模型")
        model_label.setMinimumWidth(78)
        source_layout.addWidget(model_label, 1, 0)
        self.model_combo = QComboBox()
        for m in models:
            label = m.name
            if m.name == "yolov8n.pt":
                label = "yolov8n.pt (推荐·轻量快速)"
            elif m.name == "yolov8s.pt":
                label = "yolov8s.pt (精度更高·较慢)"
            elif m.name == "yolo26n.pt":
                label = "yolo26n.pt (最新·轻量)"
            elif "best" in m.name:
                label = f"{m.name} (上次训练最佳)"
            elif m.stem.lower().startswith("yolo") and m.stem.lower().endswith("m"):
                label = f"{m.name} (中型·训练较慢)"
            elif m.stem.lower().startswith("yolo") and m.stem.lower().endswith("l"):
                label = f"{m.name} (大型·训练很慢)"
            elif m.stem.lower().startswith("yolo") and m.stem.lower().endswith("x"):
                label = f"{m.name} (超大型·训练最慢)"
            self.model_combo.addItem(label, str(m))
        self._select_recommended_model()
        self.model_combo.setToolTip("选择训练的初始权重")
        source_layout.addWidget(self.model_combo, 1, 1, 1, 2)

        self.model_hint_label = QLabel()
        self.model_hint_label.setWordWrap(True)
        source_layout.addWidget(self.model_hint_label, 2, 1, 1, 2)
        self.model_combo.currentIndexChanged.connect(self._refresh_model_hint)

        data_label = QLabel("数据集配置")
        source_layout.addWidget(data_label, 3, 0)
        self.data_combo = QComboBox()
        configs = tools.find_data_configs(self._dataset_dir)
        if configs:
            for c in configs:
                self.data_combo.addItem(c.name, str(c))
        else:
            self.data_combo.addItem("data.yaml（默认）", str(self._dataset_dir / "data.yaml"))
        self.data_combo.setToolTip("选择训练用的数据集配置文件（.yaml）")
        source_layout.addWidget(self.data_combo, 3, 1)
        self.data_browse_button = QPushButton("浏览")
        self.data_browse_button.setIcon(
            self.style().standardIcon(QStyle.SP_DialogOpenButton)
        )
        self.data_browse_button.setToolTip("选择数据集 YAML 配置文件")
        self.data_browse_button.setFixedWidth(104)
        theme.set_button_role(self.data_browse_button, "secondary")
        self.data_browse_button.clicked.connect(self._browse_data_config)
        source_layout.addWidget(self.data_browse_button, 3, 2)
        self.data_combo.currentIndexChanged.connect(self._refresh_summary)
        layout.addWidget(self.source_panel)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("TrainingTabs")
        self.settings_tabs.setDocumentMode(True)

        self.basic_tab = QWidget()
        self.basic_tab.setObjectName("TrainingBasic")
        basic_layout = QGridLayout(self.basic_tab)
        basic_layout.setContentsMargins(16, 16, 16, 16)
        basic_layout.setHorizontalSpacing(theme.SPACING["lg"])
        basic_layout.setVerticalSpacing(theme.SPACING["sm"])

        basic_layout.addWidget(QLabel("训练轮数 (epochs)"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        self.epochs_spin.setSuffix(" 轮")
        self.epochs_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        basic_layout.addWidget(self.epochs_spin, 0, 1)

        basic_layout.addWidget(QLabel("批次大小 (batch)"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(4)
        self.batch_spin.setSuffix(" 张")
        self.batch_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        basic_layout.addWidget(self.batch_spin, 1, 1)

        basic_layout.addWidget(QLabel("训练设备"), 2, 0)
        self.device_combo = QComboBox()
        devices = tools.detect_devices()
        for label, value in devices:
            self.device_combo.addItem(label, value)
        self._gpu_available = any(
            str(value) not in ("", "cpu") for _label, value in devices
        )
        self._batch_recommendation = self.batch_spin.value()
        self._apply_recommended_batch(force=True)
        self.device_combo.currentIndexChanged.connect(self._on_train_device_changed)
        self.device_combo.setToolTip("选择训练使用的计算设备\n"
                                     "GPU=优先加速；自动=交给 Ultralytics；CPU=兼容模式")
        basic_layout.addWidget(self.device_combo, 2, 1)

        basic_layout.addWidget(QLabel("训练名称"), 3, 0)
        self.name_edit = QLineEdit("coc_detect")
        self.name_edit.setToolTip("结果保存在 runs/{名称}/ 下，相同名称会递增编号")
        basic_layout.addWidget(self.name_edit, 3, 1)
        basic_layout.setColumnStretch(1, 1)
        basic_layout.setRowStretch(4, 1)
        self.settings_tabs.addTab(self.basic_tab, "基础参数")

        self.advanced_tab = QWidget()
        self.advanced_tab.setObjectName("TrainingAdvanced")
        advanced_layout = QGridLayout(self.advanced_tab)
        advanced_layout.setContentsMargins(16, 16, 16, 16)
        advanced_layout.setHorizontalSpacing(theme.SPACING["lg"])
        advanced_layout.setVerticalSpacing(theme.SPACING["md"])

        advanced_layout.addWidget(QLabel("输入图像尺寸 (imgsz)"), 0, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSuffix(" px")
        self.imgsz_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        advanced_layout.addWidget(self.imgsz_spin, 0, 1)

        self.resume_check = QCheckBox("从上次中断处继续训练（resume）")
        self.resume_check.setToolTip("勾选后从 runs/ 下最近的 last.pt 继续训练")
        advanced_layout.addWidget(self.resume_check, 1, 0, 1, 2)
        advanced_layout.setColumnStretch(1, 1)
        advanced_layout.setRowStretch(2, 1)
        self.settings_tabs.addTab(self.advanced_tab, "高级参数")

        self.data_check_tab = QWidget()
        self.data_check_tab.setObjectName("TrainingDataCheck")
        data_check_layout = QVBoxLayout(self.data_check_tab)
        data_check_layout.setContentsMargins(16, 16, 16, 16)
        data_check_layout.setSpacing(theme.SPACING["md"])
        data_check_layout.addWidget(SectionHeader("当前数据集"))
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        theme.set_status_tone(self.summary_label, "info")
        data_check_layout.addWidget(self.summary_label)
        self.training_estimate_label = QLabel()
        self.training_estimate_label.setObjectName("TrainingEstimate")
        self.training_estimate_label.setWordWrap(True)
        theme.set_text_role(self.training_estimate_label, "muted")
        data_check_layout.addWidget(self.training_estimate_label)
        data_check_layout.addStretch(1)
        self.settings_tabs.addTab(self.data_check_tab, "数据检查")
        layout.addWidget(self.settings_tabs, 1)
        self._training_summary = None
        self.batch_spin.valueChanged.connect(self._refresh_training_estimate)

        output_root = Path(self._runs_dir) if self._runs_dir else Path("runs")
        output_path = output_root / "{名称}" / "weights"
        self.footer_panel = QWidget()
        self.footer_panel.setObjectName("TrainingFooter")
        footer_layout = QVBoxLayout(self.footer_panel)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(theme.SPACING["sm"])

        self.output_label = QLabel(
            f"训练输出：{output_path}\n"
            "训练完成后将生成 best.pt（最佳）和 last.pt（末轮）。"
        )
        self.output_label.setWordWrap(True)
        theme.set_text_role(self.output_label, "hint")
        footer_layout.addWidget(self.output_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("开始训练")
        self.button_box.button(QDialogButtonBox.Cancel).setText("取消")
        self.button_box.button(QDialogButtonBox.Ok).setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay)
        )
        theme.set_button_role(self.button_box.button(QDialogButtonBox.Ok), "primary")
        theme.set_button_role(self.button_box.button(QDialogButtonBox.Cancel), "secondary")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        footer_layout.addWidget(self.button_box)
        layout.addWidget(self.footer_panel)
        self._refresh_model_hint()
        self._refresh_summary()

    def _select_recommended_model(self):
        """优先选择轻量模型，避免新增大模型后因文件名排序误选。"""
        preferred_names = ("yolov8n.pt", "yolo11n.pt", "yolo26n.pt")
        for name in preferred_names:
            for index in range(self.model_combo.count()):
                current = self.model_combo.itemData(index)
                if current and Path(current).name.lower() == name:
                    self.model_combo.setCurrentIndex(index)
                    return
        for size in ("n", "s", "m", "l", "x"):
            for index in range(self.model_combo.count()):
                current = self.model_combo.itemData(index)
                if not current:
                    continue
                stem = Path(current).stem.lower()
                if stem.startswith("yolo") and stem.endswith(size):
                    self.model_combo.setCurrentIndex(index)
                    return

    def _refresh_model_hint(self, _index=None):
        current = self.model_combo.currentData()
        stem = Path(current).stem.lower() if current else ""
        size = stem[-1:] if stem.startswith("yolo") else ""
        hints = {
            "n": ("轻量模型：适合快速迭代，当前推荐。", "success"),
            "s": ("小型模型：速度与精度较平衡。", "info"),
            "m": ("中型模型：训练耗时和显存占用会明显增加。", "warning"),
            "l": ("大型模型：训练耗时和显存占用会明显增加。", "warning"),
            "x": ("超大型模型：训练耗时很长，并需要更多显存。", "danger"),
        }
        message, tone = hints.get(
            size,
            ("自定义模型：训练速度取决于模型规模。", "info"),
        )
        self.model_hint_label.setText(message)
        theme.set_status_tone(self.model_hint_label, tone)

    def _browse_data_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置文件",
            str(self._dataset_dir),
            "YAML 配置文件 (*.yaml *.yml);;所有文件 (*)",
        )
        if path:
            # 如果不在下拉列表中，添加进去
            found = False
            for i in range(self.data_combo.count()):
                if self.data_combo.itemData(i) == path:
                    self.data_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.data_combo.addItem(Path(path).name, path)
                self.data_combo.setCurrentIndex(self.data_combo.count() - 1)
            self._refresh_summary()

    def _refresh_summary(self):
        """根据当前选择的数据集配置，刷新概要信息。"""
        current_data = self.data_combo.currentData()
        if not current_data:
            self._training_summary = None
            self.summary_label.setText("尚未选择数据集配置。")
            theme.set_status_tone(self.summary_label, "danger")
            self.data_status_badge.setText("未选择配置")
            self.data_status_badge.set_tone("danger")
            self._set_train_enabled(False)
            self._refresh_training_estimate()
            return

        data_path = Path(current_data)
        try:
            summary = tools.data_yaml_summary(data_path)
            train_n = summary["train_images"]
            val_n = summary["val_images"]
            test_n = summary["test_images"]
            n_cls = summary["n_classes"]
        except Exception as exc:
            self._training_summary = None
            self.summary_label.setText(f"无法读取数据集概要：{exc}")
            theme.set_status_tone(self.summary_label, "danger")
            self.data_status_badge.setText("读取失败")
            self.data_status_badge.set_tone("danger")
            self._set_train_enabled(False)
            self._refresh_training_estimate()
            return

        self._training_summary = {
            "train_images": train_n,
            "val_images": val_n,
        }
        html = f"<b>数据集概要</b>: 训练集 {train_n} 张 | 验证集 {val_n} 张 | 测试集 {test_n} 张 | 类别 {n_cls} 个"
        if train_n == 0:
            html += "<br>训练集为空！请先准备数据集并标注。"
            theme.set_status_tone(self.summary_label, "danger")
            self.data_status_badge.setText("训练集不可用")
            self.data_status_badge.set_tone("danger")
            self._set_train_enabled(False)
        elif n_cls == 0:
            html += "<br>未配置任何类别！请先检查 classes.txt 和 data.yaml。"
            theme.set_status_tone(self.summary_label, "danger")
            self.data_status_badge.setText("类别不可用")
            self.data_status_badge.set_tone("danger")
            self._set_train_enabled(False)
        elif val_n == 0:
            html += "<br>验证集为空，建议先执行「数据集划分」。"
            theme.set_status_tone(self.summary_label, "warning")
            self.data_status_badge.setText("建议划分验证集")
            self.data_status_badge.set_tone("warning")
            self._set_train_enabled(True)
        else:
            theme.set_status_tone(self.summary_label, "success")
            self.data_status_badge.setText("数据可用")
            self.data_status_badge.set_tone("success")
            self._set_train_enabled(True)
        self.summary_label.setText(html)
        self._refresh_training_estimate()

    def _recommended_batch_for_device(self):
        device = self.device_combo.currentData()
        if device == "cpu" or (device == "" and not self._gpu_available):
            return 4
        return 16

    def _apply_recommended_batch(self, force=False):
        previous = self._batch_recommendation
        recommended = self._recommended_batch_for_device()
        should_apply = force or self.batch_spin.value() == previous
        self._batch_recommendation = recommended
        if should_apply:
            self.batch_spin.setValue(recommended)

    def _on_train_device_changed(self, _index):
        self._apply_recommended_batch()
        self._refresh_training_estimate()

    def _refresh_training_estimate(self, _value=None):
        if not self._training_summary:
            self.training_estimate_label.setText("预计每轮：暂无可用数据")
            return
        batch = max(1, self.batch_spin.value())
        train_n = self._training_summary["train_images"]
        val_n = self._training_summary["val_images"]
        train_batches = (train_n + batch - 1) // batch
        val_batches = (val_n + batch - 1) // batch if val_n else 0
        total_batches = train_batches + val_batches
        self.training_estimate_label.setText(
            f"预计每轮：训练 {train_batches} 个 batch | "
            f"验证 {val_batches} 个 batch | 共 {total_batches} 个"
        )

    def _set_train_enabled(self, enabled):
        """仅在训练数据不可用时阻止提交对话框。"""
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(bool(enabled))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    freeze_support()
    app = QApplication(sys.argv)
    theme.apply_app_theme(app)
    runtime = tools.get_runtime_paths()
    install_global_exception_hook(runtime)
    instance_guard = SingleInstanceGuard(
        instance_name_for_state(runtime.state_dir)
    )
    if not instance_guard.acquire():
        instance_guard.notify_existing()
        return 0
    config = prepare_initial_workspace(tools.load_config())
    if config is None:
        instance_guard.close()
        return 1
    window = YoloToolLauncher(config=config)
    instance_guard.activation_requested.connect(
        lambda: _activate_existing_window(window)
    )
    app.aboutToQuit.connect(instance_guard.close)
    window._single_instance_guard = instance_guard
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

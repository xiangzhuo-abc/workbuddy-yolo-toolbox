"""YOLO 工具箱运行环境安装、选择和管理界面。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
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

try:
    from .core.external_python import (
        ExternalPythonCandidate,
        discover_external_pythons,
        probe_external_python,
    )
    from .core.ml_runtime import (
        RuntimeArtifact,
        RuntimeCandidate,
        RuntimeCatalog,
        RuntimeDiscovery,
        RuntimeProfile,
        RuntimeSelection,
        RuntimeStateStore,
    )
    from .core.runtime_installer import (
        DownloadProgress,
        RuntimeInstallError,
        RuntimeInstaller,
    )
    from .core.runtime_paths import MLRuntimePaths, RuntimePaths
except ImportError:
    from core.external_python import (
        ExternalPythonCandidate,
        discover_external_pythons,
        probe_external_python,
    )
    from core.ml_runtime import (
        RuntimeArtifact,
        RuntimeCandidate,
        RuntimeCatalog,
        RuntimeDiscovery,
        RuntimeProfile,
        RuntimeSelection,
        RuntimeStateStore,
    )
    from core.runtime_installer import (
        DownloadProgress,
        RuntimeInstallError,
        RuntimeInstaller,
    )
    from core.runtime_paths import MLRuntimePaths, RuntimePaths


CATALOG_ENV = "YOLO_TOOLBOX_RUNTIME_CATALOG"
PROJECT_DIR = Path(__file__).resolve().parent.parent


def external_worker_source_path(
    *,
    frozen: bool | None = None,
    resource_dir: Path | None = None,
) -> Path:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if not is_frozen:
        return PROJECT_DIR / "tools" / "yolo_worker_entry.py"
    resource = (
        Path(resource_dir)
        if resource_dir is not None
        else RuntimePaths.from_environment(frozen=True).resource_dir
    )
    return resource / "worker_source" / "yolo_worker_entry.py"


def load_runtime_catalog(
    *,
    frozen: bool | None = None,
    resource_dir: Path | None = None,
) -> RuntimeCatalog:
    """读取源码配置或冻结产物内置的可信运行时清单。"""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    override = str(os.environ.get(CATALOG_ENV, "")).strip()
    if override and not is_frozen:
        return RuntimeCatalog.from_file(Path(override))
    if is_frozen:
        resource = (
            Path(resource_dir)
            if resource_dir is not None
            else RuntimePaths.from_environment(frozen=True).resource_dir
        )
        candidates = (
            resource / "runtime_catalog.json",
            resource / "runtime_profiles.json",
        )
    else:
        candidates = (
            PROJECT_DIR / "tmp" / "runtime-candidate" / "runtime_catalog.json",
            PROJECT_DIR / "packaging" / "runtime_profiles.json",
        )
    for path in candidates:
        if path.is_file():
            return RuntimeCatalog.from_file(path)
    raise ValueError("未找到可信运行时清单")


def recommend_runtime_profile() -> RuntimeProfile:
    """不导入 Torch，仅通过 NVIDIA 工具判断推荐下载项。"""
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=creation_flags,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return RuntimeProfile.CPU
    return RuntimeProfile.CUDA118 if result.returncode == 0 and result.stdout.strip() else RuntimeProfile.CPU


def runtime_status_summary(
    *,
    frozen: bool | None = None,
    catalog_value: RuntimeCatalog | None = None,
    runtime_paths: MLRuntimePaths | None = None,
) -> tuple[str, str]:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if not is_frozen:
        return "模型环境: 开发环境", "success"
    try:
        paths = runtime_paths or MLRuntimePaths.from_environment()
        selection = RuntimeStateStore(paths.state_file).load()
        if selection.backend_kind == "external" and selection.external_python:
            if Path(selection.external_python).is_file():
                return "模型环境: 外部 Python 已选择", "success"
            return "模型环境: 外部 Python 已失效", "warning"
        catalog = catalog_value or load_runtime_catalog(frozen=True)
        candidates = RuntimeDiscovery(
            catalog,
            paths,
            RuntimeStateStore(paths.state_file),
        ).find_compatible()
    except Exception:
        return "模型环境: 需要修复", "danger"
    if not candidates:
        return "模型环境: 未安装", "warning"
    candidate = next((item for item in candidates if item.selected), candidates[0])
    label = "NVIDIA GPU" if candidate.artifact.profile is RuntimeProfile.CUDA118 else "CPU"
    return f"模型环境: {label} 已就绪", "success"


class RuntimeInstallThread(QThread):
    progress_changed = pyqtSignal(object)
    installed = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(
        self,
        installer: RuntimeInstaller,
        artifact: RuntimeArtifact | None = None,
        *,
        offline_path: Path | None = None,
        catalog: RuntimeCatalog | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.installer = installer
        self.artifact = artifact
        self.offline_path = offline_path
        self.catalog = catalog
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if self.offline_path is not None and self.catalog is not None:
                candidate = self.installer.import_offline(
                    self.offline_path,
                    self.catalog,
                    progress=self.progress_changed.emit,
                )
            elif self.artifact is not None:
                candidate = self.installer.download_and_install(
                    self.artifact,
                    cancel=self._cancel_event.is_set,
                    progress=self.progress_changed.emit,
                )
            else:
                raise RuntimeInstallError("missing_source", "没有可安装的运行时来源")
        except BaseException as exc:
            self.failed.emit(exc)
            return
        self.installed.emit(candidate)


class ExternalScanThread(QThread):
    scanned = pyqtSignal(object)

    def __init__(self, worker_source: Path, parent=None):
        super().__init__(parent)
        self.worker_source = Path(worker_source)

    def run(self) -> None:
        candidates = [
            probe_external_python(executable, self.worker_source)
            for executable in discover_external_pythons()
        ]
        self.scanned.emit(candidates)


class ExternalPythonDialog(QDialog):
    def __init__(
        self,
        paths: MLRuntimePaths,
        parent=None,
        *,
        worker_source: Path | None = None,
    ):
        super().__init__(parent)
        self.paths = paths
        self.worker_source = worker_source or external_worker_source_path()
        self.candidates: list[ExternalPythonCandidate] = []
        self.scan_thread: ExternalScanThread | None = None
        self.setWindowTitle("高级检测外部 Python 环境")
        self.resize(820, 420)
        self._build_ui()
        self._start_scan()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        note = QLabel(
            "仅检查 Windows Python Launcher 和 PATH 中明确列出的解释器，"
            "不会遍历磁盘，也不会修改外部环境。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Python", "版本", "架构", "GPU", "状态"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        self.status_label = QLabel("正在检查已注册的 Python 环境...")
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("重新检测")
        self.use_button = QPushButton("使用所选环境")
        close_button = QPushButton("取消")
        self.use_button.setEnabled(False)
        buttons.addWidget(self.refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(self.use_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.refresh_button.clicked.connect(self._start_scan)
        self.use_button.clicked.connect(self._use_selected)
        close_button.clicked.connect(self.reject)
        self.table.itemSelectionChanged.connect(self._refresh_selection)

    def _start_scan(self) -> None:
        if self.scan_thread is not None:
            return
        self.refresh_button.setEnabled(False)
        self.use_button.setEnabled(False)
        self.status_label.setText("正在检查已注册的 Python 环境...")
        thread = ExternalScanThread(self.worker_source, self)
        self.scan_thread = thread
        thread.scanned.connect(self._apply_candidates)
        thread.finished.connect(self._scan_finished)
        thread.start()

    def _apply_candidates(self, candidates: list[ExternalPythonCandidate]) -> None:
        self.candidates = list(candidates)
        self.table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            gpu = ", ".join(candidate.gpu_names) if candidate.gpu_available else "CPU"
            status = "可用" if candidate.ready else "; ".join(candidate.errors)
            values = (
                str(candidate.executable),
                candidate.python_version or "未知",
                candidate.architecture or "未知",
                gpu,
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, row)
                self.table.setItem(row, column, item)
        if candidates:
            self.table.selectRow(0)
        ready_count = sum(1 for item in candidates if item.ready)
        self.status_label.setText(
            f"检测到 {len(candidates)} 个解释器，其中 {ready_count} 个通过完整验证。"
        )

    def _scan_finished(self) -> None:
        thread = self.scan_thread
        self.scan_thread = None
        self.refresh_button.setEnabled(True)
        self._refresh_selection()
        if thread is not None:
            thread.deleteLater()

    def _selected_candidate(self) -> ExternalPythonCandidate | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.candidates):
            return None
        return self.candidates[row]

    def _refresh_selection(self) -> None:
        candidate = self._selected_candidate()
        self.use_button.setEnabled(bool(candidate and candidate.ready and self.scan_thread is None))

    def _use_selected(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None or not candidate.ready:
            return
        RuntimeStateStore(self.paths.state_file).save(
            RuntimeSelection(
                runtime_id=None,
                backend_kind="external",
                external_python=str(candidate.executable),
            )
        )
        self.accept()

    def closeEvent(self, event) -> None:
        if self.scan_thread is not None and self.scan_thread.isRunning():
            self.status_label.setText("正在完成当前环境检查，请稍候...")
            event.ignore()
            return
        super().closeEvent(event)


class RuntimeInstallDialog(QDialog):
    def __init__(
        self,
        catalog: RuntimeCatalog,
        paths: MLRuntimePaths | None = None,
        parent=None,
        *,
        capability: str = "模型功能",
    ):
        super().__init__(parent)
        self.catalog = catalog
        self.paths = paths or MLRuntimePaths.from_environment()
        self.installer = RuntimeInstaller(self.paths)
        self.capability = str(capability)
        self.thread: RuntimeInstallThread | None = None
        self.installed_candidate: RuntimeCandidate | None = None
        self.setWindowTitle("安装 YOLO 工具箱运行环境")
        self.setMinimumWidth(620)
        self._build_ui()
        self._select_recommended_profile()
        self._refresh_detail()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel(f"{self.capability}需要机器学习运行环境")
        title.setObjectName("DialogTitle")
        layout.addWidget(title)

        self.profile_combo = QComboBox()
        for artifact in self.catalog.artifacts:
            label = (
                "NVIDIA GPU（CUDA 11.8）"
                if artifact.profile is RuntimeProfile.CUDA118
                else "CPU（兼容模式）"
            )
            self.profile_combo.addItem(label, artifact.runtime_id)
        self.profile_combo.currentIndexChanged.connect(self._refresh_detail)
        layout.addWidget(self.profile_combo)

        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.detail_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("等待选择")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.download_button = QPushButton("下载安装")
        self.offline_button = QPushButton("选择离线包")
        self.advanced_button = QPushButton("高级检测")
        self.cancel_button = QPushButton("取消")
        button_row.addWidget(self.download_button)
        button_row.addWidget(self.offline_button)
        button_row.addWidget(self.advanced_button)
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.download_button.clicked.connect(self._start_download)
        self.offline_button.clicked.connect(self._choose_offline)
        self.advanced_button.clicked.connect(self._show_advanced_notice)
        self.cancel_button.clicked.connect(self._cancel_or_reject)

    def _select_recommended_profile(self) -> None:
        recommended = recommend_runtime_profile()
        for index in range(self.profile_combo.count()):
            artifact = self.catalog.get(str(self.profile_combo.itemData(index)))
            if artifact is not None and artifact.profile is recommended:
                self.profile_combo.setCurrentIndex(index)
                return

    def _current_artifact(self) -> RuntimeArtifact | None:
        return self.catalog.get(str(self.profile_combo.currentData() or ""))

    @staticmethod
    def _format_size(value: int | None) -> str:
        if not value:
            return "构建时确定"
        return f"{value / 1024**2:.1f} MiB"

    def _refresh_detail(self) -> None:
        artifact = self._current_artifact()
        if artifact is None:
            self.detail_label.setText("没有可用运行时配置")
            self.download_button.setEnabled(False)
            return
        profile_name = "NVIDIA GPU" if artifact.profile is RuntimeProfile.CUDA118 else "CPU"
        detail = (
            f"类型：{profile_name}\n"
            f"下载大小：{self._format_size(artifact.archive_size)}\n"
            f"安装位置：{self.paths.runtimes_dir / artifact.runtime_id}"
        )
        if not artifact.is_downloadable:
            detail += "\n当前构建未配置在线运行时地址，请选择官方离线包。"
        self.detail_label.setText(detail)
        self.download_button.setEnabled(artifact.is_downloadable and self.thread is None)

    def _set_running(self, running: bool) -> None:
        self.profile_combo.setEnabled(not running)
        self.offline_button.setEnabled(not running)
        self.advanced_button.setEnabled(not running)
        self.download_button.setEnabled(
            not running
            and self._current_artifact() is not None
            and bool(self._current_artifact().is_downloadable)
        )
        self.cancel_button.setText("停止下载" if running else "取消")
        self.progress_bar.setVisible(running)

    def _start_download(self) -> None:
        artifact = self._current_artifact()
        if artifact is None or not artifact.is_downloadable:
            QMessageBox.information(self, "尚不可下载", "当前版本未配置该运行时的在线下载地址。")
            return
        self._start_thread(RuntimeInstallThread(self.installer, artifact, parent=self))

    def _choose_offline(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择官方运行时包",
            str(Path.home()),
            "运行时包 (*.zip)",
        )
        if not filename:
            return
        self._start_thread(
            RuntimeInstallThread(
                self.installer,
                offline_path=Path(filename),
                catalog=self.catalog,
                parent=self,
            )
        )

    def _start_thread(self, thread: RuntimeInstallThread) -> None:
        if self.thread is not None:
            return
        self.thread = thread
        thread.progress_changed.connect(self._on_progress)
        thread.installed.connect(self._on_installed)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(self._on_thread_finished)
        self._set_running(True)
        self.status_label.setText("正在准备运行环境...")
        thread.start()

    def _on_progress(self, progress: DownloadProgress) -> None:
        total = max(0, int(progress.total))
        completed = max(0, int(progress.completed))
        self.progress_bar.setValue(int(completed * 100 / total) if total else 0)
        stage = "下载" if progress.stage == "downloading" else "安装"
        speed = (
            f"，{progress.bytes_per_second / 1024**2:.1f} MiB/s"
            if progress.bytes_per_second > 0
            else ""
        )
        self.status_label.setText(
            f"{stage}中：{self._format_size(completed)} / {self._format_size(total)}{speed}"
        )

    def _on_installed(self, candidate: RuntimeCandidate) -> None:
        self.installed_candidate = candidate
        self.status_label.setText("运行环境已安装并通过验证")
        self.accept()

    def _on_failed(self, error: BaseException) -> None:
        if isinstance(error, RuntimeInstallError) and error.code == "cancelled":
            self.status_label.setText("下载已停止，有效断点已保留")
            return
        self.status_label.setText(f"安装失败：{error}")
        QMessageBox.critical(self, "运行环境安装失败", str(error))

    def _on_thread_finished(self) -> None:
        thread = self.thread
        self.thread = None
        self._set_running(False)
        if thread is not None:
            thread.deleteLater()

    def _show_advanced_notice(self) -> None:
        dialog = ExternalPythonDialog(self.paths, self)
        if dialog.exec_() == QDialog.Accepted:
            self.accept()

    def _cancel_or_reject(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.request_cancel()
            self.status_label.setText("正在停止下载...")
            self.cancel_button.setEnabled(False)
            return
        self.reject()

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.request_cancel()
            self.status_label.setText("正在停止下载...")
            event.ignore()
            return
        super().closeEvent(event)


class RuntimeManagerDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        catalog: RuntimeCatalog | None = None,
        paths: MLRuntimePaths | None = None,
    ):
        super().__init__(parent)
        self.catalog = catalog or load_runtime_catalog()
        self.paths = paths or MLRuntimePaths.from_environment()
        self.setWindowTitle("运行环境管理")
        self.resize(820, 460)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["类型", "运行时编号", "状态", "下载", "位置"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.install_button = QPushButton("安装或修复")
        self.select_button = QPushButton("设为当前")
        self.delete_button = QPushButton("删除")
        self.advanced_button = QPushButton("高级检测")
        close_button = QPushButton("关闭")
        for button in (
            self.install_button,
            self.select_button,
            self.delete_button,
            self.advanced_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.install_button.clicked.connect(self._install)
        self.select_button.clicked.connect(self._select_current)
        self.delete_button.clicked.connect(self._delete)
        self.advanced_button.clicked.connect(self._advanced)
        close_button.clicked.connect(self.accept)

    def refresh(self) -> None:
        discovery = RuntimeDiscovery(
            self.catalog,
            self.paths,
            RuntimeStateStore(self.paths.state_file),
        )
        candidates = {
            item.artifact.runtime_id: item for item in discovery.find_compatible()
        }
        self.table.setRowCount(len(self.catalog.artifacts))
        for row, artifact in enumerate(self.catalog.artifacts):
            candidate = candidates.get(artifact.runtime_id)
            profile = "GPU" if artifact.profile is RuntimeProfile.CUDA118 else "CPU"
            values = (
                profile,
                artifact.runtime_id,
                "当前使用" if candidate and candidate.selected else "已安装" if candidate else "未安装",
                RuntimeInstallDialog._format_size(artifact.archive_size),
                str(self.paths.runtimes_dir / artifact.runtime_id),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, artifact.runtime_id)
                self.table.setItem(row, column, item)
        if self.table.rowCount():
            self.table.selectRow(0)
        installed = sum(1 for item in candidates.values())
        selection = RuntimeStateStore(self.paths.state_file).load()
        external = "，当前使用外部 Python" if selection.backend_kind == "external" else ""
        self.summary_label.setText(
            f"已安装 {installed} 个运行环境{external}。基础标注和数据集功能不依赖这些环境。"
        )

    def _selected_artifact(self) -> RuntimeArtifact | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return self.catalog.get(str(item.data(Qt.UserRole))) if item else None

    def _install(self) -> None:
        artifact = self._selected_artifact()
        dialog = RuntimeInstallDialog(self.catalog, self.paths, self, capability="运行环境管理")
        if artifact is not None:
            index = dialog.profile_combo.findData(artifact.runtime_id)
            if index >= 0:
                dialog.profile_combo.setCurrentIndex(index)
        dialog.exec_()
        self.refresh()

    def _select_current(self) -> None:
        artifact = self._selected_artifact()
        if artifact is None:
            return
        candidates = RuntimeDiscovery(
            self.catalog,
            self.paths,
            RuntimeStateStore(self.paths.state_file),
        ).find_compatible(artifact.profile)
        candidate = next(
            (item for item in candidates if item.artifact.runtime_id == artifact.runtime_id),
            None,
        )
        if candidate is None:
            QMessageBox.information(self, "尚未安装", "请先安装所选运行环境。")
            return
        RuntimeStateStore(self.paths.state_file).save(
            RuntimeSelection(runtime_id=artifact.runtime_id, backend_kind="managed")
        )
        self.refresh()

    def _delete(self) -> None:
        artifact = self._selected_artifact()
        if artifact is None:
            return
        runtime_dir = self.paths.runtimes_dir / artifact.runtime_id
        if not runtime_dir.is_dir():
            return
        lock_path = self.paths.root_dir / ".locks" / f"{artifact.runtime_id}.lock"
        if lock_path.exists():
            QMessageBox.warning(self, "正在使用", "该运行环境正在安装或修复，暂时不能删除。")
            return
        answer = QMessageBox.question(
            self,
            "删除运行环境",
            f"确定删除 {artifact.runtime_id}？\n不会删除模型、数据集或训练结果。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        shutil.rmtree(runtime_dir)
        selection = RuntimeStateStore(self.paths.state_file).load()
        if selection.runtime_id == artifact.runtime_id:
            RuntimeStateStore(self.paths.state_file).save(RuntimeSelection())
        self.refresh()

    def _advanced(self) -> None:
        ExternalPythonDialog(self.paths, self).exec_()
        self.refresh()


def ensure_ml_runtime(
    parent,
    capability: str,
    resume_callback: Callable[[], None] | None = None,
    *,
    catalog_value: RuntimeCatalog | None = None,
    runtime_paths: MLRuntimePaths | None = None,
    frozen: bool | None = None,
) -> bool:
    """确保模型能力有可用运行时；成功时至多恢复一次原操作。"""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if not is_frozen:
        if resume_callback is not None:
            resume_callback()
        return True

    try:
        catalog = catalog_value or load_runtime_catalog(frozen=True)
        paths = runtime_paths or MLRuntimePaths.from_environment()
        state_store = RuntimeStateStore(paths.state_file)
        selection = state_store.load()
        if selection.backend_kind == "external" and selection.external_python:
            worker_source = external_worker_source_path(frozen=True)
            external = probe_external_python(
                Path(selection.external_python),
                worker_source,
            )
            if external.ready:
                if resume_callback is not None:
                    resume_callback()
                return True
            state_store.save(RuntimeSelection())
        candidates = RuntimeDiscovery(catalog, paths, state_store).find_compatible()
    except Exception as exc:
        QMessageBox.critical(parent, "运行环境配置错误", str(exc))
        return False

    if candidates:
        candidate = next((item for item in candidates if item.selected), None)
        if candidate is None:
            recommended = recommend_runtime_profile()
            candidate = next(
                (item for item in candidates if item.artifact.profile is recommended),
                candidates[0],
            )
            state_store.save(
                RuntimeSelection(
                    runtime_id=candidate.artifact.runtime_id,
                    backend_kind="managed",
                )
            )
        if resume_callback is not None:
            resume_callback()
        return True

    dialog = RuntimeInstallDialog(
        catalog,
        paths,
        parent,
        capability=capability,
    )
    if dialog.exec_() != QDialog.Accepted:
        return False
    if resume_callback is not None:
        resume_callback()
    return True


__all__ = [
    "RuntimeInstallDialog",
    "RuntimeManagerDialog",
    "ExternalPythonDialog",
    "ensure_ml_runtime",
    "external_worker_source_path",
    "load_runtime_catalog",
    "recommend_runtime_profile",
    "runtime_status_summary",
]

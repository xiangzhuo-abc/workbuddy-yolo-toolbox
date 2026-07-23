"""源码与冻结版本共用的程序、状态和工作区路径。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


APP_STATE_DIR_NAME = "WorkBuddyYoloTool"
DEFAULT_WORKSPACE_NAME = "WorkBuddy YOLO Workspace"
STATE_DIR_ENV = "WORKBUDDY_STATE_DIR"
WORKSPACE_DIR_ENV = "WORKBUDDY_WORKSPACE_DIR"
ML_RUNTIME_DIR_NAME = "YOLOToolbox"
ML_RUNTIME_DIR_ENV = "YOLO_TOOLBOX_RUNTIME_DIR"


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_state_root(environ: Mapping[str, str]) -> Path:
    local_app_data = str(environ.get("LOCALAPPDATA", "")).strip()
    if local_app_data:
        return Path(local_app_data)
    user_profile = str(environ.get("USERPROFILE", "")).strip()
    if user_profile:
        return Path(user_profile) / "AppData" / "Local"
    return Path.home() / ".local" / "share"


def _default_documents_dir(environ: Mapping[str, str]) -> Path:
    user_profile = str(environ.get("USERPROFILE", "")).strip()
    return (Path(user_profile) if user_profile else Path.home()) / "Documents"


@dataclass(frozen=True)
class MLRuntimePaths:
    """机器学习运行时专用目录，不与应用配置或用户工作区混用。"""

    root_dir: Path
    runtimes_dir: Path
    cache_dir: Path
    staging_dir: Path
    state_file: Path

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "MLRuntimePaths":
        values = os.environ if environ is None else environ
        override = str(values.get(ML_RUNTIME_DIR_ENV, "")).strip()
        root = (
            Path(override)
            if override
            else _default_state_root(values) / ML_RUNTIME_DIR_NAME
        )
        return cls(
            root_dir=root,
            runtimes_dir=root / "runtimes",
            cache_dir=root / "cache" / "runtimes",
            staging_dir=root / "staging" / "runtimes",
            state_file=root / "config" / "runtime_state.json",
        )


@dataclass(frozen=True)
class RuntimePaths:
    """区分只读程序资源、用户状态和用户工作区。"""

    frozen: bool
    install_dir: Path
    resource_dir: Path
    state_dir: Path
    workspace_dir: Path
    config_file: Path
    logs_dir: Path
    cache_dir: Path
    crash_reports_dir: Path
    tensorboard_logdirs: Path
    dataset_dir: Path
    models_dir: Path
    runs_dir: Path
    backups_dir: Path
    legacy_config_file: Path
    worker_executable: Path

    @classmethod
    def from_environment(
        cls,
        workspace_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
        install_dir: Path | None = None,
        frozen: bool | None = None,
    ) -> "RuntimePaths":
        values = os.environ if environ is None else environ
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen

        if install_dir is None:
            install = Path(sys.executable).resolve().parent if is_frozen else _source_root()
        else:
            install = Path(install_dir)

        if is_frozen:
            bundle_dir = getattr(sys, "_MEIPASS", None)
            resource = Path(bundle_dir) if bundle_dir else install / "_internal"
        else:
            resource = install

        state_override = str(values.get(STATE_DIR_ENV, "")).strip()
        state = (
            Path(state_override)
            if state_override
            else _default_state_root(values) / APP_STATE_DIR_NAME
        )

        workspace_override = str(values.get(WORKSPACE_DIR_ENV, "")).strip()
        if workspace_dir is not None:
            workspace = Path(workspace_dir)
        elif workspace_override:
            workspace = Path(workspace_override)
        else:
            workspace = _default_documents_dir(values) / DEFAULT_WORKSPACE_NAME

        dataset = workspace / "dataset"
        return cls(
            frozen=is_frozen,
            install_dir=install,
            resource_dir=resource,
            state_dir=state,
            workspace_dir=workspace,
            config_file=state / "config" / "tool_config.json",
            logs_dir=state / "logs",
            cache_dir=state / "cache",
            crash_reports_dir=state / "crash_reports",
            tensorboard_logdirs=state / "tensorboard_logdirs",
            dataset_dir=dataset,
            models_dir=workspace / "models",
            runs_dir=workspace / "runs",
            backups_dir=dataset / "backups",
            legacy_config_file=install / "config" / "tool_config.json",
            worker_executable=install / "YOLO工具箱Worker.exe",
        )

"""生成不包含用户数据的诊断包，并持久化未处理异常报告。"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import traceback as traceback_module
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Callable
from uuid import uuid4

from .runtime_paths import RuntimePaths
from .version import DISPLAY_VERSION, LICENSE_ID, PRODUCT_NAME


DIAGNOSTIC_SCHEMA_VERSION = 1
MAX_LOG_FILES = 5
MAX_LOG_BYTES = 256 * 1024
PACKAGE_NAMES = (
    "PyQt5",
    "ultralytics",
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "Pillow",
    "PyYAML",
    "tensorboard",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "未安装"
    return versions


def _redaction_pairs(runtime_paths: RuntimePaths) -> list[tuple[str, str]]:
    pairs = [
        (str(runtime_paths.workspace_dir), "%WORKSPACE%"),
        (str(runtime_paths.state_dir), "%STATE%"),
        (str(runtime_paths.install_dir), "%INSTALL%"),
        (str(Path.home()), "%USERPROFILE%"),
    ]
    unique = {(source, token) for source, token in pairs if source}
    return sorted(unique, key=lambda item: len(item[0]), reverse=True)


def _redact_text(value: object, runtime_paths: RuntimePaths) -> str:
    text = str(value)
    for source, token in _redaction_pairs(runtime_paths):
        text = text.replace(source, token)
        text = text.replace(source.replace("\\", "/"), token)
    return text


def _nearest_existing(path: Path) -> Path | None:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    return candidate


def _path_status(path: Path, runtime_paths: RuntimePaths) -> dict[str, object]:
    candidate = Path(path)
    status: dict[str, object] = {
        "path": _redact_text(candidate, runtime_paths),
        "exists": False,
        "is_dir": False,
        "is_file": False,
        "writable_parent": False,
        "free_bytes": None,
    }
    try:
        status["exists"] = candidate.exists()
        status["is_dir"] = candidate.is_dir()
        status["is_file"] = candidate.is_file()
        anchor = _nearest_existing(candidate if candidate.is_dir() else candidate.parent)
        if anchor is not None:
            status["writable_parent"] = os.access(anchor, os.W_OK)
            status["free_bytes"] = shutil.disk_usage(anchor).free
    except OSError as exc:
        status["error"] = str(exc)
    return status


def _diagnostic_report(runtime_paths: RuntimePaths) -> dict[str, object]:
    path_values = {
        "install_dir": runtime_paths.install_dir,
        "resource_dir": runtime_paths.resource_dir,
        "state_dir": runtime_paths.state_dir,
        "workspace_dir": runtime_paths.workspace_dir,
        "config_file": runtime_paths.config_file,
        "logs_dir": runtime_paths.logs_dir,
        "dataset_dir": runtime_paths.dataset_dir,
        "models_dir": runtime_paths.models_dir,
        "runs_dir": runtime_paths.runs_dir,
    }
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product": {
            "name": PRODUCT_NAME,
            "version": DISPLAY_VERSION,
            "license": LICENSE_ID,
        },
        "system": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "frozen": runtime_paths.frozen,
            "executable": _redact_text(sys.executable, runtime_paths),
        },
        "packages": _package_versions(),
        "paths": {
            name: _path_status(path, runtime_paths)
            for name, path in path_values.items()
        },
    }


def _read_log_tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - MAX_LOG_BYTES), os.SEEK_SET)
        data = handle.read(MAX_LOG_BYTES)
    return data.decode("utf-8", errors="replace")


def _log_tails(runtime_paths: RuntimePaths) -> list[tuple[str, str]]:
    logs_dir = runtime_paths.logs_dir
    if not logs_dir.is_dir():
        return []
    try:
        root = logs_dir.resolve()
    except OSError:
        return []

    candidates: list[Path] = []
    for path in logs_dir.rglob("*.log"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                continue
            candidates.append(path)
        except OSError:
            continue
    candidates.sort(
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )

    tails: list[tuple[str, str]] = []
    for path in candidates[:MAX_LOG_FILES]:
        try:
            relative = path.relative_to(logs_dir).as_posix().replace("/", "__")
            text = _redact_text(_read_log_tail(path), runtime_paths)
            tails.append((f"logs/{relative}.tail.txt", text))
        except OSError:
            continue
    return tails


def build_diagnostic_archive(
    output_path: Path,
    *,
    runtime_paths: RuntimePaths,
    include_user_data: bool = False,
) -> Path:
    """生成诊断 ZIP；第一版固定拒绝收集任何用户数据。"""
    if include_user_data:
        raise ValueError("诊断包不允许包含用户数据")

    output = Path(output_path)
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    report = _diagnostic_report(runtime_paths)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                "diagnostics.json",
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            )
            for name, text in _log_tails(runtime_paths):
                archive.writestr(name, text)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def write_crash_report(
    runtime_paths: RuntimePaths,
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback_obj: TracebackType | None,
) -> Path:
    """将未处理异常写入状态目录，不采集局部变量或用户文件。"""
    runtime_paths.crash_reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = runtime_paths.crash_reports_dir / f"crash-{timestamp}-{uuid4().hex[:8]}.txt"
    temporary = output.with_name(f".{output.name}.tmp")
    trace = "".join(
        traceback_module.format_exception(exc_type, exc_value, traceback_obj)
    )
    content = "\n".join([
        f"product: {PRODUCT_NAME}",
        f"version: {DISPLAY_VERSION}",
        f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
        "",
        _redact_text(trace, runtime_paths),
    ])
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def install_global_exception_hook(
    runtime_paths: RuntimePaths,
) -> Callable[[type[BaseException], BaseException, TracebackType | None], object]:
    """安装全局异常钩子并返回之前的钩子，便于测试或恢复。"""
    previous = sys.excepthook

    def handle_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback_obj: TracebackType | None,
    ) -> None:
        try:
            write_crash_report(runtime_paths, exc_type, exc_value, traceback_obj)
        except BaseException:
            pass
        previous(exc_type, exc_value, traceback_obj)

    sys.excepthook = handle_exception
    return previous


__all__ = [
    "build_diagnostic_archive",
    "install_global_exception_hook",
    "write_crash_report",
]

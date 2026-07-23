"""按用户请求检测明确注册的外部 Python 模型环境。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .worker_commands import WorkerBackend


WORKER_PROTOCOL = 1
REQUIRED_MODULES = ("torch", "torchvision", "ultralytics", "tensorboard")
_PYTHON_PATH_PATTERN = re.compile(
    r"([A-Za-z]:[\\/].*?python(?:w)?\.exe)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExternalPythonCandidate:
    executable: Path
    python_version: str
    architecture: str
    worker_protocol: int
    versions: Mapping[str, str | None]
    cuda_version: str | None
    gpu_available: bool
    gpu_names: tuple[str, ...]
    ready: bool
    errors: tuple[str, ...]


def _normalise_existing_python(value: str | Path) -> Path | None:
    path = Path(str(value).strip().strip('"'))
    if path.name.lower() not in {"python.exe", "pythonw.exe"} or not path.is_file():
        return None
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def discover_external_pythons(
    runner: Callable[..., object] = subprocess.run,
) -> list[Path]:
    """仅查询 Windows Python Launcher 与 PATH，不遍历文件系统。"""
    values: list[Path] = []
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = runner(
            ["py", "-0p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=creation_flags,
            check=False,
        )
        if int(getattr(result, "returncode", 1)) == 0:
            for line in str(getattr(result, "stdout", "")).splitlines():
                match = _PYTHON_PATH_PATTERN.search(line.strip())
                if match:
                    path = _normalise_existing_python(match.group(1))
                    if path is not None:
                        values.append(path)
    except (OSError, subprocess.TimeoutExpired):
        pass

    for command in ("python", "python3"):
        located = shutil.which(command)
        if located:
            path = _normalise_existing_python(located)
            if path is not None:
                values.append(path)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in values:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def probe_external_python(
    executable: Path,
    worker_source: Path,
    *,
    timeout: int = 30,
    runner: Callable[..., object] = subprocess.run,
) -> ExternalPythonCandidate:
    python_path = Path(executable)
    source_path = Path(worker_source)
    early_errors = []
    if not python_path.is_file():
        early_errors.append(f"Python 不存在: {python_path}")
    if not source_path.is_file():
        early_errors.append(f"Worker 源码不存在: {source_path}")
    if early_errors:
        return ExternalPythonCandidate(
            python_path,
            "",
            "",
            0,
            {},
            None,
            False,
            (),
            False,
            tuple(early_errors),
        )

    runtime_id = f"external:{python_path}"
    command = [
        str(python_path),
        str(source_path),
        "probe",
        "--runtime-id",
        runtime_id,
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout)),
            creationflags=creation_flags,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ExternalPythonCandidate(
            python_path,
            "",
            "",
            0,
            {},
            None,
            False,
            (),
            False,
            (f"探针启动失败: {exc}",),
        )

    lines = [
        line.strip()
        for line in str(getattr(result, "stdout", "")).splitlines()
        if line.strip()
    ]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError as exc:
        payload = {}
        parse_errors = [f"探针返回无效 JSON: {exc}"]
    else:
        parse_errors = []

    versions_value = payload.get("versions", {})
    versions = dict(versions_value) if isinstance(versions_value, Mapping) else {}
    errors = [str(item) for item in payload.get("errors", [])] + parse_errors
    architecture = str(payload.get("architecture", ""))
    if architecture.lower() not in {"amd64", "x86_64"}:
        errors.append(f"外部 Python 必须为 x64，当前架构: {architecture or '未知'}")
    try:
        protocol = int(payload.get("worker_protocol", 0))
    except (TypeError, ValueError):
        protocol = 0
    if protocol != WORKER_PROTOCOL:
        errors.append(f"Worker 协议不兼容: {protocol}")
    if str(payload.get("runtime_id", "")) != runtime_id:
        errors.append("外部探针身份不匹配")
    for name in REQUIRED_MODULES:
        if not versions.get(name):
            message = f"缺少运行依赖: {name}"
            if not any(name in error for error in errors):
                errors.append(message)

    return_code = int(getattr(result, "returncode", 1))
    ready = bool(payload.get("ok")) and return_code == 0 and not errors
    return ExternalPythonCandidate(
        executable=python_path,
        python_version=str(payload.get("python_version", "")),
        architecture=architecture,
        worker_protocol=protocol,
        versions=versions,
        cuda_version=(
            None
            if payload.get("cuda_version") is None
            else str(payload.get("cuda_version"))
        ),
        gpu_available=bool(payload.get("gpu_available")),
        gpu_names=tuple(str(item) for item in payload.get("gpu_names", [])),
        ready=ready,
        errors=tuple(dict.fromkeys(errors)),
    )


def build_external_worker_backend(
    candidate: ExternalPythonCandidate,
    worker_source: Path,
) -> WorkerBackend:
    if not candidate.ready:
        raise ValueError("外部 Python 环境未通过验证")
    source = Path(worker_source)
    return WorkerBackend(
        kind="external",
        program=candidate.executable,
        prefix_args=(str(source),),
        runtime_id=f"external:{candidate.executable}",
    )


__all__ = [
    "ExternalPythonCandidate",
    "build_external_worker_backend",
    "discover_external_pythons",
    "probe_external_python",
]

"""构造源码模式和冻结模式共用的 Worker 启动命令。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .ml_runtime import (
    SUPPORTED_WORKER_PROTOCOLS,
    RuntimeSelection,
    RuntimeStateStore,
    is_valid_runtime_id,
)
from .runtime_paths import MLRuntimePaths, RuntimePaths


WORKER_SCRIPTS = {
    "train": "yolo_train_worker.py",
    "detect": "yolo_detect_worker.py",
    "evaluate": "yolo_evaluate_worker.py",
    "tensorboard": "launch_tensorboard.py",
}


class RuntimeUnavailableError(RuntimeError):
    """模型任务没有可用 Worker 后端。"""


@dataclass(frozen=True)
class WorkerBackend:
    kind: str
    program: Path
    prefix_args: tuple[str, ...]
    runtime_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"source", "managed", "external"}:
            raise ValueError(f"未知 Worker 后端类型: {self.kind}")


def _normalise_kind(kind: str) -> str:
    value = str(kind or "").strip().lower()
    if value not in WORKER_SCRIPTS:
        allowed = ", ".join(WORKER_SCRIPTS)
        raise ValueError(f"未知 Worker 类型: {kind!r}，可用类型: {allowed}")
    return value


def _read_runtime_identity(runtime_dir: Path) -> tuple[str, int]:
    manifest_path = runtime_dir / "runtime.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime_id = str(data.get("runtime_id", "")).strip()
        protocol = int(data.get("worker_protocol", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeUnavailableError(f"运行环境清单不可用: {exc}") from exc
    return runtime_id, protocol


def resolve_worker_backend(
    *,
    frozen: bool | None = None,
    runtime_paths: MLRuntimePaths | None = None,
    selection: RuntimeSelection | None = None,
    resource_dir: Path | None = None,
) -> WorkerBackend:
    """解析源码、托管运行时或外部 Python Worker 后端。"""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if not is_frozen:
        return WorkerBackend("source", Path(sys.executable), (), "source")

    paths = runtime_paths or MLRuntimePaths.from_environment()
    chosen = selection or RuntimeStateStore(paths.state_file).load()
    if chosen.backend_kind == "external":
        python_path = Path(chosen.external_python or "")
        resource = (
            Path(resource_dir)
            if resource_dir is not None
            else RuntimePaths.from_environment(frozen=True).resource_dir
        )
        worker_source = resource / "worker_source" / "yolo_worker_entry.py"
        if not python_path.is_file():
            raise RuntimeUnavailableError(f"外部 Python 不存在: {python_path}")
        if not worker_source.is_file():
            raise RuntimeUnavailableError(f"外部 Worker 源码不存在: {worker_source}")
        return WorkerBackend(
            "external",
            python_path,
            (str(worker_source),),
            f"external:{python_path}",
        )

    runtime_id = str(chosen.runtime_id or "").strip()
    if not runtime_id:
        raise RuntimeUnavailableError("模型运行环境尚未安装或选择")
    if not is_valid_runtime_id(runtime_id):
        raise RuntimeUnavailableError("模型运行环境编号无效")
    runtime_dir = paths.runtimes_dir / runtime_id
    manifest_id, protocol = _read_runtime_identity(runtime_dir)
    if manifest_id != runtime_id:
        raise RuntimeUnavailableError("模型运行环境清单与目录不一致")
    if protocol not in SUPPORTED_WORKER_PROTOCOLS:
        raise RuntimeUnavailableError(f"Worker 协议不兼容: {protocol}")
    worker = runtime_dir / "YOLO工具箱Worker.exe"
    if not worker.is_file():
        raise RuntimeUnavailableError(f"模型 Worker 不存在: {worker}")
    return WorkerBackend("managed", worker, (), runtime_id)


def build_worker_command(
    kind: str,
    argv: Sequence[str],
    *,
    frozen: bool | None = None,
    resource_dir: Path | None = None,
    backend: WorkerBackend | None = None,
) -> tuple[str, list[str]]:
    """返回 QProcess 可直接使用的程序路径和参数列表。"""
    worker_kind = _normalise_kind(kind)
    arguments = [str(value) for value in argv]
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)

    if is_frozen:
        actual_backend = backend or resolve_worker_backend(
            frozen=True,
            resource_dir=resource_dir,
        )
        return str(actual_backend.program), [
            *actual_backend.prefix_args,
            worker_kind,
            *arguments,
        ]

    resource = (
        Path(resource_dir)
        if resource_dir is not None
        else RuntimePaths.from_environment(frozen=False).resource_dir
    )
    script = resource / "tools" / WORKER_SCRIPTS[worker_kind]
    return sys.executable, [str(script), *arguments]


__all__ = [
    "RuntimeUnavailableError",
    "WORKER_SCRIPTS",
    "WorkerBackend",
    "build_worker_command",
    "resolve_worker_backend",
]

"""冻结版 Worker 的轻量分发入口，不导入 PyQt。"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from multiprocessing import freeze_support
from typing import Sequence, TextIO


WORKER_MODULES = {
    "train": "yolo_train_worker",
    "detect": "yolo_detect_worker",
    "evaluate": "yolo_evaluate_worker",
    "tensorboard": "launch_tensorboard",
}
WORKER_ERROR_PREFIX = "__YOLO_WORKER_ERROR__"
WORKER_PROTOCOL = 1


def _error_payload(kind: str, message: str) -> dict[str, str]:
    return {"kind": str(kind), "error": str(message)}


def _write_error(kind: str, message: str, stream: TextIO | None = None) -> None:
    output = stream if stream is not None else sys.stdout
    output.write(
        WORKER_ERROR_PREFIX
        + json.dumps(_error_payload(kind, message), ensure_ascii=False)
        + "\n"
    )
    output.flush()


def collect_runtime_probe(runtime_id: str = "") -> dict[str, object]:
    """实际导入模型依赖并返回可供主程序校验的运行时信息。"""
    modules: dict[str, object] = {}
    versions: dict[str, str | None] = {}
    errors: list[str] = []
    for name in ("torch", "torchvision", "ultralytics", "tensorboard"):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            versions[name] = str(getattr(module, "__version__", "未知"))
        except BaseException as exc:
            versions[name] = None
            errors.append(f"{name} 导入失败: {exc}")

    cuda_version = None
    gpu_available = False
    gpu_names: list[str] = []
    torch_module = modules.get("torch")
    if torch_module is not None:
        try:
            cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
            cuda_api = getattr(torch_module, "cuda")
            gpu_available = bool(cuda_api.is_available())
            if gpu_available:
                gpu_names = [
                    str(cuda_api.get_device_name(index))
                    for index in range(int(cuda_api.device_count()))
                ]
        except BaseException as exc:
            errors.append(f"CUDA 状态检查失败: {exc}")

    return {
        "ok": not errors,
        "worker_protocol": WORKER_PROTOCOL,
        "runtime_id": str(runtime_id or ""),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "versions": versions,
        "cuda_version": None if cuda_version is None else str(cuda_version),
        "gpu_available": gpu_available,
        "gpu_names": gpu_names,
        "errors": errors,
    }


def run_probe(
    argv: Sequence[str] | None = None,
    *,
    stream: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="检查 YOLO 工具箱 Worker 运行环境")
    parser.add_argument("--runtime-id", default="")
    args = parser.parse_args(list(argv or ()))
    payload = collect_runtime_probe(args.runtime_id)
    output = stream if stream is not None else sys.stdout
    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    output.flush()
    return 0 if payload["ok"] else 1


def main(argv: Sequence[str] | None = None, *, stream: TextIO | None = None) -> int:
    """分发第一个子命令，剩余参数原样交给现有 Worker。"""
    freeze_support()
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        _write_error(
            "",
            "缺少 Worker 类型，可用类型: train, detect, evaluate, tensorboard, probe",
            stream,
        )
        return 2

    kind = str(values[0]).strip().lower()
    if kind == "probe":
        return run_probe(values[1:], stream=stream)
    module_name = WORKER_MODULES.get(kind)
    if module_name is None:
        _write_error(kind, f"未知 Worker 类型: {values[0]}", stream)
        return 2

    try:
        module = importlib.import_module(module_name)
        worker_main = getattr(module, "main")
        return int(worker_main(values[1:]) or 0)
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:
        _write_error(kind, f"Worker 启动失败: {exc}", stream)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

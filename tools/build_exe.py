"""构建 Windows onedir EXE，并在输出前后执行发布物隔离检查。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .core.ml_runtime import RuntimeCatalog, RuntimeProfile
except ImportError:
    from core.ml_runtime import RuntimeCatalog, RuntimeProfile

PROJECT_DIR = Path(__file__).resolve().parent.parent
_VERSION_SPEC = importlib.util.spec_from_file_location(
    "workbuddy_release_version",
    PROJECT_DIR / "tools" / "core" / "version.py",
)
if _VERSION_SPEC is None or _VERSION_SPEC.loader is None:
    raise ImportError("无法加载发布版本模块")
_VERSION_MODULE = importlib.util.module_from_spec(_VERSION_SPEC)
_VERSION_SPEC.loader.exec_module(_VERSION_MODULE)
DISPLAY_VERSION = _VERSION_MODULE.DISPLAY_VERSION
FILE_VERSION = _VERSION_MODULE.FILE_VERSION
LICENSE_ID = _VERSION_MODULE.LICENSE_ID
PRODUCT_NAME = _VERSION_MODULE.PRODUCT_NAME
PUBLISHER_NAME = _VERSION_MODULE.PUBLISHER_NAME
DEFAULT_OUT_DIR = PROJECT_DIR / "tmp" / "exe-candidate"
GUI_EXE_NAME = "YOLO工具箱.exe"
ENTRY_SCRIPT_NAMES = frozenset({"yolo_tool_launcher", "yolo_worker_entry"})
REQUIRED_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CHANGELOG.md",
    "BUILD_INFO.json",
)
RUNTIME_DIR_NAMES = {
    ".git",
    ".workbuddy",
    "dataset",
    "debug",
    "logs",
    "models",
    "release",
    "runs",
    "tmp",
    "tests",
}
SUPPORTED_BUILD_PYTHON = {(3, 12), (3, 13)}
MIN_FREE_BUILD_BYTES = 12 * 1024**3
MAX_BASE_BYTES = 300 * 1024**2
RUNTIME_CATALOG_ENV = "YOLO_TOOLBOX_RUNTIME_CATALOG_BUILD"
DEFAULT_RUNTIME_CATALOG = PROJECT_DIR / "tmp" / "runtime-candidate" / "runtime_catalog.json"
RUNTIME_IMPORTS = (
    "PyQt5",
    "cv2",
    "numpy",
    "PIL",
    "yaml",
)
FORBIDDEN_ML_PARTS = {
    "torch",
    "torchvision",
    "ultralytics",
    "tensorboard",
    "polars",
    "nvidia",
}
FORBIDDEN_ML_FILES = {
    "_polars_runtime.pyd",
    "torch_cpu.dll",
    "torch_cuda.dll",
}
FORBIDDEN_ML_PREFIXES = (
    "cublas",
    "cudnn",
    "cufft",
    "cusolver",
    "cusparse",
)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "未安装"


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (path for path in root.rglob("*") if path.is_file())


def _has_file_named(root: Path, name: str) -> bool:
    return any(path.name.lower() == name.lower() for path in _iter_files(root))


def select_entry_scripts(
    scripts: Sequence[tuple[str, str, str]],
    entry_name: str,
) -> list[tuple[str, str, str]]:
    """保留全部运行时 hook，并仅选择一个业务入口脚本。"""
    if entry_name not in ENTRY_SCRIPT_NAMES:
        raise ValueError(f"未知 EXE 入口: {entry_name}")
    selected = [
        item
        for item in scripts
        if item[0] not in ENTRY_SCRIPT_NAMES or item[0] == entry_name
    ]
    if not any(item[0] == entry_name for item in selected):
        raise ValueError(f"Analysis 中缺少 EXE 入口: {entry_name}")
    return selected


def check_build_disk_space(
    path: Path,
    *,
    free_bytes: int | None = None,
) -> str | None:
    """检查构建目录所在磁盘是否有足够空间容纳临时文件和产物。"""
    try:
        available = (
            shutil.disk_usage(Path(path)).free
            if free_bytes is None
            else int(free_bytes)
        )
    except OSError as exc:
        return f"无法检查构建磁盘空间: {exc}"
    if available < MIN_FREE_BUILD_BYTES:
        minimum_gib = MIN_FREE_BUILD_BYTES / 1024**3
        available_gib = available / 1024**3
        return (
            f"构建磁盘空间不足: 至少需要 {minimum_gib:.1f} GiB，"
            f"当前可用 {available_gib:.1f} GiB"
        )
    return None


def check_exe_manifest(path: Path) -> list[str]:
    """检查轻量基础 EXE，不允许机器学习运行时混入。"""
    root = Path(path)
    errors: list[str] = []
    if not root.is_dir():
        return [f"EXE 输出目录不存在: {root}"]

    if not (root / GUI_EXE_NAME).is_file():
        errors.append(f"缺少主程序: {GUI_EXE_NAME}")
    if (root / "YOLO工具箱Worker.exe").exists():
        errors.append("禁止基础程序包含 Worker: YOLO工具箱Worker.exe")
    internal = root / "_internal"
    if not internal.is_dir():
        errors.append("缺少共享依赖目录: _internal")

    for required in REQUIRED_FILES:
        if not (root / required).is_file():
            label = "许可证" if required == "LICENSE" else (
                "第三方声明" if required == "THIRD_PARTY_NOTICES.md" else (
                    "更新日志" if required == "CHANGELOG.md" else "构建元数据"
                )
            )
            errors.append(f"缺少{label}: {required}")

    if not _has_file_named(internal / "PyQt5", "qwindows.dll"):
        errors.append("缺少 Qt Windows 平台插件: qwindows.dll")
    catalog_path = internal / "runtime_catalog.json"
    if not catalog_path.is_file():
        errors.append("缺少可信运行时清单: runtime_catalog.json")
    else:
        try:
            catalog = RuntimeCatalog.from_file(catalog_path)
            profiles = {artifact.profile for artifact in catalog.artifacts}
            if profiles != {RuntimeProfile.CPU, RuntimeProfile.CUDA118}:
                errors.append("可信运行时清单必须同时包含 CPU 和 CUDA 11.8")
            if any(not artifact.is_downloadable for artifact in catalog.artifacts):
                errors.append("可信运行时清单包含不可下载的运行时项")
        except ValueError as exc:
            errors.append(f"可信运行时清单无效: {exc}")

    total_size = 0
    for file_path in _iter_files(root):
        try:
            total_size += file_path.stat().st_size
        except OSError:
            pass
        relative = file_path.relative_to(root)
        parts = {part.lower() for part in relative.parts}
        lower_name = file_path.name.lower()
        if file_path.suffix.lower() in {".pt", ".onnx", ".engine"}:
            errors.append(f"禁止发布物包含模型文件: {relative.as_posix()}")
        elif relative.parts and relative.parts[0].lower() in {
            name.lower() for name in RUNTIME_DIR_NAMES
        }:
            errors.append(f"禁止发布物包含用户运行目录: {relative.as_posix()}")
        if (
            parts & FORBIDDEN_ML_PARTS
            or lower_name in FORBIDDEN_ML_FILES
            or lower_name.endswith(".dll")
            and lower_name.startswith(FORBIDDEN_ML_PREFIXES)
        ):
            errors.append(f"禁止基础程序包含机器学习运行时: {relative.as_posix()}")

    if total_size > MAX_BASE_BYTES:
        errors.append(
            f"基础程序体积超过 {MAX_BASE_BYTES / 1024**2:.0f} MiB: "
            f"{total_size / 1024**2:.1f} MiB"
        )

    return sorted(set(errors))


def build_info() -> dict[str, object]:
    """生成不包含本机绝对路径的可复核构建信息。"""
    return {
        "product": PRODUCT_NAME,
        "version": DISPLAY_VERSION,
        "license": LICENSE_ID,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in (
                "PyInstaller",
                "pyinstaller-hooks-contrib",
                "PyQt5",
                "opencv-python",
                "numpy",
                "Pillow",
                "PyYAML",
            )
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _copy_release_metadata(output_dir: Path) -> None:
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"):
        shutil.copy2(PROJECT_DIR / name, output_dir / name)
    (output_dir / "BUILD_INFO.json").write_text(
        json.dumps(build_info(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _check_build_environment() -> list[str]:
    errors: list[str] = []
    disk_error = check_build_disk_space(PROJECT_DIR)
    if disk_error:
        errors.append(disk_error)
    if sys.version_info[:2] not in SUPPORTED_BUILD_PYTHON:
        errors.append(
            "构建环境要求 Python 3.12/3.13 x64，"
            f"当前为 {platform.python_version()}"
        )
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        errors.append(f"构建环境要求 x64，当前架构为 {platform.machine()}")
    if importlib.util.find_spec("PyInstaller") is None:
        errors.append("未找到 PyInstaller，请先安装 requirements-build.txt")
    for module_name in RUNTIME_IMPORTS:
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"运行依赖不可导入: {module_name}")
    return errors


def _run_pyinstaller(
    out_dir: Path,
    clean: bool,
    runtime_catalog: Path,
) -> int:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(out_dir),
        "--workpath",
        str(PROJECT_DIR / "tmp" / "pyinstaller-build"),
    ]
    if clean:
        command.append("--clean")
    command.append(str(PROJECT_DIR / "packaging" / "workbuddy_yolo.spec"))
    environment = os.environ.copy()
    environment[RUNTIME_CATALOG_ENV] = str(runtime_catalog)
    return subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        env=environment,
    ).returncode


def _built_output_dir(out_dir: Path) -> Path:
    """返回 spec 的 COLLECT 实际目录，同时兼容直接输出到 out_dir 的产物。"""
    nested = out_dir / "YOLO数据标注工具箱"
    if (nested / GUI_EXE_NAME).is_file():
        return nested
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 YOLO 工具箱 Windows onedir EXE")
    parser.add_argument("--clean", action="store_true", help="清理 PyInstaller 临时构建目录")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="EXE 输出目录")
    parser.add_argument("--dry-run", action="store_true", help="仅显示构建环境和清单，不构建")
    parser.add_argument("--python-version", default=None, help="仅用于记录/校验的 Python 版本")
    parser.add_argument(
        "--runtime-catalog",
        default=str(DEFAULT_RUNTIME_CATALOG),
        help="已经包含 CPU/GPU 实际大小、SHA256 和 HTTPS 地址的可信运行时清单",
    )
    args = parser.parse_args(argv)

    errors = _check_build_environment()
    runtime_catalog = Path(args.runtime_catalog).resolve()
    try:
        catalog = RuntimeCatalog.from_file(runtime_catalog)
        profiles = {artifact.profile for artifact in catalog.artifacts}
        if profiles != {RuntimeProfile.CPU, RuntimeProfile.CUDA118}:
            errors.append("运行时清单必须同时包含 CPU 和 CUDA 11.8")
        if any(not artifact.is_downloadable for artifact in catalog.artifacts):
            errors.append("运行时清单存在不可下载项")
    except ValueError as exc:
        errors.append(f"运行时清单不可用: {exc}")
    if args.python_version and args.python_version not in {"3.12", "3.13"}:
        errors.append(f"--python-version 必须为 3.12 或 3.13，当前为 {args.python_version}")
    print(f"产品: {PRODUCT_NAME} {DISPLAY_VERSION}")
    print(f"目标: Windows x64 onedir，输出目录: {args.out_dir}")
    for name, value in build_info()["packages"].items():
        print(f"依赖: {name} {value}")
    if errors:
        for error in errors:
            print(f"[错误] {error}")
        return 2
    if args.dry_run:
        print("[通过] 构建环境满足要求；未生成 EXE。")
        return 0

    out_dir = Path(args.out_dir).resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return_code = _run_pyinstaller(
        out_dir,
        clean=args.clean,
        runtime_catalog=runtime_catalog,
    )
    if return_code != 0:
        return return_code

    built_dir = _built_output_dir(out_dir)
    _copy_release_metadata(built_dir)
    errors = check_exe_manifest(built_dir)
    if errors:
        for error in errors:
            print(f"[错误] {error}")
        return 3
    print(f"[通过] EXE 清单检查通过: {built_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

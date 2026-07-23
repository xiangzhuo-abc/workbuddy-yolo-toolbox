"""构建、校验并打包按需下载的 YOLO 机器学习运行时。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from dataclasses import replace
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

try:
    from .build_exe import check_build_disk_space
    from .core.ml_runtime import (
        SUPPORTED_WORKER_PROTOCOLS,
        WORKER_EXE_NAME,
        RuntimeArtifact,
        RuntimeCatalog,
        RuntimeProfile,
    )
except ImportError:
    from build_exe import check_build_disk_space
    from core.ml_runtime import (
        SUPPORTED_WORKER_PROTOCOLS,
        WORKER_EXE_NAME,
        RuntimeArtifact,
        RuntimeCatalog,
        RuntimeProfile,
    )


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES = PROJECT_DIR / "packaging" / "runtime_profiles.json"
DEFAULT_OUT_DIR = PROJECT_DIR / "tmp" / "runtime-candidate"
DEFAULT_BUILD_DIR = PROJECT_DIR / "tmp" / "runtime-pyinstaller"
RUNTIME_SPEC = PROJECT_DIR / "packaging" / "workbuddy_yolo_runtime.spec"
RUNTIME_PROFILE_ENV = "YOLO_RUNTIME_PROFILE"
RUNTIME_ID_ENV = "YOLO_RUNTIME_ID"
GUI_EXE_NAME = "YOLO工具箱.exe"
SUPPORTED_BUILD_PYTHON = {(3, 12), (3, 13)}
REQUIRED_FILES = ("runtime.json", "LICENSE", "THIRD_PARTY_NOTICES.md")
FORBIDDEN_ROOT_DIRS = {
    ".git",
    ".workbuddy",
    "dataset",
    "debug",
    "logs",
    "models",
    "release",
    "runs",
    "tests",
    "tmp",
}
MODEL_SUFFIXES = {".pt", ".onnx", ".engine"}
CUDA_DLL_PREFIXES = (
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cudart",
    "nvrtc",
    "nvtoolsext",
)


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (path for path in root.rglob("*") if path.is_file())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile(value: RuntimeProfile | str) -> RuntimeProfile:
    return value if isinstance(value, RuntimeProfile) else RuntimeProfile(str(value))


def _find_named_file(root: Path, name: str) -> Path | None:
    expected = name.lower()
    return next(
        (path for path in _iter_files(root) if path.name.lower() == expected),
        None,
    )


def _find_prefixed_dll(root: Path, prefixes: Sequence[str]) -> Path | None:
    lowered = tuple(prefix.lower() for prefix in prefixes)
    return next(
        (
            path
            for path in _iter_files(root)
            if path.suffix.lower() == ".dll"
            and path.name.lower().startswith(lowered)
        ),
        None,
    )


def _is_cuda_library(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() == ".dll"
        and (
            name.startswith(CUDA_DLL_PREFIXES)
            or name in {"torch_cuda.dll", "c10_cuda.dll"}
            or "_cuda" in name
        )
    )


def _load_runtime_manifest(path: Path) -> RuntimeArtifact:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 runtime.json: {exc}") from exc
    return RuntimeArtifact.from_dict(data)


def check_runtime_manifest(
    path: Path,
    profile: RuntimeProfile | str,
) -> list[str]:
    """检查 Worker 运行时目录的结构、隔离性和 CPU/GPU 依赖边界。"""
    root = Path(path)
    expected = _profile(profile)
    errors: list[str] = []
    if not root.is_dir():
        return [f"运行时输出目录不存在: {root}"]

    if not (root / WORKER_EXE_NAME).is_file():
        errors.append(f"缺少 Worker: {WORKER_EXE_NAME}")
    if (root / GUI_EXE_NAME).exists():
        errors.append(f"禁止运行时包含主界面: {GUI_EXE_NAME}")
    internal = root / "_internal"
    if not internal.is_dir():
        errors.append("缺少 Worker 依赖目录: _internal")

    for required in REQUIRED_FILES:
        if not (root / required).is_file():
            errors.append(f"缺少运行时文件: {required}")

    manifest_path = root / "runtime.json"
    if manifest_path.is_file():
        try:
            artifact = _load_runtime_manifest(manifest_path)
            if artifact.profile is not expected:
                errors.append(
                    f"runtime.json profile 不匹配: 期望 {expected.value}，"
                    f"实际 {artifact.profile.value}"
                )
            if artifact.worker_protocol not in SUPPORTED_WORKER_PROTOCOLS:
                errors.append(f"不支持的 Worker 协议: {artifact.worker_protocol}")
            expected_suffix = "+cpu" if expected is RuntimeProfile.CPU else "+cu118"
            if not artifact.torch_version.endswith(expected_suffix):
                errors.append(
                    f"Torch 版本与 {expected.value} 运行时不匹配: "
                    f"{artifact.torch_version}"
                )
            if not artifact.torchvision_version.endswith(expected_suffix):
                errors.append(
                    f"TorchVision 版本与 {expected.value} 运行时不匹配: "
                    f"{artifact.torchvision_version}"
                )
        except ValueError as exc:
            errors.append(f"runtime.json 无效: {exc}")

    torch_cpu = _find_named_file(internal, "torch_cpu.dll")
    if torch_cpu is None:
        errors.append("缺少 Torch 动态库: torch_cpu.dll")

    if expected is RuntimeProfile.CPU:
        cuda_files = [path for path in _iter_files(internal) if _is_cuda_library(path)]
        for file_path in cuda_files:
            relative = file_path.relative_to(root).as_posix()
            errors.append(f"CPU 运行时包含 CUDA 动态库: {relative}")
    else:
        gpu_requirements = (
            ("torch_cuda.dll", _find_named_file(internal, "torch_cuda.dll")),
            ("cuDNN", _find_prefixed_dll(internal, ("cudnn",))),
            ("cuBLAS", _find_prefixed_dll(internal, ("cublas",))),
        )
        for label, found in gpu_requirements:
            if found is None:
                errors.append(f"缺少 GPU 动态库: {label}")

    for file_path in _iter_files(root):
        relative = file_path.relative_to(root)
        top_level = relative.parts[0].lower() if relative.parts else ""
        parts = {part.lower() for part in relative.parts}
        if file_path.suffix.lower() in MODEL_SUFFIXES:
            errors.append(f"禁止运行时包含模型文件: {relative.as_posix()}")
        if top_level in FORBIDDEN_ROOT_DIRS:
            errors.append(f"禁止运行时包含用户运行目录: {relative.as_posix()}")
        if "pyqt5" in parts:
            errors.append(f"禁止运行时包含 PyQt 主界面依赖: {relative.as_posix()}")

    return sorted(set(errors))


def prune_duplicate_torch_dlls(runtime_dir: Path) -> list[str]:
    """只删除与 torch/lib 中同名且内容完全一致的根目录 DLL 副本。"""
    internal = Path(runtime_dir) / "_internal"
    canonical = internal / "torch" / "lib"
    if not internal.is_dir() or not canonical.is_dir():
        return []

    removed: list[str] = []
    for duplicate in sorted(internal.glob("*.dll"), key=lambda path: path.name.lower()):
        original = canonical / duplicate.name
        if not original.is_file():
            continue
        try:
            identical = (
                duplicate.stat().st_size == original.stat().st_size
                and _sha256_file(duplicate) == _sha256_file(original)
            )
        except OSError:
            identical = False
        if identical:
            duplicate.unlink()
            removed.append(duplicate.name)
    return removed


def _installed_size(root: Path) -> int:
    return sum(path.stat().st_size for path in _iter_files(root))


def package_runtime_directory(
    runtime_dir: Path,
    static_artifact: RuntimeArtifact,
    output_dir: Path,
    *,
    base_url: str,
) -> tuple[Path, RuntimeArtifact]:
    """将已校验运行时目录压缩为 ZIP，并返回带真实下载元数据的清单项。"""
    root = Path(runtime_dir).resolve()
    output = Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("运行时 ZIP 输出目录不能位于待打包运行时目录内部")

    errors = check_runtime_manifest(root, static_artifact.profile)
    if errors:
        raise ValueError("运行时目录校验失败: " + "; ".join(errors))
    manifest = _load_runtime_manifest(root / "runtime.json")
    if not static_artifact.is_compatible_with(manifest):
        raise ValueError("runtime.json 与静态运行时清单不兼容")

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"{static_artifact.runtime_id}.zip"
    temp_path = archive_path.with_suffix(".zip.tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for file_path in sorted(_iter_files(root), key=lambda item: item.relative_to(root).as_posix()):
                archive.write(file_path, file_path.relative_to(root).as_posix())
        os.replace(temp_path, archive_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    url = f"{str(base_url).rstrip('/')}/{archive_path.name}"
    completed = replace(
        static_artifact,
        archive_size=archive_path.stat().st_size,
        installed_size=_installed_size(root),
        sha256=_sha256_file(archive_path),
        url=url,
    )
    completed = RuntimeArtifact.from_dict(completed.to_dict())
    archive_errors = check_runtime_archive(archive_path, completed)
    if archive_errors:
        raise ValueError("运行时 ZIP 校验失败: " + "; ".join(archive_errors))
    return archive_path, completed


def check_runtime_archive(
    archive_path: Path,
    artifact: RuntimeArtifact,
) -> list[str]:
    """不解压地检查运行时 ZIP 的可信元数据、目录结构和依赖边界。"""
    path = Path(archive_path)
    errors: list[str] = []
    if not path.is_file():
        return [f"运行时 ZIP 不存在: {path}"]
    if not artifact.is_downloadable:
        errors.append("运行时清单项缺少下载大小、SHA256 或 URL")
    else:
        if path.stat().st_size != artifact.archive_size:
            errors.append(
                f"运行时 ZIP 大小不匹配: 期望 {artifact.archive_size}，"
                f"实际 {path.stat().st_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != artifact.sha256:
            errors.append(
                f"运行时 ZIP SHA256 不匹配: 期望 {artifact.sha256}，"
                f"实际 {actual_sha256}"
            )

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                errors.append("运行时 ZIP 包含重复路径")
            for name in names:
                member = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or member.is_absolute()
                    or ".." in member.parts
                    or (member.parts and ":" in member.parts[0])
                ):
                    errors.append(f"运行时 ZIP 包含非法路径: {name}")

            name_set = set(names)
            for required in (*REQUIRED_FILES, WORKER_EXE_NAME):
                if required not in name_set:
                    errors.append(f"运行时 ZIP 缺少文件: {required}")
            if GUI_EXE_NAME in name_set:
                errors.append(f"禁止运行时 ZIP 包含主界面: {GUI_EXE_NAME}")

            total_size = sum(item.file_size for item in infos)
            if artifact.installed_size is not None and total_size != artifact.installed_size:
                errors.append(
                    f"运行时安装大小不匹配: 期望 {artifact.installed_size}，"
                    f"实际 {total_size}"
                )

            if not any(PurePosixPath(name).name.lower() == "torch_cpu.dll" for name in names):
                errors.append("运行时 ZIP 缺少 Torch 动态库: torch_cpu.dll")

            cuda_names = [
                name
                for name in names
                if _is_cuda_library(Path(PurePosixPath(name).name))
            ]
            if artifact.profile is RuntimeProfile.CPU:
                for name in cuda_names:
                    errors.append(f"CPU 运行时 ZIP 包含 CUDA 动态库: {name}")
            else:
                basenames = [PurePosixPath(name).name.lower() for name in names]
                gpu_requirements = (
                    ("torch_cuda.dll", any(name == "torch_cuda.dll" for name in basenames)),
                    ("cuDNN", any(name.startswith("cudnn") for name in basenames)),
                    ("cuBLAS", any(name.startswith("cublas") for name in basenames)),
                )
                for label, found in gpu_requirements:
                    if not found:
                        errors.append(f"运行时 ZIP 缺少 GPU 动态库: {label}")

            for name in names:
                member = PurePosixPath(name)
                parts = {part.lower() for part in member.parts}
                top_level = member.parts[0].lower() if member.parts else ""
                if member.suffix.lower() in MODEL_SUFFIXES:
                    errors.append(f"禁止运行时 ZIP 包含模型文件: {name}")
                if top_level in FORBIDDEN_ROOT_DIRS:
                    errors.append(f"禁止运行时 ZIP 包含用户运行目录: {name}")
                if "pyqt5" in parts:
                    errors.append(f"禁止运行时 ZIP 包含 PyQt 主界面依赖: {name}")

            runtime_name = next(
                (name for name in names if name.lower() == "runtime.json"),
                None,
            )
            if runtime_name is not None:
                try:
                    manifest = RuntimeArtifact.from_dict(
                        json.loads(archive.read(runtime_name).decode("utf-8"))
                    )
                    if not artifact.is_compatible_with(manifest):
                        errors.append("运行时 ZIP 中的 runtime.json 与可信清单不兼容")
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"运行时 ZIP 中的 runtime.json 无效: {exc}")
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"无法读取运行时 ZIP: {exc}")
    return sorted(set(errors))


def write_runtime_catalog(
    artifacts: Iterable[RuntimeArtifact],
    output_path: Path,
) -> Path:
    """原子写入只包含真实可下载项的可信运行时清单。"""
    items = tuple(
        sorted(
            artifacts,
            key=lambda item: (
                0 if item.profile is RuntimeProfile.CPU else 1,
                item.runtime_id,
            ),
        )
    )
    if not items:
        raise ValueError("运行时清单不能为空")
    if any(not item.is_downloadable for item in items):
        raise ValueError("运行时清单只能写入已生成大小、SHA256 和 URL 的项目")
    catalog = RuntimeCatalog.from_dict(
        {
            "catalog_version": 1,
            "artifacts": [item.to_dict() for item in items],
        }
    )

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(
                {
                    "catalog_version": catalog.catalog_version,
                    "artifacts": [item.to_dict() for item in catalog.artifacts],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return target


def _load_static_artifact(
    profile: RuntimeProfile,
    profiles_path: Path = DEFAULT_PROFILES,
) -> RuntimeArtifact:
    catalog = RuntimeCatalog.from_file(profiles_path)
    matches = catalog.for_profile(profile)
    if len(matches) != 1:
        raise ValueError(f"静态清单必须且只能包含一个 {profile.value} 运行时")
    artifact = matches[0]
    if artifact.is_downloadable:
        raise ValueError("静态运行时清单不能预填下载大小、SHA256 或 URL")
    return artifact


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _check_build_environment(artifact: RuntimeArtifact) -> list[str]:
    errors: list[str] = []
    disk_error = check_build_disk_space(PROJECT_DIR)
    if disk_error:
        errors.append(disk_error)
    if sys.version_info[:2] not in SUPPORTED_BUILD_PYTHON:
        errors.append(
            "运行时构建环境要求 Python 3.12/3.13 x64，"
            f"当前为 {platform.python_version()}"
        )
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        errors.append(f"运行时构建环境要求 x64，当前架构为 {platform.machine()}")
    for module_name in (
        "PyInstaller",
        "cv2",
        "numpy",
        "PIL",
        "yaml",
        "torch",
        "torchvision",
        "ultralytics",
        "tensorboard",
    ):
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"运行时构建依赖不可导入: {module_name}")

    expected_versions = {
        "torch": artifact.torch_version,
        "torchvision": artifact.torchvision_version,
        "ultralytics": artifact.ultralytics_version,
    }
    for package_name, expected in expected_versions.items():
        actual = _package_version(package_name)
        if actual is not None and actual != expected:
            errors.append(
                f"{package_name} 版本不匹配: 期望 {expected}，实际 {actual}"
            )

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            actual_cuda = None if torch.version.cuda is None else str(torch.version.cuda)
            expected_cuda = artifact.cuda_version
            if actual_cuda != expected_cuda:
                errors.append(
                    f"Torch CUDA 版本不匹配: 期望 {expected_cuda or 'CPU'}，"
                    f"实际 {actual_cuda or 'CPU'}"
                )
        except BaseException as exc:
            errors.append(f"Torch 导入失败: {exc}")
    return errors


def _write_runtime_json(runtime_dir: Path, artifact: RuntimeArtifact) -> None:
    (Path(runtime_dir) / "runtime.json").write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_runtime_metadata(runtime_dir: Path) -> None:
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(PROJECT_DIR / name, Path(runtime_dir) / name)


def _run_pyinstaller(
    artifact: RuntimeArtifact,
    build_root: Path,
    *,
    clean: bool,
) -> tuple[int, Path]:
    dist_dir = Path(build_root) / "dist" / artifact.profile.value
    work_dir = Path(build_root) / "work" / artifact.profile.value
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
    ]
    if clean:
        command.append("--clean")
    command.append(str(RUNTIME_SPEC))
    environment = os.environ.copy()
    environment[RUNTIME_PROFILE_ENV] = artifact.profile.value
    environment[RUNTIME_ID_ENV] = artifact.runtime_id
    completed = subprocess.run(command, cwd=str(PROJECT_DIR), env=environment)
    return completed.returncode, dist_dir / artifact.runtime_id


def _merge_catalog_entry(output_dir: Path, artifact: RuntimeArtifact) -> Path:
    catalog_path = Path(output_dir) / "runtime_catalog.json"
    current: dict[str, RuntimeArtifact] = {}
    if catalog_path.is_file():
        existing = RuntimeCatalog.from_file(catalog_path)
        current.update((item.runtime_id, item) for item in existing.artifacts)
    current[artifact.runtime_id] = artifact
    return write_runtime_catalog(current.values(), catalog_path)


def build_runtime_archive(
    profile: RuntimeProfile | str,
    output_dir: Path,
    base_url: str,
    *,
    profiles_path: Path = DEFAULT_PROFILES,
    build_root: Path = DEFAULT_BUILD_DIR,
    clean: bool = False,
) -> Path:
    """构建单个 Worker 运行时，并增量更新候选可信清单。"""
    expected = _profile(profile)
    artifact = _load_static_artifact(expected, Path(profiles_path))
    errors = _check_build_environment(artifact)
    if errors:
        raise RuntimeError("; ".join(errors))

    return_code, runtime_dir = _run_pyinstaller(artifact, Path(build_root), clean=clean)
    if return_code != 0:
        raise RuntimeError(f"PyInstaller 构建失败，退出码 {return_code}")
    _write_runtime_json(runtime_dir, artifact)
    _copy_runtime_metadata(runtime_dir)
    prune_duplicate_torch_dlls(runtime_dir)
    errors = check_runtime_manifest(runtime_dir, expected)
    if errors:
        raise RuntimeError("运行时清单检查失败: " + "; ".join(errors))

    archive, completed = package_runtime_directory(
        runtime_dir,
        artifact,
        Path(output_dir),
        base_url=base_url,
    )
    entry_path = Path(output_dir) / f"{artifact.runtime_id}.artifact.json"
    entry_path.write_text(
        json.dumps(completed.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _merge_catalog_entry(Path(output_dir), completed)
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 YOLO 工具箱按需机器学习运行时")
    parser.add_argument("--profile", required=True, choices=[item.value for item in RuntimeProfile])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="ZIP 和可信清单输出目录")
    parser.add_argument("--base-url", default="", help="正式 HTTPS Release 资源目录")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES), help="静态运行时版本清单")
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD_DIR), help="PyInstaller 临时目录")
    parser.add_argument("--clean", action="store_true", help="清理当前 profile 的 PyInstaller 缓存")
    parser.add_argument("--dry-run", action="store_true", help="只检查环境和静态清单")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        artifact = _load_static_artifact(
            RuntimeProfile(args.profile),
            Path(args.profiles).resolve(),
        )
        errors = _check_build_environment(artifact)
        if errors:
            raise RuntimeError("; ".join(errors))
        if args.dry_run:
            print(f"[通过] {artifact.runtime_id} 构建环境和静态清单有效；未生成运行时。")
            return 0
        if not str(args.base_url).strip():
            raise ValueError("真实运行时构建必须通过 --base-url 提供固定 HTTPS Release 地址")
        archive = build_runtime_archive(
            artifact.profile,
            Path(args.out_dir).resolve(),
            args.base_url,
            profiles_path=Path(args.profiles).resolve(),
            build_root=Path(args.build_root).resolve(),
            clean=args.clean,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[错误] {exc}")
        return 2

    print(f"[通过] 已生成运行时: {archive}")
    print(f"大小: {archive.stat().st_size / 1024**2:.2f} MiB")
    print(f"SHA256: {_sha256_file(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

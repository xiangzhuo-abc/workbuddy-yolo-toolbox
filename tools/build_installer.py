"""校验 PyInstaller 便携目录并构建 Inno Setup 安装器。"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

try:
    from .build_exe import GUI_EXE_NAME, check_exe_manifest
    from .core.version import (
        DISPLAY_VERSION,
        FILE_VERSION,
        PRODUCT_NAME,
        get_source_url,
    )
except ImportError:
    from build_exe import GUI_EXE_NAME, check_exe_manifest
    from core.version import DISPLAY_VERSION, FILE_VERSION, PRODUCT_NAME, get_source_url


PROJECT_DIR = Path(__file__).resolve().parent.parent
INSTALLER_SCRIPT = PROJECT_DIR / "packaging" / "installer.iss"
DEFAULT_EXE_DIR = PROJECT_DIR / "tmp" / "exe-candidate"
DEFAULT_OUT_DIR = PROJECT_DIR / "tmp" / "installer-candidate"
INSTALLER_BASE_NAME = f"YOLO数据标注工具箱-{DISPLAY_VERSION}-windows-x64-setup"
INSTALLER_FILE_NAME = f"{INSTALLER_BASE_NAME}.exe"
ISCC_ENV = "WORKBUDDY_ISCC"


def resolve_exe_dir(path: Path) -> Path:
    """兼容 dist 根目录和实际 COLLECT 子目录。"""
    candidate = Path(path).resolve()
    if (candidate / GUI_EXE_NAME).is_file():
        return candidate
    nested = candidate / "YOLO数据标注工具箱"
    if (nested / GUI_EXE_NAME).is_file():
        return nested.resolve()
    return candidate


def check_installer_manifest(path: Path) -> list[str]:
    """复用 EXE 隔离规则检查安装器输入目录。"""
    return check_exe_manifest(resolve_exe_dir(path))


def find_iscc(explicit: Path | str | None = None) -> Path | None:
    """按显式参数、环境变量、PATH 和常见安装目录查找 ISCC。"""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None

    environment_path = str(os.environ.get(ISCC_ENV, "")).strip()
    if environment_path:
        path = Path(environment_path).expanduser().resolve()
        if path.is_file():
            return path

    command_path = shutil.which("ISCC.exe") or shutil.which("iscc")
    if command_path:
        return Path(command_path).resolve()

    local_app_data = str(os.environ.get("LOCALAPPDATA", "")).strip()
    candidates = []
    if local_app_data:
        candidates.extend([
            Path(local_app_data) / "Programs" / "Inno Setup 6" / "ISCC.exe",
            Path(local_app_data) / "Programs" / "Inno Setup 7" / "ISCC.exe",
        ])
    candidates.extend([
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files (x86)\Inno Setup 7\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 7\ISCC.exe"),
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def build_iscc_command(
    exe_dir: Path,
    out_dir: Path,
    *,
    source_url: str,
    iscc: Path,
) -> list[str]:
    """构造不依赖当前工作目录的 Inno Setup 编译命令。"""
    source = resolve_exe_dir(exe_dir)
    output = Path(out_dir).resolve()
    file_version = ".".join(str(value) for value in FILE_VERSION)
    return [
        str(Path(iscc)),
        "/Qp",
        f"/DAppVersion={DISPLAY_VERSION}",
        f"/DFileVersion={file_version}",
        f"/DInstallerBaseName={INSTALLER_BASE_NAME}",
        f"/DSourceDir={source}",
        f"/DOutputDir={output}",
        f"/DSourceUrl={source_url}",
        str(INSTALLER_SCRIPT.resolve()),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 YOLO 工具箱 Inno Setup 安装器")
    parser.add_argument("--exe-dir", default=str(DEFAULT_EXE_DIR), help="便携 EXE 目录")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="安装器输出目录")
    parser.add_argument("--source-url", default=None, help="本次二进制对应的公开源码地址")
    parser.add_argument("--official-release", action="store_true", help="正式发布模式，要求源码地址")
    parser.add_argument("--iscc", default=None, help="ISCC.exe 路径")
    parser.add_argument("--dry-run", action="store_true", help="仅检查并显示命令")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        source_url = get_source_url(args.source_url, required=args.official_release)
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2

    exe_dir = resolve_exe_dir(Path(args.exe_dir))
    errors = check_installer_manifest(exe_dir)
    if errors:
        for error in errors:
            print(f"[错误] {error}")
        return 2

    compiler = find_iscc(args.iscc)
    if compiler is None and not args.dry_run:
        print(
            "[错误] 未找到 Inno Setup 编译器 ISCC.exe。"
            f"请安装 Inno Setup 6/7，或通过 --iscc / {ISCC_ENV} 指定路径。"
        )
        return 3

    out_dir = Path(args.out_dir).resolve()
    command = build_iscc_command(
        exe_dir,
        out_dir,
        source_url=source_url,
        iscc=compiler or Path("ISCC.exe"),
    )
    print(f"产品: {PRODUCT_NAME} {DISPLAY_VERSION}")
    print(f"安装器输入: {exe_dir}")
    print(f"安装器输出: {out_dir / INSTALLER_FILE_NAME}")
    if args.dry_run:
        print("命令: " + subprocess.list2cmdline(command))
        if compiler is None:
            print("[信息] 当前未找到 ISCC.exe；dry-run 未执行编译。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / INSTALLER_FILE_NAME
    if output_path.exists():
        output_path.unlink()
    completed = subprocess.run(command, cwd=str(PROJECT_DIR))
    if completed.returncode != 0:
        print(f"[错误] Inno Setup 编译失败，退出码 {completed.returncode}。")
        return completed.returncode
    if not output_path.is_file():
        print(f"[错误] Inno Setup 未生成预期文件: {output_path}")
        return 4

    print(f"[通过] 已生成安装器: {output_path}")
    print(f"大小: {output_path.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"SHA256: {sha256_file(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""YOLO 工具箱依赖安装助手。

这个脚本不依赖 PyQt，适合在 GUI 无法启动时先检查和安装依赖。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from core.python_support import describe_python


PROJECT_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS = PROJECT_DIR / "requirements.txt"

REQUIRED_PACKAGES = [
    ("PyQt5", "PyQt5", "PyQt5"),
    ("opencv-python", "cv2", "opencv-python"),
    ("numpy", "numpy", "numpy"),
    ("pillow", "PIL", "pillow"),
    ("pytesseract", "pytesseract", "pytesseract"),
    ("ultralytics", "ultralytics", "ultralytics"),
    ("torch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision"),
    ("PyYAML", "yaml", "PyYAML"),
    ("tensorboard", "tensorboard", "tensorboard"),
]


def _print_title(text: str) -> None:
    print()
    print("=" * 64)
    print(text)
    print("=" * 64)
    sys.stdout.flush()


def _run_command(cmd: list[str], cwd: Path | None = None) -> int:
    print("> " + " ".join(cmd))
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=str(cwd or PROJECT_DIR))
    return proc.returncode


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except Exception:
        return "已安装"


def check_python() -> int:
    _print_title("Python 环境")
    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    print(f"项目目录: {PROJECT_DIR}")
    support = describe_python()
    if not support.supported:
        print(f"[错误] {support.message}")
        return 1
    if support.recommended:
        print(f"[通过] {support.message}")
    else:
        print(f"[警告] {support.message}")
    return 0


def check_pip() -> int:
    _print_title("pip 检查")
    cmd = [sys.executable, "-m", "pip", "--version"]
    return _run_command(cmd)


def check_imports() -> int:
    _print_title("依赖导入检查")
    missing = []
    for display_name, module_name, package_name in REQUIRED_PACKAGES:
        if find_spec(module_name) is None:
            missing.append(display_name)
            print(f"[缺失] {display_name}")
        else:
            print(f"[通过] {display_name} {_package_version(package_name)}")

    if missing:
        print()
        print("[提示] 缺失依赖:")
        for name in missing:
            print(f"  - {name}")
        print("请运行: python tools\\install_dependencies.py")
        return 1
    print("[通过] 运行依赖已完整安装。")
    return 0


def install_requirements(args) -> int:
    if not REQUIREMENTS.exists():
        print(f"[错误] 未找到 requirements.txt: {REQUIREMENTS}")
        return 1

    _print_title("安装依赖")
    print("将使用当前 Python 解释器安装依赖。")
    print("如果你需要 CUDA 版 PyTorch，请先按显卡和 CUDA 版本安装对应 torch/torchvision。")
    print("当前 requirements 会接受已安装的 CUDA 版 torch。")
    print()

    if args.upgrade_pip:
        rc = _run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        if rc != 0:
            print("[错误] pip 升级失败，已停止。")
            return rc

    cmd = [sys.executable, "-m", "pip", "install"]
    if args.dry_run:
        cmd.extend(["--dry-run", "--no-deps"])
    if args.index_url:
        cmd.extend(["--index-url", args.index_url])
    if args.extra_index_url:
        for value in args.extra_index_url:
            cmd.extend(["--extra-index-url", value])
    cmd.extend(["-r", str(REQUIREMENTS)])

    rc = _run_command(cmd)
    if rc != 0:
        print()
        print("[错误] 依赖安装失败。")
        print("常见处理：")
        print("  1. 确认网络可以访问 pip 源。")
        print("  2. 确认 Python 版本为 3.11。")
        print("  3. 如果 torch 安装失败，请按显卡/CUDA 版本先安装 PyTorch。")
        print("  4. 可重试: python tools\\install_dependencies.py --upgrade-pip")
        return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 工具箱依赖安装助手")
    parser.add_argument("--check-only", action="store_true", help="只检查依赖，不安装")
    parser.add_argument("--dry-run", action="store_true", help="只验证 requirements 能否被 pip 解析，不实际安装")
    parser.add_argument("--upgrade-pip", action="store_true", help="安装前先升级 pip")
    parser.add_argument("--index-url", default=None, help="指定 pip 主索引源")
    parser.add_argument("--extra-index-url", action="append", default=None, help="指定额外 pip 索引源，可重复传入")
    args = parser.parse_args()

    rc = check_python()
    if rc != 0:
        return rc
    rc = check_pip()
    if rc != 0:
        print("[错误] pip 不可用，请先修复 Python/pip 安装。")
        return rc

    if args.check_only:
        return check_imports()

    rc = install_requirements(args)
    if rc != 0:
        return rc
    if args.dry_run:
        print("[通过] dry-run 完成，requirements.txt 可以被 pip 解析。")
        return 0

    return check_imports()


if __name__ == "__main__":
    raise SystemExit(main())

"""生成源码绿色发布包，避免把训练数据和运行产物混入发布物。"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

from core.python_support import MAX_SUPPORTED, MIN_SUPPORTED, selection_text
from core.version import DISPLAY_VERSION, PRODUCT_NAME, get_source_url


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_NAME = "YOLO数据标注工具箱"

ROOT_FILES = [
    "CHANGELOG.md",
    "LICENSE",
    "README_YOLO.md",
    "README_VISION.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "requirements-base-windows.txt",
    "requirements-build.txt",
    "requirements-runtime-windows-cpu.txt",
    "requirements-release-windows-cu118.txt",
]

EXCLUDED_TOP_DIRS = {
    ".git",
    ".workbuddy",
    "build",
    "debug",
    "dist",
    "logs",
    "release",
    "runs",
    "venv",
    ".venv",
}

EXCLUDED_DIR_NAMES = {
    "__pycache__",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".cache",
    ".tmp",
    ".bak",
}


def _is_model_file(path: Path) -> bool:
    return path.suffix.lower() == ".pt"


def _should_include_source(rel_path: Path, include_models: bool) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_TOP_DIRS:
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return False
    if rel_path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if _is_model_file(rel_path):
        return include_models and (parts[0] == "models" or len(parts) == 1)

    if len(parts) == 1 and rel_path.as_posix() in ROOT_FILES:
        return True
    if parts[0] == "tools" and rel_path.suffix.lower() == ".py":
        return True
    if parts[0] == "vision" and rel_path.suffix.lower() == ".py":
        return True
    if parts[0] == "packaging" and rel_path.suffix.lower() in {
        ".spec",
        ".iss",
        ".isl",
        ".json",
        ".ps1",
    }:
        return True
    if parts[0] == "tests" and rel_path.suffix.lower() == ".py":
        return True
    if parts[0] == "templates":
        return rel_path.name.lower().startswith("_debug") is False
    return False


def collect_sources(include_models: bool) -> list[Path]:
    """收集需要复制到发布包的真实文件。"""
    files = []
    for path in PROJECT_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(PROJECT_DIR)
        if _should_include_source(rel_path, include_models):
            files.append(rel_path)
    return sorted(files, key=lambda p: p.as_posix().lower())


def generated_files(source_url: str | None = None) -> dict[str, bytes]:
    """生成发布包里的空数据集骨架和便捷入口。"""
    source_url = get_source_url(source_url)
    source_line = source_url or "未配置（开发构建）"
    data_yaml = "\n".join([
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names: {}",
        "",
    ])
    release_readme = "\n".join([
        f"{PRODUCT_NAME} {DISPLAY_VERSION} - 源码绿色包",
        "",
        "首次使用：",
        "1. 双击 安装依赖.bat；脚本会从 Python 3.9-3.14 中选择可用版本，创建包内 .venv 并安装依赖。",
        "2. 如果依赖安装失败，按窗口中的中文提示处理网络、Python 或 PyTorch/CUDA 问题。",
        "3. 双击 启动YOLO工具箱.bat 打开主界面。",
        "4. 在主界面先运行「环境体检」和「发布前自检」，再按数据准备、标注、划分、训练、测试流程操作。",
        "",
        "说明：",
        "- .venv 是发布包自己的运行环境，不会把依赖安装到系统 Python。",
        "- 推荐 Python 3.11-3.13；Python 3.9-3.10 为兼容模式，Python 3.14 为实验性兼容。",
        "- 发布包默认不包含你的 dataset、runs、debug、logs 和本机配置。",
        "- dataset/ 只是空骨架，用户数据会在本机生成。",
        "- 如需随包附带模型，请用 build_release.py --include-models 重新打包。",
        "- 本项目使用 AGPL-3.0-only；许可证见 LICENSE。",
        f"- 对应源码: {source_line}",
        "",
    ])
    install_bat = "\r\n".join([
        "@echo off",
        "setlocal EnableExtensions EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "chcp 65001 >nul",
        "set \"VENV_DIR=%~dp0.venv\"",
        "set \"PYTHON_EXE=%VENV_DIR%\\Scripts\\python.exe\"",
        "if not exist \"%PYTHON_EXE%\" (",
        "  echo 正在选择受支持的 Python 环境...",
        "  set \"PYTHON_LAUNCHER=\"",
        "  where py >nul 2>&1",
        "  if not errorlevel 1 for %%V in (" + selection_text() + ") do (",
        "    if not defined PYTHON_LAUNCHER (",
        "      py -%%V -c \"import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)\" >nul 2>&1",
        "      if not errorlevel 1 set \"PYTHON_LAUNCHER=py -%%V\"",
        "    )",
        "  )",
        "  if not defined PYTHON_LAUNCHER (",
        "    where python >nul 2>&1",
        "    if errorlevel 1 (",
        "      echo 未找到 Python 3.9-3.14。请先安装受支持的 Python 版本。",
        "      pause",
        "      exit /b 1",
        "    )",
        "    python -c \"import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)\" >nul 2>&1",
        "    if errorlevel 1 (",
        "      echo 当前 Python 不在支持范围 Python 3.9-3.14 内。",
        "      pause",
        "      exit /b 1",
        "    )",
        "    set \"PYTHON_LAUNCHER=python\"",
        "  )",
        "  !PYTHON_LAUNCHER! -m venv \"%VENV_DIR%\"",
        ")",
        "if not exist \"%PYTHON_EXE%\" (",
        "  echo 创建包内 Python 环境失败，请确认已安装 Python 3.9-3.14。",
        "  pause",
        "  exit /b 1",
        ")",
        "\"%PYTHON_EXE%\" -c \"import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)\"",
        "if errorlevel 1 (",
        "  echo 包内环境不在 Python 3.9-3.14 支持范围内，请删除 .venv 后重新运行此脚本。",
        "  pause",
        "  exit /b 1",
        ")",
        "\"%PYTHON_EXE%\" -u tools\\install_dependencies.py",
        "set \"RC=%ERRORLEVEL%\"",
        "if not \"%RC%\"==\"0\" echo 依赖安装失败，退出码: %RC%",
        "pause",
        "exit /b %RC%",
        "",
    ])
    launch_bat = "\r\n".join([
        "@echo off",
        "setlocal EnableExtensions",
        "cd /d \"%~dp0\"",
        "chcp 65001 >nul",
        "set \"PYTHON_EXE=%~dp0.venv\\Scripts\\python.exe\"",
        "if not exist \"%PYTHON_EXE%\" (",
        "  echo 未找到包内 Python 环境，请先双击 安装依赖.bat。",
        "  pause",
        "  exit /b 1",
        ")",
        "\"%PYTHON_EXE%\" -c \"import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)\"",
        "if errorlevel 1 (",
        "  echo 包内环境不在 Python 3.9-3.14 支持范围内，请删除 .venv 后重新运行 安装依赖.bat。",
        "  pause",
        "  exit /b 1",
        ")",
        "\"%PYTHON_EXE%\" -u tools\\install_dependencies.py --check-only",
        "if errorlevel 1 (",
        "  echo.",
        "  echo 依赖未完整安装，请先双击 安装依赖.bat。",
        "  pause",
        "  exit /b 1",
        ")",
        "\"%PYTHON_EXE%\" -u tools\\yolo_tool_launcher.py",
        "set \"RC=%ERRORLEVEL%\"",
        "pause",
        "exit /b %RC%",
        "",
    ])
    files = {
        "README_RELEASE.txt": release_readme.encode("utf-8"),
        "安装依赖.bat": install_bat.encode("utf-8-sig"),
        "启动YOLO工具箱.bat": launch_bat.encode("utf-8-sig"),
        "config/.keep": b"",
        "models/.keep": b"",
        "logs/.keep": b"",
        "dataset/classes.txt": b"",
        "dataset/data.yaml": data_yaml.encode("utf-8"),
    }
    for split in ("train", "val", "test", "unlabeled"):
        files[f"dataset/images/{split}/.keep"] = b""
    for split in ("train", "val", "test"):
        files[f"dataset/labels/{split}/.keep"] = b""
    return files


def write_zip(out_path: Path, package_name: str, sources: list[Path], generated: dict[str, bytes]) -> None:
    """写入 zip 包。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in sources:
            arcname = Path(package_name) / rel_path
            zf.write(PROJECT_DIR / rel_path, arcname.as_posix())
        for rel_text, content in generated.items():
            arcname = Path(package_name) / rel_text
            zf.writestr(arcname.as_posix(), content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 YOLO 工具箱源码绿色发布包")
    parser.add_argument("--name", default=DEFAULT_PACKAGE_NAME, help="压缩包内顶层目录名")
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "release"), help="输出目录")
    parser.add_argument("--include-models", action="store_true", help="包含根目录和 models/ 下的 .pt 模型")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要打包的文件，不生成 zip")
    parser.add_argument("--source-url", default=None, help="本次二进制对应的公开源码地址")
    parser.add_argument("--official-release", action="store_true", help="正式发布模式，要求提供源码地址")
    args = parser.parse_args()

    try:
        source_url = get_source_url(args.source_url, required=args.official_release)
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2

    sources = collect_sources(include_models=args.include_models)
    generated = generated_files(source_url)
    total_source_bytes = sum((PROJECT_DIR / p).stat().st_size for p in sources)

    print(f"项目目录: {PROJECT_DIR}")
    print(f"版本: {DISPLAY_VERSION}")
    print(f"真实文件: {len(sources)} 个，{total_source_bytes / 1024 / 1024:.2f} MB")
    print(f"生成文件: {len(generated)} 个")
    print("默认排除: dataset 实际数据、runs、debug、logs、.git、.workbuddy、__pycache__")
    if not args.include_models:
        print("模型文件: 已排除（可用 --include-models 包含）")

    if args.dry_run:
        print("\n真实文件清单:")
        for rel_path in sources:
            print(f"  {rel_path.as_posix()}")
        print("\n生成文件清单:")
        for rel_path in sorted(generated):
            print(f"  {rel_path}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    suffix = "-with-models" if args.include_models else ""
    out_path = out_dir / f"{args.name}-{DISPLAY_VERSION}-{timestamp}{suffix}.zip"
    write_zip(out_path, args.name, sources, generated)
    digest = sha256_file(out_path)
    print(f"\n已生成: {out_path}")
    print(f"大小: {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

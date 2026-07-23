"""发布前自检：检查运行环境、依赖清单和发布包隔离规则。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import build_release
import build_exe
import build_runtime
import yolo_dataset_tools as tools
from core.ml_runtime import RuntimeCatalog, RuntimeProfile
from core.python_support import describe_python


PROJECT_DIR = tools.get_project_dir()

RUNTIME_TOP_DIRS = {
    ".git",
    ".workbuddy",
    "dataset",
    "debug",
    "logs",
    "release",
    "runs",
}

REQUIRED_SOURCE_FILES = [
    "CHANGELOG.md",
    "LICENSE",
    "README_YOLO.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "requirements-base-windows.txt",
    "requirements-build.txt",
    "requirements-runtime-windows-cpu.txt",
    "requirements-release-windows-cu118.txt",
    "packaging/runtime_profiles.json",
    "packaging/smoke_test.ps1",
    "packaging/workbuddy_yolo.spec",
    "packaging/workbuddy_yolo_runtime.spec",
    "packaging/installer.iss",
    "packaging/languages/ChineseSimplified.isl",
    "tools/build_installer.py",
    "tools/build_release.py",
    "tools/build_runtime.py",
    "tools/core/__init__.py",
    "tools/core/annotation_service.py",
    "tools/core/config_store.py",
    "tools/core/dataset_scanner.py",
    "tools/core/dataset_quality.py",
    "tools/core/dataset_state.py",
    "tools/core/dataset_split.py",
    "tools/core/dataset_split_executor.py",
    "tools/core/issues.py",
    "tools/core/diagnostics.py",
    "tools/core/model_evaluation.py",
    "tools/core/paths.py",
    "tools/core/python_support.py",
    "tools/core/model_registry.py",
    "tools/core/ml_runtime.py",
    "tools/core/runtime_installer.py",
    "tools/core/runtime_paths.py",
    "tools/core/external_python.py",
    "tools/core/task_manager.py",
    "tools/core/task_protocol.py",
    "tools/core/worker_commands.py",
    "tools/core/version.py",
    "tools/install_dependencies.py",
    "tools/launch_tensorboard.py",
    "tools/preflight_check.py",
    "tools/yolo_annotate.py",
    "tools/yolo_annotate_gui.py",
    "tools/yolo_dataset_tools.py",
    "tools/yolo_detect_gui.py",
    "tools/yolo_detect_worker.py",
    "tools/yolo_evaluate_worker.py",
    "tools/yolo_evaluation_dialog.py",
    "tools/yolo_model_manager.py",
    "tools/yolo_quality_dialog.py",
    "tools/yolo_runtime_dialog.py",
    "tools/yolo_split_dialog.py",
    "tools/yolo_train_worker.py",
    "tools/yolo_worker_entry.py",
    "tools/build_exe.py",
    "tools/yolo_tool_launcher.py",
    "tools/yolo_ui_theme.py",
    "tools/yolo_ui_widgets.py",
]

CORE_TEST_COMMAND = [
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests/core",
    "-p",
    "test_*.py",
    "-v",
]
PIP_REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-base-windows.txt",
    "requirements-runtime-windows-cpu.txt",
    "requirements-release-windows-cu118.txt",
)


def _tail(text: str, max_lines: int = 8) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def _count_files(path: Path, limit: int = 10000) -> tuple[int, bool]:
    """计数目录文件，超过 limit 后停止，避免大数据集自检过慢。"""
    if not path.exists():
        return 0, False
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            if count >= limit:
                return count, True
    return count, False


def _resolve_base_candidate(path: Path) -> Path:
    root = Path(path).resolve()
    nested = root / "YOLO数据标注工具箱"
    if (nested / build_exe.GUI_EXE_NAME).is_file():
        return nested
    return root


def _check_release_artifacts(
    *,
    exe_dir: Path | None,
    runtime_dir: Path | None,
    add,
) -> None:
    """按需检查轻量基础目录、运行时 ZIP 和两侧可信清单的一致性。"""
    base_catalog: RuntimeCatalog | None = None
    if exe_dir is not None:
        candidate = _resolve_base_candidate(exe_dir)
        errors = build_exe.check_exe_manifest(candidate)
        if errors:
            for error in errors:
                add("error", f"基础程序候选: {error}")
        else:
            add("info", f"轻量基础程序候选检查通过: {candidate}")
            try:
                base_catalog = RuntimeCatalog.from_file(
                    candidate / "_internal" / "runtime_catalog.json"
                )
            except ValueError as exc:
                add("error", f"基础程序内嵌清单无效: {exc}")

    if runtime_dir is None:
        return
    root = Path(runtime_dir).resolve()
    catalog_path = root / "runtime_catalog.json"
    try:
        catalog = RuntimeCatalog.from_file(catalog_path)
    except ValueError as exc:
        add("error", f"运行时候选清单无效: {exc}")
        return

    runtime_errors: list[str] = []
    profiles = {item.profile for item in catalog.artifacts}
    if profiles != {RuntimeProfile.CPU, RuntimeProfile.CUDA118}:
        runtime_errors.append("运行时候选必须同时包含 CPU 和 CUDA 11.8")
    for artifact in catalog.artifacts:
        if not artifact.is_downloadable:
            runtime_errors.append(f"运行时候选不可下载: {artifact.runtime_id}")
            continue
        archive = root / f"{artifact.runtime_id}.zip"
        runtime_errors.extend(
            f"{artifact.runtime_id}: {error}"
            for error in build_runtime.check_runtime_archive(archive, artifact)
        )
    if runtime_errors:
        for error in sorted(set(runtime_errors)):
            add("error", error)
    else:
        add("info", f"CPU/GPU 运行时 ZIP 候选检查通过: {root}")

    if base_catalog is not None:
        base_items = {item.runtime_id: item.to_dict() for item in base_catalog.artifacts}
        runtime_items = {item.runtime_id: item.to_dict() for item in catalog.artifacts}
        if base_items != runtime_items:
            add("error", "基础程序内嵌清单与运行时候选清单不一致")
        else:
            add("info", "基础程序内嵌清单与运行时候选清单一致。")


def run_preflight(
    dataset_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    include_pip: bool = True,
    include_env: bool = True,
    include_core_tests: bool = True,
    exe_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    emit=None,
) -> dict:
    """执行发布前自检，返回包含错误和警告数量的报告。"""
    dataset_dir = Path(dataset_dir) if dataset_dir else tools.get_dataset_dir()
    runs_dir = Path(runs_dir) if runs_dir else PROJECT_DIR / "runs"
    lines: list[tuple[str, str]] = []
    errors = 0
    warnings = 0

    def add(level: str, message: str) -> None:
        nonlocal errors, warnings
        if level == "error":
            errors += 1
        elif level == "warning":
            warnings += 1
        lines.append((level, message))
        if emit:
            emit(level, message)

    add("info", "发布前自检")
    add("info", f"项目目录: {PROJECT_DIR}")
    add("info", f"Python: {sys.version.split()[0]} ({sys.executable})")

    support = describe_python()
    if not support.supported:
        add("error", support.message)
    elif support.recommended:
        add("info", support.message)
    else:
        add("warning", support.message)

    for rel_text in REQUIRED_SOURCE_FILES:
        path = PROJECT_DIR / rel_text
        if path.exists():
            add("info", f"关键文件存在: {rel_text}")
        else:
            add("error", f"关键文件缺失: {rel_text}")

    tests_dir = PROJECT_DIR / "tests" / "core"
    if include_core_tests and not tests_dir.is_dir():
        add("warning", "测试未随发布包附带，已跳过核心测试。")
    elif include_core_tests:
        add("info", "运行核心服务层测试...")
        try:
            test_env = os.environ.copy()
            # 发布自检必须可在无桌面环境运行，避免 Qt 平台插件把测试挂住。
            test_env["QT_QPA_PLATFORM"] = "offscreen"
            proc = subprocess.run(
                CORE_TEST_COMMAND,
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                env=test_env,
            )
        except subprocess.TimeoutExpired:
            add("error", "核心服务层测试超过 60 秒，已终止等待。")
        except OSError as exc:
            add("error", f"核心服务层测试无法启动: {exc}")
        else:
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode == 0:
                add("info", "核心服务层测试通过。")
            else:
                add("error", f"核心服务层测试失败，退出码 {proc.returncode}。")
                for line in _tail(output):
                    add("error", f"core: {line}")

    if include_pip:
        for requirement_name in PIP_REQUIREMENT_FILES:
            requirements = PROJECT_DIR / requirement_name
            if not requirements.exists():
                add("error", f"{requirement_name} 缺失，无法检查依赖清单。")
                continue
            add("info", f"检查 {requirement_name} 是否可被 pip 解析...")
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "--dry-run",
                "--no-deps",
            ]
            if requirement_name != "requirements.txt":
                cmd.extend([
                    "--ignore-installed",
                    "--only-binary=:all:",
                    "--python-version",
                    "3.13",
                ])
            cmd.extend(["-r", str(requirements)])
            try:
                pip_env = os.environ.copy()
                pip_env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_DIR),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                    env=pip_env,
                )
            except subprocess.TimeoutExpired:
                add("error", f"{requirement_name} pip dry-run 超时，依赖解析可能被网络或索引源阻塞。")
            else:
                output = (proc.stdout or "") + "\n" + (proc.stderr or "")
                if proc.returncode == 0:
                    add("info", f"{requirement_name} 解析通过。")
                else:
                    add("error", f"{requirement_name} 解析失败，退出码 {proc.returncode}。")
                    for line in _tail(output):
                        add("error", f"pip: {line}")
    else:
        add("warning", "已跳过 pip 依赖解析检查。")

    try:
        sources = build_release.collect_sources(include_models=False)
        generated = build_release.generated_files()
    except Exception as exc:
        add("error", f"发布文件收集失败: {exc}")
        sources = []
        generated = {}

    if sources or generated:
        add("info", f"发布真实文件: {len(sources)} 个；生成骨架文件: {len(generated)} 个。")
        bad_sources = []
        for rel_path in sources:
            parts = rel_path.parts
            if parts and parts[0] in RUNTIME_TOP_DIRS:
                bad_sources.append(rel_path.as_posix())
            elif rel_path.suffix.lower() in {".pt", ".onnx", ".engine"}:
                bad_sources.append(rel_path.as_posix())
        if bad_sources:
            add("error", "发布包真实文件清单包含运行数据或模型文件。")
            for rel_text in bad_sources[:20]:
                add("error", f"不应打包: {rel_text}")
            if len(bad_sources) > 20:
                add("error", f"另有 {len(bad_sources) - 20} 个风险文件未显示。")
        else:
            add("info", "发布真实文件清单未包含 dataset/runs/logs/debug/release 或模型文件。")

        source_names = {path.as_posix() for path in sources}
        for rel_text in REQUIRED_SOURCE_FILES:
            if rel_text not in source_names:
                add("error", f"关键文件未进入发布清单: {rel_text}")

        for rel_text in [
            "README_RELEASE.txt",
            "启动YOLO工具箱.bat",
            "安装依赖.bat",
            "dataset/classes.txt",
            "dataset/data.yaml",
        ]:
            if rel_text in generated:
                add("info", f"发布骨架会生成: {rel_text}")
            else:
                add("error", f"发布骨架缺失: {rel_text}")

    if exe_dir is not None or runtime_dir is not None:
        _check_release_artifacts(
            exe_dir=None if exe_dir is None else Path(exe_dir),
            runtime_dir=None if runtime_dir is None else Path(runtime_dir),
            add=add,
        )

    for label, path in [
        ("本地数据集", dataset_dir),
        ("训练结果", runs_dir),
        ("调试输出", PROJECT_DIR / "debug"),
        ("运行日志", PROJECT_DIR / "logs"),
    ]:
        count, truncated = _count_files(path)
        if count:
            suffix = " 个以上" if truncated else " 个"
            add("info", f"{label}存在 {count}{suffix}文件；发布脚本默认排除这些运行数据。")

    if include_env:
        env_report = tools.collect_environment_report(dataset_dir=dataset_dir, runs_dir=runs_dir)
        add(
            "info",
            f"运行环境体检: 错误 {env_report['errors']} 项，警告 {env_report['warnings']} 项。",
        )
        for level, message in env_report["lines"]:
            if level in {"warning", "error"}:
                add(level, f"运行环境: {message}")
    else:
        add("warning", "已跳过运行环境体检。")

    status = "error" if errors else ("warning" if warnings else "ok")
    add("info", f"自检完成: 错误 {errors} 项，警告 {warnings} 项。")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "lines": lines,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="YOLO 工具箱发布前自检")
    parser.add_argument("--dataset-dir", default=None, help="数据集目录，默认读取项目 dataset")
    parser.add_argument("--runs-dir", default=None, help="训练结果目录，默认读取项目 runs")
    parser.add_argument("--no-pip", action="store_true", help="跳过 pip dry-run 依赖解析")
    parser.add_argument("--release-only", action="store_true", help="只检查发布清单和骨架，跳过运行环境体检")
    parser.add_argument("--exe-dir", default=None, help="可选：轻量基础程序候选目录")
    parser.add_argument("--runtime-dir", default=None, help="可选：包含 CPU/GPU ZIP 和可信清单的目录")
    args = parser.parse_args()

    report = run_preflight(
        dataset_dir=args.dataset_dir,
        runs_dir=args.runs_dir,
        include_pip=not args.no_pip,
        include_env=not args.release_only,
        include_core_tests=not args.release_only,
        exe_dir=args.exe_dir,
        runtime_dir=args.runtime_dir,
    )
    label = {
        "info": "[信息]",
        "warning": "[警告]",
        "error": "[错误]",
    }
    for level, message in report["lines"]:
        print(f"{label.get(level, '[信息]')} {message}")
    return 0 if report["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

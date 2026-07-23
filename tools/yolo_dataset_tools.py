"""YOLO 数据集工具集 —— 后端逻辑模块

整合并扩展原 prepare_dataset.py / split_dataset.py 的核心逻辑：
  - 数据集准备（图片复制 + 可选格式转换 + 目录规范化）
  - 可配置的 train/val/test 数据集划分
  - YOLO 标签格式校验
  - 数据集统计（类别分布、标注数量等）
  - 依赖环境检测
  - 统一日志记录

设计原则：
  - 保留原脚本的复制/移动/随机划分等核心行为不变，仅在功能层面扩展。
  - 所有函数支持 emit(level, msg) 回调，便于 GUI 实时输出日志。
  - 路径基于本文件位置自动定位项目根目录，与启动位置无关。
"""

from __future__ import annotations

import logging
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from core.dataset_split import (
    ClassCoveragePolicy,
    SplitMode,
    SplitPlan,
    SplitPlanner,
    SplitPolicy,
)
from core.dataset_split_executor import (
    SplitExecutionResult,
    SplitExecutor,
)
from core.paths import ProjectPaths
from core.runtime_paths import RuntimePaths

# 模块级日志器 —— 所有函数内的 log() 内部函数都引用此变量
logger = logging.getLogger("yolo_tool")

try:
    import cv2  # type: ignore
    import numpy as _np  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# cv2 中文路径兼容层
# ---------------------------------------------------------------------------
def _cv2_imread(path, flags=cv2.IMREAD_COLOR if _HAS_CV2 else 1):
    """兼容中文路径的 cv2.imread 替代。Windows 上 cv2.imread 不支持 Unicode 路径。"""
    if not _HAS_CV2:
        return None
    buf = _np.fromfile(str(path), dtype=_np.uint8)
    return cv2.imdecode(buf, flags)


def _cv2_imwrite(path, img, ext=None, params=None):
    """兼容中文路径的 cv2.imwrite 替代。Windows 上 cv2.imwrite 不支持 Unicode 路径。"""
    if not _HAS_CV2:
        return False
    if ext is None:
        ext = Path(path).suffix if Path(path).suffix else ".png"
    result, buf = cv2.imencode(ext, img, params or [])
    if result:
        buf.tofile(str(path))
        return True
    return False

# 支持的图片扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# 可转换的目标格式
TARGET_FORMATS = ("jpg", "png")
DATASET_SPLITS = {"train", "val", "test", "unlabeled"}


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def get_runtime_paths(workspace_dir=None) -> RuntimePaths:
    """返回当前源码或冻结环境的路径集合，不创建任何目录。"""
    return RuntimePaths.from_environment(
        workspace_dir=Path(workspace_dir) if workspace_dir else None
    )


def get_project_dir() -> Path:
    """返回只读程序资源目录；保留旧函数名供现有调用方迁移。"""
    return get_runtime_paths().resource_dir


def get_dataset_dir() -> Path:
    """返回默认工作区的数据集目录；实际使用时应优先读配置。"""
    return get_runtime_paths().dataset_dir


def get_config_path() -> Path:
    """返回用户状态目录中的配置文件路径。"""
    return get_runtime_paths().config_file


def get_models_dir() -> Path:
    """返回默认工作区的模型目录。"""
    return get_runtime_paths().models_dir


def get_runs_dir() -> Path:
    """返回默认工作区的训练结果目录。"""
    return get_runtime_paths().runs_dir


def get_logs_dir() -> Path:
    """返回用户状态目录中的日志目录。"""
    return get_runtime_paths().logs_dir


def get_tensorboard_logdirs() -> Path:
    """返回 TensorBoard ASCII 别名目录。"""
    return get_runtime_paths().tensorboard_logdirs


def _resolve_project_path(value, default: Path, project_dir: Path) -> str:
    """把配置中的路径解析为当前项目可用的绝对路径。"""
    if value is None or str(value).strip() == "":
        return str(default)
    path = Path(str(value).strip())
    if not path.is_absolute():
        return str((project_dir / path).resolve())
    if path.exists():
        return str(path)

    # 旧版本把项目内目录写成了绝对路径。项目被复制后旧路径失效，
    # 仅对明确属于 dataset/runs 的路径做迁移，外部数据目录不改写。
    if path.name.lower() in {"dataset", "runs"}:
        parent_name = path.parent.name.lower()
        known_project_name = project_dir.name.lower()
        looks_like_project = (
            parent_name == known_project_name
            or any(token in parent_name for token in ("yolo", "coc", "workbuddy"))
        )
        candidate = project_dir / path.name
        if candidate.exists() and looks_like_project:
            return str(candidate.resolve())
    return str(path)


def _serialize_project_path(value, project_dir: Path) -> str:
    """项目内路径用相对值保存，外部路径保留绝对值。"""
    if value is None or str(value).strip() == "":
        return ""
    path = Path(str(value).strip())
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _normalize_last_image_positions(value, project_dir: Path, *, serialize: bool):
    if not isinstance(value, dict):
        return value
    normalized = {}
    for raw_key, raw_path in value.items():
        if serialize:
            key = _serialize_project_path(raw_key, project_dir)
            item = _serialize_project_path(raw_path, project_dir)
        else:
            key = _resolve_project_path(raw_key, Path(raw_key), project_dir)
            item = _resolve_project_path(raw_path, Path(raw_path), project_dir)
        normalized[key] = item
    return normalized


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (OSError, ValueError):
        return False


def _config_workspace(config: dict, cfg_path: Path, runtime: RuntimePaths) -> Path:
    configured = str(config.get("workspace_dir") or "").strip()
    if configured:
        return Path(configured).resolve()

    # 兼容源码绿色包和既有测试：配置仍位于项目目录时，项目本身就是工作区。
    project_dir = get_project_dir()
    if _path_is_within(cfg_path, project_dir):
        return project_dir.resolve()
    return runtime.workspace_dir.resolve()


def load_config() -> dict:
    """读取用户配置；新配置缺失时只读旧项目配置，不自动搬迁数据。"""
    runtime = get_runtime_paths()
    cfg_path = get_config_path()
    legacy_path = (
        runtime.legacy_config_file
        if runtime.frozen
        else get_project_dir() / "config" / "tool_config.json"
    )
    source_path = cfg_path
    using_legacy = False
    if not cfg_path.exists() and legacy_path.exists() and legacy_path != cfg_path:
        source_path = legacy_path
        using_legacy = True

    saved = {}
    if source_path.exists():
        try:
            with source_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
            if isinstance(value, dict):
                saved = value
        except Exception:
            pass

    if using_legacy and not str(saved.get("workspace_dir") or "").strip():
        saved["workspace_dir"] = str(get_project_dir().resolve())
    workspace_dir = _config_workspace(saved, source_path, runtime)
    dataset_dir = workspace_dir / "dataset"
    runs_dir = workspace_dir / "runs"
    models_dir = workspace_dir / "models"
    defaults = {
        "workspace_dir": str(workspace_dir),
        "dataset_dir": str(dataset_dir.resolve()),
        "runs_dir": str(runs_dir.resolve()),
        "models_dir": str(models_dir.resolve()),
        "source_dir": "",
    }
    defaults.update(saved)
    defaults["workspace_dir"] = str(workspace_dir)

    raw_dataset = Path(str(defaults.get("dataset_dir") or dataset_dir))
    raw_runs = Path(str(defaults.get("runs_dir") or runs_dir))
    raw_models = Path(str(defaults.get("models_dir") or models_dir))
    defaults["dataset_dir"] = _resolve_project_path(
        raw_dataset, dataset_dir, workspace_dir
    )
    defaults["runs_dir"] = _resolve_project_path(
        raw_runs, runs_dir, workspace_dir
    )
    defaults["models_dir"] = _resolve_project_path(
        raw_models, models_dir, workspace_dir
    )

    # 数据集从失效旧项目迁移到当前工作区时，同级 runs 也应一起修正。
    migrated_dataset = (
        Path(defaults["dataset_dir"]).resolve() == dataset_dir.resolve()
        and raw_dataset.is_absolute()
        and raw_dataset.resolve() != dataset_dir.resolve()
        and raw_dataset.name.lower() == "dataset"
        and raw_dataset.parent != workspace_dir
    )
    if migrated_dataset and raw_runs.is_absolute():
        if (
            raw_runs.name.lower() == "runs"
            and raw_runs.parent == raw_dataset.parent
            and not raw_runs.exists()
        ):
            defaults["runs_dir"] = str(runs_dir.resolve())
    if "annotator_last_images" in defaults:
        defaults["annotator_last_images"] = _normalize_last_image_positions(
            defaults["annotator_last_images"], workspace_dir, serialize=False
        )
    return defaults


def save_config(config: dict):
    """将配置原子保存到用户状态目录，工作区内部路径使用相对值。"""
    runtime = get_runtime_paths()
    cfg_path = get_config_path()
    workspace_dir = _config_workspace(config, cfg_path, runtime)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    saved = dict(config)
    saved["workspace_dir"] = str(workspace_dir)
    saved["dataset_dir"] = _serialize_project_path(
        saved.get("dataset_dir"), workspace_dir
    )
    saved["runs_dir"] = _serialize_project_path(
        saved.get("runs_dir"), workspace_dir
    )
    saved["models_dir"] = _serialize_project_path(
        saved.get("models_dir") or workspace_dir / "models", workspace_dir
    )
    if "annotator_last_images" in saved:
        saved["annotator_last_images"] = _normalize_last_image_positions(
            saved["annotator_last_images"], workspace_dir, serialize=True
        )
    temp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(cfg_path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def diagnose_dataset_dir_path(path) -> tuple[bool, str, Path]:
    """判断用户选择的数据集目录是否误选到了 images/labels 内部层级。"""
    selected = Path(path)
    name = selected.name.lower()
    parent_name = selected.parent.name.lower()
    if name in DATASET_SPLITS and parent_name in {"images", "labels"}:
        suggested = selected.parent.parent
        return (
            False,
            f"选择的是数据集内部子集目录: {selected}\n"
            f"应选择包含 images/、labels/、classes.txt 的数据集根目录。",
            suggested,
        )
    if name in {"images", "labels"}:
        suggested = selected.parent
        return (
            False,
            f"选择的是数据集内部目录: {selected}\n"
            f"应选择它的上一级数据集根目录。",
            suggested,
        )
    return True, "", selected


def diagnose_prepare_source(source, dataset_dir) -> tuple[bool, str]:
    """判断导入源目录是否误选到了当前数据集的 images/labels 内部。"""
    src = Path(source)
    base = Path(dataset_dir)
    try:
        src_resolved = src.resolve()
        base_resolved = base.resolve()
    except Exception:
        src_resolved = src.absolute()
        base_resolved = base.absolute()

    for internal_name in ("images", "labels"):
        internal = (base_resolved / internal_name)
        try:
            src_resolved.relative_to(internal)
        except ValueError:
            continue
        return (
            False,
            f"源目录位于当前数据集的 {internal_name}/ 内部: {src}\n"
            "这会把数据集复制进自己里面，产生 images/train/images/train 这类嵌套目录。\n"
            "请改选外部截图目录，或直接把新图片放入 images/train 后进行标注。",
        )
    return True, ""


def setup_logging(log_file=None, console=True):
    """配置 yolo_tool 日志器，同时输出到文件与控制台。返回日志器。"""
    root = logging.getLogger("yolo_tool")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


def check_dependencies(include_ml=True):
    """检测运行所需依赖。返回 (缺失列表, 信息字典)。"""
    missing = []
    info = {}
    checks = [
        ("PyQt5", "from PyQt5 import QtCore, QtGui, QtWidgets"),
        ("opencv-python", "import cv2"),
        ("numpy", "import numpy"),
    ]
    if include_ml:
        checks.append(("ultralytics", "from ultralytics import YOLO"))
    for name, stmt in checks:
        try:
            exec(stmt, {})  # noqa: S102
            info[name] = "已安装"
        except Exception:
            missing.append(name)
            info[name] = "未安装"
    return missing, info


def _emit_or_log(emit, level, msg):
    getattr(logger, level)(msg)
    if emit:
        emit(level, msg)


def backup_dataset_state(dataset_dir=None, reason="manual", emit=None):
    """备份数据集的标签和配置文件，返回备份目录路径。

    备份内容只包含 labels/、classes.txt、data.yaml 等小文件，不复制图片，避免备份目录膨胀。
    """
    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason) or "manual"
    backup_dir = base / "backups" / f"{timestamp}-{safe_reason}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for file_name in ("classes.txt", "data.yaml"):
        src = base / file_name
        if src.exists() and src.is_file():
            shutil.copy2(src, backup_dir / file_name)
            copied += 1

    labels_dir = base / "labels"
    if labels_dir.exists():
        dst = backup_dir / "labels"
        shutil.copytree(
            labels_dir,
            dst,
            ignore=shutil.ignore_patterns("*.cache"),
            dirs_exist_ok=True,
        )
        copied += sum(1 for p in dst.rglob("*") if p.is_file())

    manifest = backup_dir / "README.txt"
    manifest.write_text(
        "\n".join([
            "YOLO 数据集自动备份",
            f"原因: {reason}",
            f"原数据集: {base}",
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "范围: labels/、classes.txt、data.yaml；未复制图片文件。",
        ]) + "\n",
        encoding="utf-8",
    )

    if copied:
        _emit_or_log(emit, "info", f"已自动备份标签和配置: {backup_dir}")
    else:
        _emit_or_log(emit, "warning", f"未找到可备份的标签或配置，已创建备份说明: {backup_dir}")
    return backup_dir


def _parse_backup_name(name: str) -> tuple[str, str]:
    """从备份目录名中解析显示时间和原因。"""
    parts = name.split("-", 2)
    if len(parts) >= 2:
        raw_time = f"{parts[0]}-{parts[1]}"
        try:
            dt = datetime.strptime(raw_time, "%Y%m%d-%H%M%S")
            display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            display_time = raw_time
        reason = parts[2] if len(parts) >= 3 else "manual"
        return display_time, reason
    return name, "manual"


def list_dataset_backups(dataset_dir=None) -> list[dict]:
    """列出数据集备份摘要，按目录名倒序返回。"""
    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    backup_root = base / "backups"
    if not backup_root.exists() or not backup_root.is_dir():
        return []

    backups = []
    for backup_dir in sorted([p for p in backup_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        display_time, reason = _parse_backup_name(backup_dir.name)
        labels_dir = backup_dir / "labels"
        label_files = 0
        if labels_dir.exists():
            label_files = sum(1 for p in labels_dir.rglob("*.txt") if p.is_file())
        files = [p for p in backup_dir.rglob("*") if p.is_file()]
        size_bytes = sum(p.stat().st_size for p in files)
        has_classes = (backup_dir / "classes.txt").exists()
        has_data_yaml = (backup_dir / "data.yaml").exists()
        has_split_manifest = False
        manifest_path = backup_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                has_split_manifest = (
                    manifest_data.get("kind") == "smart_split"
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                has_split_manifest = False
        backups.append({
            "name": backup_dir.name,
            "path": backup_dir,
            "time": display_time,
            "reason": reason,
            "label_files": label_files,
            "file_count": len(files),
            "size_bytes": size_bytes,
            "has_classes": has_classes,
            "has_data_yaml": has_data_yaml,
            "has_labels": labels_dir.exists(),
            "has_split_manifest": has_split_manifest,
            "restore_scope": (
                "完整分组恢复"
                if has_split_manifest
                else "仅标签和配置"
            ),
        })
    return backups


def clear_label_caches(dataset_dir=None, emit=None) -> int:
    """删除 Ultralytics 标签缓存，避免恢复后训练读取旧缓存。"""
    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    label_root = base / "labels"
    removed = 0
    if not label_root.exists():
        return 0
    for cache_file in label_root.rglob("*.cache"):
        try:
            cache_file.unlink()
            removed += 1
        except Exception as exc:
            _emit_or_log(emit, "warning", f"删除缓存失败 {cache_file}: {exc}")
    if removed:
        _emit_or_log(emit, "info", f"已清理 Ultralytics 标签缓存: {removed} 个")
    return removed


def restore_dataset_backup(backup_dir, dataset_dir=None, emit=None) -> bool:
    """从自动备份恢复 labels/、classes.txt、data.yaml。

    恢复前会先备份当前状态到 before_restore，且只允许恢复当前数据集 backups/ 下的目录。
    """
    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    backup_root = (base / "backups").resolve()
    source = Path(backup_dir).resolve()

    try:
        source.relative_to(backup_root)
    except ValueError:
        _emit_or_log(emit, "error", f"拒绝恢复非当前数据集备份目录: {source}")
        return False

    if not source.exists() or not source.is_dir():
        _emit_or_log(emit, "error", f"备份目录不存在: {source}")
        return False

    source_labels = source / "labels"
    source_classes = source / "classes.txt"
    source_data_yaml = source / "data.yaml"
    if not (source_labels.exists() or source_classes.exists() or source_data_yaml.exists()):
        _emit_or_log(emit, "error", f"备份目录中没有可恢复内容: {source}")
        return False

    _emit_or_log(emit, "info", f"准备恢复备份: {source.name}")
    backup_dataset_state(base, reason="before_restore", emit=emit)

    restored = 0
    if source_labels.exists():
        target_labels = base / "labels"
        if target_labels.exists():
            shutil.rmtree(target_labels)
        shutil.copytree(source_labels, target_labels)
        restored += sum(1 for p in target_labels.rglob("*.txt") if p.is_file())
        _emit_or_log(emit, "info", f"已恢复 labels/: {restored} 个标签文件")
    else:
        _emit_or_log(emit, "warning", "备份中没有 labels/，未恢复标签目录")

    for file_name, source_file in (("classes.txt", source_classes), ("data.yaml", source_data_yaml)):
        if source_file.exists():
            shutil.copy2(source_file, base / file_name)
            _emit_or_log(emit, "info", f"已恢复 {file_name}")
        else:
            _emit_or_log(emit, "warning", f"备份中没有 {file_name}，已跳过")

    clear_label_caches(base, emit=emit)
    _emit_or_log(emit, "info", "备份恢复完成。建议训练前重新运行「格式校验」和「数据集统计」。")
    return True


def _module_version(module_name, package_name=None):
    package = package_name or module_name
    try:
        return metadata.version(package)
    except Exception:
        try:
            module = __import__(module_name)
            return getattr(module, "__version__", "已安装")
        except Exception:
            return None


def collect_environment_report(dataset_dir=None, runs_dir=None):
    """收集发布前/运行前环境体检报告。"""
    dataset_dir = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    runs_dir = Path(runs_dir) if runs_dir else get_runs_dir()
    project_dir = get_project_dir()
    lines = []
    errors = 0
    warnings = 0

    def add(level, msg):
        nonlocal errors, warnings
        if level == "error":
            errors += 1
        elif level == "warning":
            warnings += 1
        lines.append((level, msg))

    add("info", "环境体检")
    add("info", f"Python: {sys.version.split()[0]} ({sys.executable})")
    add("info", f"项目目录: {project_dir}")
    add("info", f"数据集目录: {dataset_dir}")
    add("info", f"训练结果目录: {runs_dir}")

    for name, module_name, package_name, required in [
        ("PyQt5", "PyQt5", "PyQt5", True),
        ("opencv-python", "cv2", "opencv-python", True),
        ("numpy", "numpy", "numpy", True),
        ("ultralytics", "ultralytics", "ultralytics", True),
        ("torch", "torch", "torch", False),
        ("tensorboard", "tensorboard", "tensorboard", False),
        ("PyYAML", "yaml", "PyYAML", True),
        ("Pillow", "PIL", "pillow", False),
    ]:
        if find_spec(module_name) is None:
            add("error" if required else "warning", f"{name}: 未安装")
        else:
            version = _module_version(module_name, package_name)
            add("info", f"{name}: 已安装 {version or ''}".rstrip())

    for label, path, should_exist in [
        ("项目目录", project_dir, True),
        ("数据集目录", dataset_dir, True),
        ("训练结果目录", runs_dir, False),
        ("配置目录", get_config_path().parent, False),
    ]:
        if path.exists():
            add("info", f"{label}: 存在")
        elif should_exist:
            add("error", f"{label}: 不存在 ({path})")
        else:
            add("warning", f"{label}: 尚未创建 ({path})")
        try:
            write_dir = path if path.exists() else path.parent
            if not write_dir.exists():
                add("warning", f"{label}: 父目录不存在，暂无法测试写入权限")
                continue
            probe = write_dir / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            if path.exists():
                add("info", f"{label}: 可写")
            else:
                add("info", f"{label}: 父目录可写，可在需要时创建")
        except Exception as exc:
            add("error", f"{label}: 不可写 ({exc})")

    classes = load_classes(dataset_dir)
    add("info", f"类别数量: {len(classes)}")
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        try:
            summary = data_yaml_summary(data_yaml)
            add("info", f"训练图片: {summary['train_images']} | 验证图片: {summary['val_images']} | 测试图片: {summary['test_images']}")
            if summary["train_images"] == 0:
                add("warning", "训练集为空")
            if summary["n_classes"] == 0:
                add("warning", "类别为空")
        except Exception as exc:
            add("error", f"data.yaml 读取失败: {exc}")
    else:
        add("warning", f"未找到 data.yaml: {data_yaml}")

    models = find_pretrained_models(project_dir)
    add("info", f"项目/模型目录下可用 .pt 模型: {len(models)} 个")

    if bool(getattr(sys, "frozen", False)):
        add("info", "CUDA: 由独立模型运行环境检测和管理")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                add("info", f"CUDA: 可用，GPU 数量 {torch.cuda.device_count()}")
            else:
                add("warning", "CUDA: 未检测到 GPU，将使用 CPU")
        except Exception:
            add("warning", "CUDA: 未安装 torch，无法检测")

    status = "error" if errors else ("warning" if warnings else "ok")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "lines": lines,
    }


def load_classes(dataset_dir=None) -> list:
    """读取 classes.txt 类别列表。"""
    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    classes_file = base / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []


def regenerate_data_yaml(dataset_dir, emit=None):
    """在指定数据集目录下生成/更新 data.yaml，同步 path 和 names 字段。"""
    def log(level, msg):
        getattr(logger, level)(msg)
        if emit:
            emit(level, msg)

    base = Path(dataset_dir)
    classes = load_classes(base)
    yaml_path = base / "data.yaml"

    # 读取现有 data.yaml 保留 train/val/test 字段
    existing = {}
    if yaml_path.exists():
        try:
            import yaml as _yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing = _yaml.safe_load(f) or {}
        except Exception:
            pass

    train = existing.get("train", "images/train")
    val = existing.get("val", "images/val")
    test = existing.get("test", "images/test")

    lines = [
        "# YOLOv8 数据集配置",
        f"# 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"path: {str(base).replace(chr(92), '/')}",
        f"train: {train}",
        f"val: {val}",
        f"test: {test}",
        "",
        "# 类别名称从 dataset/classes.txt 自动生成",
        "names:",
    ]
    for i, name in enumerate(classes):
        lines.append(f"  {i}: {name}")

    base.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log("info", f"data.yaml 已更新: {yaml_path}（{len(classes)} 个类别）")
    return str(yaml_path)


def iter_images(directory) -> list:
    """列出目录下所有支持格式的图片，按文件名排序。"""
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def normalize_directory(dataset_dir=None, emit=None):
    """规范化目录结构：确保 images/{train,val,test} 与 labels/{train,val,test} 存在。
    返回新建的目录路径列表。
    """
    def log(level, msg):
        getattr(logger, level)(msg)
        if emit:
            emit(level, msg)

    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    created = []
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = base / sub / split
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created.append(str(d))
    if created:
        log("info", f"已规范化目录，新建: {len(created)} 个")
        for c in created:
            log("debug", f"  {c}")
    else:
        log("debug", "目录结构已规范，无需新建。")
    return created


# ---------------------------------------------------------------------------
# 模块 1：数据集准备
# ---------------------------------------------------------------------------
def prepare_dataset(source, split="train", clean=False, convert_format=None, dataset_dir=None, emit=None):
    """数据集准备 —— 保留原 prepare_dataset.py 的复制核心逻辑并扩展。

    原核心逻辑：将源目录图片复制到 dataset/images/{split}，支持 --clean 清空目标。
    扩展：
      - 可选 convert_format（'jpg'/'png'）对图片做格式转换（需 opencv）
      - 自动规范化 images/labels 目录结构
      - 可选 dataset_dir 指定数据集存放位置（默认 {project}/dataset）

    返回 True 表示成功，False 表示失败（路径缺失等）。
    """
    def log(level, msg):
        getattr(logger, level)(msg)
        if emit:
            emit(level, msg)

    dataset_dir = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    ok, msg, suggested = diagnose_dataset_dir_path(dataset_dir)
    if not ok:
        log("error", msg)
        log("error", f"建议数据集根目录: {suggested}")
        return False
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    # 规范化目录
    normalize_directory(dataset_dir, emit=emit)

    source = Path(source)
    if not source.exists():
        log("error", f"源路径不存在: {source}")
        log("error", "请检查截图目录路径是否正确。")
        return False
    ok, msg = diagnose_prepare_source(source, dataset_dir)
    if not ok:
        log("error", msg)
        return False

    # 收集图片（保留原逻辑）
    if source.is_file():
        files = [source]
    else:
        files = iter_images(source)

    if not files:
        log("error", f"未在源目录找到图片: {source}")
        log("error", "请检查截图目录路径是否正确。")
        return False

    log("info", f"源目录: {source}")
    log("info", f"目标目录: {image_dir}")

    # clean 逻辑（保留原 prepare_dataset.py 行为）
    if clean:
        backup_dataset_state(dataset_dir, reason=f"prepare_clean_{split}", emit=emit)
        if image_dir.exists():
            for p in image_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        if label_dir.exists():
            for p in label_dir.iterdir():
                if p.is_file() and p.suffix == ".txt":
                    try:
                        p.unlink()
                    except Exception:
                        pass
        log("info", f"已清空目标目录: {image_dir} 和 {label_dir}")

    # 校验格式转换
    if convert_format:
        convert_format = convert_format.lstrip(".").lower()
        if convert_format not in TARGET_FORMATS:
            log("warning", f"不支持的目标格式 {convert_format}，跳过格式转换。")
            convert_format = None
        if convert_format and not _HAS_CV2:
            log("error", "缺少 opencv-python，无法进行图片格式转换。")
            log("error", "请在 venv 中执行: pip install opencv-python")
            return False

    copied, existed, converted, failed = 0, 0, 0, 0

    for f in files:
        if convert_format:
            # 源文件已是目标格式 → 直接复制而非无谓的读-转-写
            same_format = (f.suffix.lower() in {".jpg", ".jpeg"} and convert_format == "jpg") or \
                          (f.suffix.lower() == ".png" and convert_format == "png")
            if same_format:
                dest = image_dir / f.name
                if dest.exists() and not clean:
                    existed += 1
                    continue
                try:
                    shutil.copy2(f, dest)
                    copied += 1
                except Exception as e:
                    log("warning", f"复制失败 {f.name}: {e}")
                    failed += 1
                continue
            dest = image_dir / (f.stem + "." + convert_format)
        else:
            dest = image_dir / f.name

        if dest.exists() and not clean:
            existed += 1
            continue

        if convert_format:
            try:
                mat = _cv2_imread(f, cv2.IMREAD_UNCHANGED)
                if mat is None:
                    log("warning", f"无法读取，跳过: {f.name}")
                    failed += 1
                    continue
                if not _cv2_imwrite(dest, mat):
                    log("warning", f"写入失败，跳过: {f.name}")
                    failed += 1
                    continue
                converted += 1
            except Exception as e:
                log("warning", f"转换失败 {f.name}: {e}")
                failed += 1
        else:
            try:
                shutil.copy2(f, dest)
                copied += 1
            except Exception as e:
                log("warning", f"复制失败 {f.name}: {e}")
                failed += 1

    total_in_target = len(iter_images(image_dir))
    log("info", f"扫描到 {len(files)} 张图片")
    log("info", f"  新复制: {copied} 张")
    if convert_format:
        log("info", f"  格式转换: {converted} 张 (-> {convert_format})")
    log("info", f"  已存在(跳过): {existed} 张")
    if failed:
        log("warning", f"  失败: {failed} 张")
    log("info", f"  目标目录现有: {total_in_target} 张")
    regenerate_data_yaml(dataset_dir, emit=emit)
    log("info", "下一步：可启动标注工具进行标注，或进行数据集划分。")
    return True


# ---------------------------------------------------------------------------
# 模块 2：类别感知数据集划分
# ---------------------------------------------------------------------------
def _split_project_paths(dataset_dir=None) -> ProjectPaths:
    dataset = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    return ProjectPaths.from_runtime_paths(
        get_runtime_paths(),
        dataset_dir=dataset,
    )


def build_split_plan(
    train_ratio=0.8,
    val_ratio=0.15,
    test_ratio=0.05,
    seed=42,
    mode="repair",
    min_train_images=5,
    dataset_dir=None,
    emit=None,
) -> SplitPlan:
    """只读生成类别感知划分方案。"""
    selected_mode = mode if isinstance(mode, SplitMode) else SplitMode(mode)
    policy = SplitPolicy(
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        mode=selected_mode,
        coverage=ClassCoveragePolicy(int(min_train_images)),
    )
    plan = SplitPlanner(_split_project_paths(dataset_dir)).plan(policy)

    _emit_or_log(emit, "info", f"划分计划: {plan.plan_id[:12]}")
    _emit_or_log(emit, "info", f"有效样本: {len(plan.samples)} 张")
    _emit_or_log(emit, "info", f"计划移动: {len(plan.moves)} 张")
    _emit_or_log(
        emit,
        "info",
        "计划数量: "
        + " / ".join(
            f"{split}={count}" for split, count in plan.planned_counts
        ),
    )
    for issue in plan.blocking_issues:
        _emit_or_log(emit, "error", f"[{issue.code}] {issue.message}")
    for risk in plan.risks:
        _emit_or_log(emit, "warning", f"[{risk.code}] {risk.message}")
    return plan


def apply_split_plan(plan: SplitPlan, emit=None) -> SplitExecutionResult:
    """执行已经预览并确认的不可变划分方案。"""
    result = SplitExecutor(
        _split_project_paths(plan.dataset_dir)
    ).apply(plan)
    _emit_or_log(
        emit,
        "info",
        f"划分完成: 移动 {result.moved_pairs} 张，备份 {result.backup_dir}",
    )
    return result


def split_dataset(
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    dataset_dir=None,
    emit=None,
    mode="full",
    min_train_images=5,
):
    """兼容旧调用：生成方案后执行，GUI 和 CLI 应优先使用两步接口。"""
    try:
        plan = build_split_plan(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            mode=mode,
            min_train_images=min_train_images,
            dataset_dir=dataset_dir,
            emit=emit,
        )
        if not plan.is_executable:
            return False
        return apply_split_plan(plan, emit=emit).success
    except Exception as exc:
        _emit_or_log(emit, "error", f"数据集划分失败: {exc}")
        return False
# ---------------------------------------------------------------------------
# 模块 4：YOLO 格式校验
# ---------------------------------------------------------------------------
def validate_labels(dataset_dir=None, emit=None):
    """校验所有标签文件是否符合 YOLO 格式。

    规则：
      - 每行 5 个字段: class_id cx cy w h
      - class_id 为非负整数，且 < 类别总数
      - cx, cy, w, h 为 [0, 1] 范围内的浮点数
    返回 (总文件数, 错误数, 错误列表[(file, lineno, msg)])。
    """
    def log(level, msg):
        getattr(logger, level)(msg)
        if emit:
            emit(level, msg)

    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    label_root = base / "labels"
    if not label_root.exists():
        log("error", f"标签目录不存在: {label_root}")
        return 0, 0, []

    classes = load_classes(base)
    n_classes = len(classes)
    if n_classes == 0:
        log("warning", "classes.txt 不存在或为空，类别ID范围校验将被跳过。")

    errors = []       # 格式错误（不符合 YOLO 规范）
    warnings = []     # 信息性提示（空标签文件等合法但值得关注的情况）
    total_files = 0
    total_lines = 0
    checked_splits = []

    for split_dir in sorted([p for p in label_root.iterdir() if p.is_dir()]):
        if split_dir.name == "unlabeled":
            continue
        txt_files = sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix == ".txt" and p.name != "classes.txt"])
        if not txt_files:
            continue
        checked_splits.append(split_dir.name)
        split_errors = 0
        for tf in txt_files:
            total_files += 1
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                errors.append((str(tf), 0, f"读取失败: {e}"))
                split_errors += 1
                continue

            non_empty = [l for l in lines if l.strip()]
            if not non_empty:
                # YOLO 格式中空标签文件是合法的（表示该图片无目标对象）
                # 不计入格式错误，仅作为信息性提示
                warnings.append((str(tf), 0, "空标签文件（该图片无目标对象，YOLO 格式允许）"))
                continue

            for i, line in enumerate(lines, 1):
                s = line.strip()
                if not s:
                    continue
                total_lines += 1
                parts = s.split()
                if len(parts) != 5:
                    errors.append((str(tf), i, f"字段数应为 5，实际 {len(parts)}: {s}"))
                    split_errors += 1
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    errors.append((str(tf), i, f"类别ID不是整数: {parts[0]}"))
                    split_errors += 1
                    continue
                if cid < 0:
                    errors.append((str(tf), i, f"类别ID为负: {cid}"))
                    split_errors += 1
                    continue
                if n_classes and cid >= n_classes:
                    errors.append((str(tf), i, f"类别ID {cid} 超出范围（共 {n_classes} 类）"))
                    split_errors += 1
                    continue
                try:
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    errors.append((str(tf), i, f"坐标不是数字: {parts[1:]}"))
                    split_errors += 1
                    continue
                for name, val in (("cx", cx), ("cy", cy), ("bw", bw), ("bh", bh)):
                    if not (0.0 <= val <= 1.0):
                        errors.append((str(tf), i, f"{name}={val:.4f} 不在 [0,1] 范围"))
                        split_errors += 1
                        break
        log("info", f"[{split_dir.name}] 检查 {len(txt_files)} 个文件，错误 {split_errors} 处")

    log("info", f"共检查 {total_files} 个标签文件 / {total_lines} 行标注，发现 {len(errors)} 处格式错误")
    if warnings:
        log("info", f"另有 {len(warnings)} 个空标签文件（图片无目标对象，YOLO 格式允许）")
    if checked_splits:
        log("info", f"已检查子集: {', '.join(checked_splits)}")
    if not errors:
        log("info", "✅ 所有标签文件格式正确，符合 YOLO 规范。")
    else:
        log("warning", "⚠️ 发现格式问题，详见上方列表。建议修正后再进行训练。")
        for f, lineno, msg in errors[:50]:
            tag = f"第{lineno}行" if lineno else "整文件"
            log("warning", f"  {Path(f).name} {tag}: {msg}")
        if len(errors) > 50:
            log("warning", f"  ...另有 {len(errors) - 50} 处问题未显示，详见日志文件。")
    if warnings:
        for f, lineno, msg in warnings[:20]:
            log("info", f"  {Path(f).name}: {msg}")
        if len(warnings) > 20:
            log("info", f"  ...另有 {len(warnings) - 20} 个空标签文件未显示。")
    return total_files, len(errors), errors


# ---------------------------------------------------------------------------
# 模块 5：数据集统计
# ---------------------------------------------------------------------------
def dataset_stats(dataset_dir=None, emit=None):
    """输出数据集统计：各子集图片数/标注数、类别分布、标注总数、平均每图标注数。"""
    def log(level, msg):
        getattr(logger, level)(msg)
        if emit:
            emit(level, msg)

    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    classes = load_classes(base)
    n_classes = len(classes)

    log("info", "=" * 56)
    log("info", "YOLO 数据集统计")
    log("info", "=" * 56)
    log("info", f"数据集目录: {base}")
    log("info", f"类别数: {n_classes}")
    if classes:
        for i, name in enumerate(classes):
            log("debug", f"  {i}: {name}")

    img_root = base / "images"
    lbl_root = base / "labels"
    overall = Counter()
    split_summary = {}
    total_anns = 0
    total_imgs = 0

    splits = ["train", "val", "test", "unlabeled"]
    for split in splits:
        img_dir = img_root / split
        lbl_dir = lbl_root / split
        n_img = len(iter_images(img_dir)) if img_dir.exists() else 0
        n_lbl = 0
        cls_counter = Counter()
        if lbl_dir.exists():
            for tf in lbl_dir.iterdir():
                if not (tf.is_file() and tf.suffix == ".txt" and tf.name != "classes.txt"):
                    continue
                try:
                    with open(tf, "r", encoding="utf-8") as f:
                        for line in f:
                            s = line.strip()
                            if not s:
                                continue
                            parts = s.split()
                            if len(parts) >= 1:
                                try:
                                    cls_counter[int(parts[0])] += 1
                                    n_lbl += 1
                                except ValueError:
                                    pass
                except Exception:
                    pass
        split_summary[split] = (n_img, n_lbl, dict(cls_counter))
        overall.update(cls_counter)
        total_anns += n_lbl
        total_imgs += n_img
        log("info", f"[{split:<9}] 图片 {n_img:>4} 张 / 标注 {n_lbl:>5} 个")

    log("info", "-" * 56)
    log("info", f"合计: 图片 {total_imgs} 张 / 标注 {total_anns} 个")
    if total_imgs:
        log("info", f"平均每图标注: {total_anns / total_imgs:.2f} 个")

    log("info", "-" * 56)
    log("info", "类别分布（全部子集合计）:")
    if not overall:
        log("info", "  （暂无标注）")
    else:
        for cid in sorted(overall):
            name = classes[cid] if cid < len(classes) else f"类别{cid}"
            pct = (overall[cid] / total_anns * 100) if total_anns else 0
            log("info", f"  {cid:>2}: {name}  ->  {overall[cid]:>4} 个  ({pct:5.1f}%)")

    log("info", "=" * 56)
    return {
        "splits": split_summary,
        "overall": dict(overall),
        "total_images": total_imgs,
        "total_annotations": total_anns,
        "n_classes": n_classes,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# 模块 6：YOLO 模型训练
# ---------------------------------------------------------------------------
class _EmitStream:
    """将 stdout/stderr 输出重定向到 emit 回调，供 GUI 实时显示训练日志。"""

    def __init__(self, emit_func, level="info"):
        self.emit_func = emit_func
        self.level = level
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            # 过滤空行和纯进度条刷新字符
            if line.strip() and not line.strip().startswith("\r"):
                self.emit_func(self.level, line.rstrip())

    def flush(self):
        if self.buffer.strip():
            self.emit_func(self.level, self.buffer.rstrip())
        self.buffer = ""


def find_pretrained_models(project_dir=None, models_dir=None):
    """扫描工作区根目录和配置的模型目录，返回可供训练选择的模型。"""
    base = Path(project_dir) if project_dir else get_runtime_paths().workspace_dir
    configured_models = Path(models_dir) if models_dir else base / "models"
    models = []
    search_dirs = []
    for search_dir in (base, configured_models):
        if search_dir not in search_dirs:
            search_dirs.append(search_dir)
    for search_dir in search_dirs:
        if search_dir.exists():
            for p in search_dir.iterdir():
                if p.is_file() and p.suffix == ".pt":
                    models.append(p)
    # 去重并排序
    seen = set()
    unique = []
    for m in models:
        if str(m) not in seen:
            seen.add(str(m))
            unique.append(m)
    return sorted(unique, key=lambda p: p.name)


def detect_devices():
    """检测可用训练设备。返回 (display_label, device_value) 列表。

    display_label: 下拉框显示的文字（含 GPU 名称）
    device_value:  传给 ultralytics 的实际值（""=自动, "cpu", "0", "0,1"）
    """
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            devices = []
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                # 截断过长的 GPU 名称
                if len(name) > 40:
                    name = name[:37] + "..."
                suffix = "（推荐）" if i == 0 else ""
                devices.append((f"GPU {i}: {name}{suffix}", str(i)))
            if n >= 2:
                ids = ",".join(str(i) for i in range(n))
                devices.append((f"全部 {n} 块 GPU", ids))
            devices.extend([
                ("自动选择（由 Ultralytics 决定）", ""),
                ("CPU（兼容模式，速度较慢）", "cpu"),
            ])
        else:
            devices = [
                ("CPU（未检测到 GPU）", "cpu"),
                ("自动选择（当前将使用 CPU）", ""),
            ]
    except ImportError:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                creationflags=creation_flags,
                check=False,
            )
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            names = []
        if names:
            devices = []
            for index, name in enumerate(names):
                if len(name) > 40:
                    name = name[:37] + "..."
                suffix = "（推荐）" if index == 0 else ""
                devices.append((f"GPU {index}: {name}{suffix}", str(index)))
            if len(names) >= 2:
                devices.append(
                    (f"全部 {len(names)} 块 GPU", ",".join(str(i) for i in range(len(names))))
                )
            devices.extend(
                [
                    ("自动选择（由模型运行环境决定）", ""),
                    ("CPU（兼容模式，速度较慢）", "cpu"),
                ]
            )
        else:
            devices = [
                ("CPU（未检测到 GPU）", "cpu"),
                ("自动选择（当前将使用 CPU）", ""),
            ]
    return devices


def find_data_configs(dataset_dir=None):
    """扫描数据集目录下的 .yaml 配置文件，返回可供训练选择的配置列表。"""
    base = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    configs = []
    if base.exists():
        for p in sorted(base.glob("*.yaml")) + sorted(base.glob("*.yml")):
            if p.name != "args.yaml":  # 排除训练产物
                configs.append(p)
    return configs


def resolve_data_yaml(data_yaml):
    """解析 YOLO data.yaml，返回配置、数据集根目录和 names 列表。"""
    data_path = Path(data_yaml)
    cfg = {}
    if data_path.exists():
        try:
            import yaml as _yaml
            with open(data_path, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

    raw_root = cfg.get("path")
    if raw_root:
        root = Path(raw_root)
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()
    else:
        root = data_path.parent

    names = cfg.get("names", [])
    if isinstance(names, dict):
        def _name_sort_key(key):
            text = str(key)
            return (0, int(text)) if text.isdigit() else (1, text)

        names = [names[k] for k in sorted(names, key=_name_sort_key)]
    elif not isinstance(names, list):
        names = []

    if not names:
        names = load_classes(root)

    return cfg, root, names


def _resolve_split_dir(data_root, split_value, default_rel):
    """根据 data.yaml 的 split 字段解析图片目录。"""
    rel = split_value or default_rel
    if isinstance(rel, (list, tuple)):
        rel = rel[0] if rel else default_rel
    path = Path(str(rel))
    return path if path.is_absolute() else Path(data_root) / path


def data_yaml_summary(data_yaml):
    """返回 data.yaml 指向的数据集概要，用于训练前检查和 GUI 展示。"""
    cfg, data_root, names = resolve_data_yaml(data_yaml)
    train_dir = _resolve_split_dir(data_root, cfg.get("train"), "images/train")
    val_dir = _resolve_split_dir(data_root, cfg.get("val"), "images/val")
    test_dir = _resolve_split_dir(data_root, cfg.get("test"), "images/test")
    return {
        "config": cfg,
        "root": data_root,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "train_images": len(iter_images(train_dir)),
        "val_images": len(iter_images(val_dir)),
        "test_images": len(iter_images(test_dir)),
        "n_classes": len(names),
        "classes": names,
    }


def train_model(model_path, data_yaml=None, epochs=100, batch=8, imgsz=640,
                device="", project=None, name="coc_detect", patience=20,
                resume=False, emit=None):
    """启动 YOLO 训练 —— 在 Worker 线程中调用，通过 emit 实时推送日志。

    参数:
      model_path:  预训练模型路径（.pt）
      data_yaml:   数据集配置文件（默认 dataset/data.yaml）
      epochs:      训练轮数
      batch:       批次大小
      imgsz:       输入图像尺寸
      device:      训练设备（""=自动, "cpu", "0", "0,1"）
      project:     结果保存目录（默认 runs/）
      name:        本次训练名称
      patience:    早停耐心值
      resume:      是否从上次中断处继续训练
      emit:        日志回调 emit(level, msg)

    返回 True 表示训练成功，False 表示失败。
    """
    def log(level, msg):
        getattr(logger, level)(msg)
        if emit:
            emit(level, msg)

    # ── 前置检查 ──
    try:
        from ultralytics import YOLO
    except ImportError:
        log("error", "缺少 ultralytics，请安装: pip install ultralytics")
        return False

    model_path = Path(model_path)
    if not model_path.exists():
        log("error", f"模型文件不存在: {model_path}")
        return False

    if data_yaml is None:
        data_yaml = get_dataset_dir() / "data.yaml"
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        log("error", f"数据配置文件不存在: {data_yaml}")
        return False

    if project is None:
        project = str(get_runs_dir())

    # ── 数据集概要 ──
    summary = data_yaml_summary(data_yaml)
    train_imgs = summary["train_images"]
    val_imgs = summary["val_images"]
    log("info", "=" * 50)
    log("info", "YOLO 模型训练")
    log("info", "=" * 50)
    log("info", f"预训练模型: {model_path}")
    log("info", f"数据配置:   {data_yaml}")
    log("info", f"数据集根目录: {summary['root']}")
    log("info", f"训练集: {train_imgs} 张 | 验证集: {val_imgs} 张 | 类别数: {summary['n_classes']}")
    log("info", f"参数: epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device or '自动'}")
    if resume:
        log("info", "模式: 继续上次训练（resume=True）")
    log("info", "-" * 50)

    if train_imgs == 0:
        log("error", "训练集为空，请先准备数据集并标注。")
        return False
    if val_imgs == 0:
        log("warning", "验证集为空，训练时将用训练集做验证（建议先划分数据集）。")

    # ── 启动训练 ──
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stream = _EmitStream(log, "info")
    sys.stdout = stream
    sys.stderr = stream

    try:
        model = YOLO(str(model_path))
        train_kwargs = dict(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=name,
            patience=patience,
            save=True,
            verbose=True,
            workers=0,
        )
        if device:
            train_kwargs["device"] = device
        if resume:
            train_kwargs["resume"] = True

        model.train(**train_kwargs)

        # flush 残余输出
        stream.flush()

        log("info", "-" * 50)
        log("info", "✅ 训练完成！")

        # 定位最佳模型
        best_pt = Path(project) / name / "weights" / "best.pt"
        last_pt = Path(project) / name / "weights" / "last.pt"
        if best_pt.exists():
            log("info", f"最佳模型: {best_pt}")
        if last_pt.exists():
            log("info", f"末轮模型: {last_pt}")
        results_dir = Path(project) / name
        if results_dir.exists():
            log("info", f"结果目录: {results_dir}")
        log("info", "可使用「启动 TensorBoard」查看训练曲线，或用最佳模型进行推理测试。")
        return True
    except KeyboardInterrupt:
        log("warning", "训练被用户中断。")
        return False
    except Exception as e:
        stream.flush()
        log("error", f"训练失败: {e}")
        logger.exception("train error")
        return False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == "__main__":
    # 命令行直接运行：输出统计并校验
    setup_logging(console=True)
    print("=== 依赖检测 ===")
    missing, info = check_dependencies()
    for k, v in info.items():
        print(f"  {k}: {v}")
    if missing:
        print(f"缺失依赖: {missing}")
    print()
    dataset_stats(emit=lambda lvl, msg: print(msg))
    print()
    validate_labels(emit=lambda lvl, msg: print(msg))

"""在独立进程中运行 YOLO 验证，并输出统一评估任务事件。"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import traceback
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.model_evaluation import (  # noqa: E402
    ClassEvaluationMetrics,
    EvaluationMetrics,
    EvaluationSession,
    ModelEvaluationReport,
    compare_model_reports,
    save_evaluation_session,
    scan_evaluation_dataset,
)
from core.runtime_paths import RuntimePaths  # noqa: E402
from core.task_protocol import TaskEventEmitter  # noqa: E402


PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")
RUNTIME_PATHS = RuntimePaths.from_environment()
PROJECT_DIR = RUNTIME_PATHS.resource_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 模型评估子进程")
    parser.add_argument("--task-id", default=None, help="统一任务 ID")
    parser.add_argument("--model", required=True, help="候选模型 .pt 路径")
    parser.add_argument("--baseline", default="", help="可选基准模型 .pt 路径")
    parser.add_argument("--data", required=True, help="数据集 data.yaml")
    parser.add_argument("--split", default="test", help="评估分组")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--device", default="", help="评估设备")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--batch", type=int, default=16, help="验证批次大小")
    parser.add_argument(
        "--output",
        default="",
        help="evaluation.json 路径，默认写入 runs/evaluations",
    )
    parser.add_argument(
        "--project",
        default=str(RUNTIME_PATHS.runs_dir / "evaluations"),
        help="评估产物根目录",
    )
    return parser


def _default_output(project: Path, model_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path.stem) or "model"
    return project / f"{stamp}-{safe_name}" / "evaluation.json"


def _metric_from_box(box: Any, metric_name: str, default: float = 0.0) -> float:
    value = getattr(box, metric_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _standardize_metrics(metrics: Any, snapshot) -> EvaluationMetrics:
    results = getattr(metrics, "results_dict", {}) or {}

    def result(*keys: str) -> float:
        for key in keys:
            if key in results:
                try:
                    return float(results[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    box = getattr(metrics, "box", None)
    precision = result("metrics/precision(B)", "metrics/precision")
    recall = result("metrics/recall(B)", "metrics/recall")
    map50 = result("metrics/mAP50(B)", "metrics/mAP50")
    map50_95 = result("metrics/mAP50-95(B)", "metrics/mAP50-95")
    if box is not None:
        precision = _metric_from_box(box, "mp", precision)
        recall = _metric_from_box(box, "mr", recall)
        map50 = _metric_from_box(box, "map50", map50)
        map50_95 = _metric_from_box(box, "map", map50_95)

    class_precision = getattr(box, "p", []) if box is not None else []
    class_recall = getattr(box, "r", []) if box is not None else []
    class_map50 = getattr(box, "ap50", []) if box is not None else []
    class_map = getattr(box, "ap", []) if box is not None else []

    raw_class_indices = (
        getattr(box, "ap_class_index", None) if box is not None else None
    )
    if raw_class_indices is None:
        raw_class_indices = range(len(class_precision))
    metric_positions = {}
    for position, raw_class_id in enumerate(raw_class_indices):
        if hasattr(raw_class_id, "item"):
            raw_class_id = raw_class_id.item()
        try:
            metric_positions[int(raw_class_id)] = position
        except (TypeError, ValueError):
            continue

    def item_value(values, class_id, instances):
        if instances <= 0:
            return 0.0
        index = metric_positions.get(class_id)
        if index is None:
            return 0.0
        try:
            value = values[index]
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except (IndexError, TypeError, ValueError):
            return 0.0

    class_items = []
    for class_id, name in enumerate(snapshot.class_names):
        instances = (
            snapshot.class_target_counts[class_id]
            if class_id < len(snapshot.class_target_counts)
            else 0
        )
        class_items.append(
            ClassEvaluationMetrics(
                class_id=class_id,
                name=name,
                instances=instances,
                precision=item_value(class_precision, class_id, instances),
                recall=item_value(class_recall, class_id, instances),
                map50=item_value(class_map50, class_id, instances),
                map50_95=item_value(class_map, class_id, instances),
            )
        )
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        image_count=snapshot.image_count,
        target_count=snapshot.target_count,
        classes=tuple(class_items),
    )


@contextlib.contextmanager
def _disable_ultralytics_cache():
    """阻止验证过程刷新标签缓存，避免评估写入真实数据目录。"""
    try:
        from ultralytics.data import utils as data_utils
    except ImportError:
        yield
        return

    def blocked(*args, **kwargs):
        return None

    patched = []
    original = getattr(data_utils, "save_dataset_cache_file", None)
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith(
            "ultralytics.data"
        ):
            continue
        if getattr(module, "save_dataset_cache_file", None) is not None:
            patched.append((module, module.save_dataset_cache_file))
            module.save_dataset_cache_file = blocked
    if original is not None and not any(module is data_utils for module, _ in patched):
        patched.append((data_utils, original))
        data_utils.save_dataset_cache_file = blocked
    try:
        yield
    finally:
        for module, value in patched:
            module.save_dataset_cache_file = value


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device: str,
    output_dir: Path,
    snapshot,
    *,
    conf: float = 0.001,
    batch: int = 16,
    model_factory: Callable[[Path], Any] | None = None,
    emit: Callable[[str, str], None] | None = None,
) -> ModelEvaluationReport:
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    if emit is not None:
        emit("info", f"加载模型: {model_path}")
    model = model_factory(Path(model_path))
    with _disable_ultralytics_cache():
        if emit is not None:
            emit("info", f"开始评估 {split} 分组，共 {snapshot.image_count} 张图片")
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            device=device,
            conf=conf,
            batch=batch,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            plots=True,
            save_json=False,
            verbose=False,
        )
    return ModelEvaluationReport(
        model_path=Path(model_path),
        data_yaml=Path(data_yaml).resolve(),
        split=split,
        imgsz=imgsz,
        device=str(device),
        dataset=snapshot,
        metrics=_standardize_metrics(metrics, snapshot),
        output_dir=Path(output_dir),
    )


def main(
    argv=None,
    evaluate_func=None,
    snapshot_func=None,
    stream=None,
) -> int:
    args = _build_parser().parse_args(argv)
    task_id = args.task_id or str(uuid4())
    emitter = TaskEventEmitter(task_id, "evaluation", stream=stream)
    project = Path(args.project)
    if not project.is_absolute():
        project = RUNTIME_PATHS.workspace_dir / project
    output = Path(args.output) if args.output else None
    if output is None:
        output = _default_output(project, Path(args.model))
    if not output.is_absolute():
        output = RUNTIME_PATHS.workspace_dir / output
    output = output.resolve()
    output_dir = output.parent

    emitter.started(
        "评估任务已启动",
        {
            "model": args.model,
            "baseline": args.baseline,
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "device": args.device,
            "output": str(output),
        },
    )
    emitter.progress(0.0, "准备数据快照")

    def emit(level: str, message: str):
        emitter.log(message, level=level)
        match = PROGRESS_PATTERN.search(str(message))
        if match:
            current, total = (int(value) for value in match.groups())
            if total > 0:
                emitter.progress(
                    min(0.95, current / total),
                    f"评估进度 {current}/{total}",
                    {"current": current, "total": total},
                )

    try:
        snapshot_loader = snapshot_func or scan_evaluation_dataset
        snapshot = snapshot_loader(Path(args.data), args.split)
        initial_fingerprint = snapshot.fingerprint
        backend = evaluate_func or evaluate_model
        candidate = backend(
            model_path=Path(args.model),
            data_yaml=Path(args.data),
            split=args.split,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=output_dir / "candidate",
            snapshot=snapshot,
            conf=args.conf,
            batch=args.batch,
            emit=emit,
        )
        current_snapshot = scan_evaluation_dataset(Path(args.data), args.split)
        if current_snapshot.fingerprint != initial_fingerprint:
            raise RuntimeError("评估期间数据指纹发生变化，已拒绝保存结果")

        baseline = None
        if args.baseline.strip():
            baseline = backend(
                model_path=Path(args.baseline),
                data_yaml=Path(args.data),
                split=args.split,
                imgsz=args.imgsz,
                device=args.device,
                output_dir=output_dir / "baseline",
                snapshot=snapshot,
                conf=args.conf,
                batch=args.batch,
                emit=emit,
            )
            current_snapshot = scan_evaluation_dataset(Path(args.data), args.split)
            if current_snapshot.fingerprint != initial_fingerprint:
                raise RuntimeError("基准评估期间数据指纹发生变化，已拒绝保存结果")

        comparison = compare_model_reports(candidate, baseline) if baseline else None
        session = EvaluationSession(candidate, baseline, comparison)
        save_evaluation_session(session, output)
        emitter.progress(1.0, "评估报告已保存", {"output": str(output)})
        emitter.result(
            "模型评估完成",
            {
                "output": str(output),
                "image_count": candidate.metrics.image_count,
                "target_count": candidate.metrics.target_count,
                "verdict": comparison.verdict if comparison else "无基准",
            },
        )
        return 0
    except KeyboardInterrupt:
        emitter.cancelled("评估任务已取消")
        return 130
    except BaseException as exc:
        emitter.failed(
            f"模型评估失败: {exc}",
            {"traceback": traceback.format_exc()},
        )
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

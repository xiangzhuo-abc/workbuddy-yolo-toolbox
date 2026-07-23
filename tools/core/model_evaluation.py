"""模型评估所需的只读数据快照、标准报告和比较规则。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .annotation_service import load_annotation_document


IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class EvaluationDatasetSnapshot:
    data_yaml: Path
    dataset_dir: Path
    split: str
    class_names: tuple[str, ...]
    image_paths: tuple[Path, ...]
    label_paths: tuple[Path, ...]
    image_count: int
    target_count: int
    fingerprint: str
    class_target_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class ClassEvaluationMetrics:
    class_id: int
    name: str
    instances: int
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    map50: float
    map50_95: float
    image_count: int
    target_count: int
    classes: tuple[ClassEvaluationMetrics, ...] = ()


@dataclass(frozen=True)
class MetricDelta:
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class ClassMetricDelta:
    class_id: int
    name: str
    instances: int
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class ModelEvaluationReport:
    model_path: Path
    data_yaml: Path
    split: str
    imgsz: int
    device: str
    dataset: EvaluationDatasetSnapshot
    metrics: EvaluationMetrics
    output_dir: Path


@dataclass(frozen=True)
class ModelComparison:
    comparable: bool
    verdict: str
    reason: str
    low_sample: bool
    delta: MetricDelta
    class_deltas: tuple[ClassMetricDelta, ...] = ()


@dataclass(frozen=True)
class EvaluationSession:
    candidate: ModelEvaluationReport
    baseline: ModelEvaluationReport | None = None
    comparison: ModelComparison | None = None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_dataset_dir(data_yaml: Path, raw_path: Any) -> Path:
    configured = Path(str(raw_path or "."))
    if not configured.is_absolute():
        configured = data_yaml.parent / configured
    return configured.resolve()


def _load_data_config(data_yaml: Path) -> tuple[Path, Mapping[str, Any]]:
    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_yaml}")
    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取数据配置: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("数据配置根节点必须是对象")
    return data_yaml, payload


def _class_names(payload: Mapping[str, Any], dataset_dir: Path) -> tuple[str, ...]:
    names = payload.get("names")
    if isinstance(names, Mapping):
        numeric = sorted(
            ((int(key), str(value)) for key, value in names.items()),
            key=lambda item: item[0],
        )
        if numeric and [key for key, _ in numeric] != list(range(len(numeric))):
            raise ValueError("数据配置中的类别 ID 必须从 0 连续编号")
        if numeric:
            return tuple(value for _, value in numeric)
    if isinstance(names, (list, tuple)):
        return tuple(str(value) for value in names)

    classes_file = dataset_dir / "classes.txt"
    if classes_file.is_file():
        return tuple(
            line.strip()
            for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    raise ValueError("数据配置没有 names，且数据集缺少 classes.txt")


def _split_dir(dataset_dir: Path, raw_split: Any) -> Path:
    if not isinstance(raw_split, (str, Path)) or not str(raw_split).strip():
        raise ValueError("评估分组未在 data.yaml 中配置")
    path = Path(str(raw_split))
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()


def _fingerprint_file(digest: Any, root: Path, path: Path) -> None:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = path.resolve().as_posix()
    digest.update(f"FILE\0{relative}\0".encode("utf-8"))
    if not path.is_file():
        digest.update(b"MISSING\0")
        return
    stat = path.stat()
    digest.update(f"{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def scan_evaluation_dataset(
    data_yaml: Path,
    split: str,
) -> EvaluationDatasetSnapshot:
    """只读扫描评估分组并返回包含数据指纹的快照。"""
    data_yaml, payload = _load_data_config(Path(data_yaml))
    dataset_dir = _resolve_dataset_dir(data_yaml, payload.get("path"))
    class_names = _class_names(payload, dataset_dir)
    split = str(split).strip()
    if not split:
        raise ValueError("评估分组不能为空")
    image_dir = _split_dir(dataset_dir, payload.get(split))
    if not image_dir.is_dir():
        raise FileNotFoundError(f"评估图片目录不存在: {image_dir}")

    label_dir = dataset_dir / "labels" / split
    image_paths = tuple(
        sorted(
            (
                path.resolve()
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    )
    label_paths = tuple(label_dir / f"{path.stem}.txt" for path in image_paths)
    target_count = 0
    for label_path in label_paths:
        document = load_annotation_document(
            label_path,
            class_count=len(class_names),
        )
        target_count += len(document.boxes)

    class_target_counts = [0] * len(class_names)
    for label_path in label_paths:
        document = load_annotation_document(
            label_path,
            class_count=len(class_names),
        )
        for box in document.boxes:
            class_target_counts[box.class_id] += 1

    digest = hashlib.sha256()
    digest.update(f"SPLIT\0{split}\0".encode("utf-8"))
    digest.update(
        json.dumps(class_names, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(b"\0")
    _fingerprint_file(digest, dataset_dir, data_yaml)
    for image_path, label_path in zip(image_paths, label_paths):
        _fingerprint_file(digest, dataset_dir, image_path)
        _fingerprint_file(digest, dataset_dir, label_path)

    return EvaluationDatasetSnapshot(
        data_yaml=data_yaml,
        dataset_dir=dataset_dir,
        split=split,
        class_names=class_names,
        image_paths=image_paths,
        label_paths=label_paths,
        image_count=len(image_paths),
        target_count=target_count,
        fingerprint=digest.hexdigest(),
        class_target_counts=tuple(class_target_counts),
    )


def _metric_delta(
    candidate: EvaluationMetrics,
    baseline: EvaluationMetrics,
) -> MetricDelta:
    return MetricDelta(
        precision=candidate.precision - baseline.precision,
        recall=candidate.recall - baseline.recall,
        map50=candidate.map50 - baseline.map50,
        map50_95=candidate.map50_95 - baseline.map50_95,
    )


def _class_deltas(
    candidate: EvaluationMetrics,
    baseline: EvaluationMetrics,
) -> tuple[ClassMetricDelta, ...]:
    baseline_by_id = {item.class_id: item for item in baseline.classes}
    deltas = []
    for item in candidate.classes:
        other = baseline_by_id.get(item.class_id)
        if other is None or item.name != other.name:
            continue
        deltas.append(
            ClassMetricDelta(
                class_id=item.class_id,
                name=item.name,
                instances=item.instances,
                precision=item.precision - other.precision,
                recall=item.recall - other.recall,
                map50=item.map50 - other.map50,
                map50_95=item.map50_95 - other.map50_95,
            )
        )
    return tuple(deltas)


def compare_model_reports(
    candidate: ModelEvaluationReport,
    baseline: ModelEvaluationReport,
) -> ModelComparison:
    """按固定规则比较两个使用同一数据协议的评估报告。"""
    zero = MetricDelta(0.0, 0.0, 0.0, 0.0)
    if candidate.split != baseline.split:
        return ModelComparison(False, "无法比较", "评估分组不一致", False, zero)
    if candidate.imgsz != baseline.imgsz or candidate.device != baseline.device:
        return ModelComparison(False, "无法比较", "评估参数不一致", False, zero)
    if candidate.dataset.class_names != baseline.dataset.class_names:
        return ModelComparison(False, "无法比较", "类别定义不一致，不能比较", False, zero)
    if candidate.data_yaml.resolve() != baseline.data_yaml.resolve():
        return ModelComparison(False, "无法比较", "data.yaml 不一致，不能比较", False, zero)
    if candidate.dataset.fingerprint != baseline.dataset.fingerprint:
        return ModelComparison(False, "无法比较", "数据指纹不一致，不能比较", False, zero)

    delta = _metric_delta(candidate.metrics, baseline.metrics)
    class_deltas = _class_deltas(candidate.metrics, baseline.metrics)
    low_sample = min(
        candidate.dataset.image_count,
        baseline.dataset.image_count,
    ) < 20
    if low_sample:
        return ModelComparison(
            True,
            "仅供参考",
            "测试图片少于 20 张，差值仅供参考",
            True,
            delta,
            class_deltas,
        )
    if delta.map50_95 >= 0.01 and delta.recall > -0.02 and delta.map50 > -0.02:
        return ModelComparison(
            True,
            "推荐候选",
            "mAP50-95 提升且召回率、mAP50 未明显下降",
            False,
            delta,
            class_deltas,
        )
    if delta.map50_95 <= -0.01 or delta.recall <= -0.03:
        return ModelComparison(
            True,
            "保持基准",
            "mAP50-95 或召回率明显下降",
            False,
            delta,
            class_deltas,
        )
    return ModelComparison(
        True,
        "基本持平",
        "主要指标变化未达到替换阈值",
        False,
        delta,
        class_deltas,
    )


def _path(value: Any) -> Path:
    return Path(str(value))


def _snapshot_to_dict(snapshot: EvaluationDatasetSnapshot) -> dict[str, Any]:
    return {
        "data_yaml": str(snapshot.data_yaml),
        "dataset_dir": str(snapshot.dataset_dir),
        "split": snapshot.split,
        "class_names": list(snapshot.class_names),
        "image_paths": [str(path) for path in snapshot.image_paths],
        "label_paths": [str(path) for path in snapshot.label_paths],
        "image_count": snapshot.image_count,
        "target_count": snapshot.target_count,
        "fingerprint": snapshot.fingerprint,
        "class_target_counts": list(snapshot.class_target_counts),
    }


def _snapshot_from_dict(data: Mapping[str, Any]) -> EvaluationDatasetSnapshot:
    return EvaluationDatasetSnapshot(
        data_yaml=_path(data["data_yaml"]),
        dataset_dir=_path(data["dataset_dir"]),
        split=str(data["split"]),
        class_names=tuple(str(value) for value in data.get("class_names", [])),
        image_paths=tuple(_path(value) for value in data.get("image_paths", [])),
        label_paths=tuple(_path(value) for value in data.get("label_paths", [])),
        image_count=_as_int(data.get("image_count")),
        target_count=_as_int(data.get("target_count")),
        fingerprint=str(data["fingerprint"]),
        class_target_counts=tuple(
            _as_int(value) for value in data.get("class_target_counts", [])
        ),
    )


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "image_count": metrics.image_count,
        "target_count": metrics.target_count,
        "classes": [
            {
                "class_id": item.class_id,
                "name": item.name,
                "instances": item.instances,
                "precision": item.precision,
                "recall": item.recall,
                "map50": item.map50,
                "map50_95": item.map50_95,
            }
            for item in metrics.classes
        ],
    }


def _metrics_from_dict(data: Mapping[str, Any]) -> EvaluationMetrics:
    return EvaluationMetrics(
        precision=_as_float(data.get("precision")),
        recall=_as_float(data.get("recall")),
        map50=_as_float(data.get("map50")),
        map50_95=_as_float(data.get("map50_95")),
        image_count=_as_int(data.get("image_count")),
        target_count=_as_int(data.get("target_count")),
        classes=tuple(
            ClassEvaluationMetrics(
                class_id=_as_int(item.get("class_id")),
                name=str(item.get("name", "")),
                instances=_as_int(item.get("instances")),
                precision=_as_float(item.get("precision")),
                recall=_as_float(item.get("recall")),
                map50=_as_float(item.get("map50")),
                map50_95=_as_float(item.get("map50_95")),
            )
            for item in data.get("classes", [])
        ),
    )


def _report_to_dict(report: ModelEvaluationReport) -> dict[str, Any]:
    return {
        "model_path": str(report.model_path),
        "data_yaml": str(report.data_yaml),
        "split": report.split,
        "imgsz": report.imgsz,
        "device": report.device,
        "dataset": _snapshot_to_dict(report.dataset),
        "metrics": _metrics_to_dict(report.metrics),
        "output_dir": str(report.output_dir),
    }


def _report_from_dict(data: Mapping[str, Any]) -> ModelEvaluationReport:
    return ModelEvaluationReport(
        model_path=_path(data["model_path"]),
        data_yaml=_path(data["data_yaml"]),
        split=str(data["split"]),
        imgsz=_as_int(data.get("imgsz")),
        device=str(data.get("device", "")),
        dataset=_snapshot_from_dict(data["dataset"]),
        metrics=_metrics_from_dict(data["metrics"]),
        output_dir=_path(data["output_dir"]),
    )


def _delta_to_dict(delta: MetricDelta) -> dict[str, float]:
    return {
        "precision": delta.precision,
        "recall": delta.recall,
        "map50": delta.map50,
        "map50_95": delta.map50_95,
    }


def _delta_from_dict(data: Mapping[str, Any]) -> MetricDelta:
    return MetricDelta(
        precision=_as_float(data.get("precision")),
        recall=_as_float(data.get("recall")),
        map50=_as_float(data.get("map50")),
        map50_95=_as_float(data.get("map50_95")),
    )


def _class_delta_to_dict(delta: ClassMetricDelta) -> dict[str, Any]:
    return {
        "class_id": delta.class_id,
        "name": delta.name,
        "instances": delta.instances,
        "precision": delta.precision,
        "recall": delta.recall,
        "map50": delta.map50,
        "map50_95": delta.map50_95,
    }


def _class_delta_from_dict(data: Mapping[str, Any]) -> ClassMetricDelta:
    return ClassMetricDelta(
        class_id=_as_int(data.get("class_id")),
        name=str(data.get("name", "")),
        instances=_as_int(data.get("instances")),
        precision=_as_float(data.get("precision")),
        recall=_as_float(data.get("recall")),
        map50=_as_float(data.get("map50")),
        map50_95=_as_float(data.get("map50_95")),
    )


def _comparison_to_dict(comparison: ModelComparison | None) -> dict[str, Any] | None:
    if comparison is None:
        return None
    return {
        "comparable": comparison.comparable,
        "verdict": comparison.verdict,
        "reason": comparison.reason,
        "low_sample": comparison.low_sample,
        "delta": _delta_to_dict(comparison.delta),
        "class_deltas": [
            _class_delta_to_dict(item) for item in comparison.class_deltas
        ],
    }


def _comparison_from_dict(data: Mapping[str, Any] | None) -> ModelComparison | None:
    if data is None:
        return None
    return ModelComparison(
        comparable=bool(data.get("comparable")),
        verdict=str(data.get("verdict", "")),
        reason=str(data.get("reason", "")),
        low_sample=bool(data.get("low_sample")),
        delta=_delta_from_dict(data.get("delta", {})),
        class_deltas=tuple(
            _class_delta_from_dict(item) for item in data.get("class_deltas", [])
        ),
    )


def _session_to_dict(session: EvaluationSession) -> dict[str, Any]:
    return {
        "candidate": _report_to_dict(session.candidate),
        "baseline": _report_to_dict(session.baseline) if session.baseline else None,
        "comparison": _comparison_to_dict(session.comparison),
    }


def _session_from_dict(data: Mapping[str, Any]) -> EvaluationSession:
    baseline_data = data.get("baseline")
    return EvaluationSession(
        candidate=_report_from_dict(data["candidate"]),
        baseline=_report_from_dict(baseline_data) if baseline_data else None,
        comparison=_comparison_from_dict(data.get("comparison")),
    )


def save_evaluation_session(session: EvaluationSession, path: Path) -> None:
    """以原子替换方式保存评估会话。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        _session_to_dict(session),
        ensure_ascii=False,
        indent=2,
    )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_evaluation_session(path: Path) -> EvaluationSession:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取评估报告: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("评估报告根节点必须是对象")
    try:
        return _session_from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"评估报告格式无效: {exc}") from exc

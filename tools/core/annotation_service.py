from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .issues import Issue, IssueSeverity


AUTO_DEDUPE_IOU_THRESHOLD = 0.65
AUTO_DEDUPE_CONTAINMENT_THRESHOLD = 0.90
AUTO_DEDUPE_MIN_AREA_RATIO = 0.55

PixelBox = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class NormalizedBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class AnnotationDocument:
    label_path: Path
    boxes: tuple[NormalizedBox, ...]
    issues: tuple[Issue, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is IssueSeverity.ERROR for issue in self.issues)


def _label_issue(code: str, message: str, label_path: Path) -> Issue:
    return Issue(
        code=code,
        severity=IssueSeverity.ERROR,
        message=message,
        path=label_path,
        suggested_action="在标注工具中检查并重新保存该标签",
    )


def load_annotation_document(
    label_path: Path, class_count: int | None = None
) -> AnnotationDocument:
    label_path = Path(label_path)
    if not label_path.exists():
        return AnnotationDocument(label_path, (), ())

    try:
        content = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        issue = _label_issue(
            "label.read_error",
            f"无法读取标签文件: {exc}",
            label_path,
        )
        return AnnotationDocument(label_path, (), (issue,))

    boxes: list[NormalizedBox] = []
    issues: list[Issue] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            issues.append(
                _label_issue(
                    "label.field_count",
                    f"第 {line_number} 行应包含 5 个字段，实际为 {len(fields)} 个",
                    label_path,
                )
            )
            continue

        line_issues: list[Issue] = []
        class_id: int | None = None
        try:
            class_id = int(fields[0])
        except ValueError:
            line_issues.append(
                _label_issue(
                    "label.invalid_class_id",
                    f"第 {line_number} 行类别 ID 不是整数",
                    label_path,
                )
            )
        else:
            if class_id < 0:
                line_issues.append(
                    _label_issue(
                        "label.invalid_class_id",
                        f"第 {line_number} 行类别 ID 不能为负数",
                        label_path,
                    )
                )
            elif class_count is not None and class_id >= class_count:
                line_issues.append(
                    _label_issue(
                        "label.class_out_of_range",
                        f"第 {line_number} 行类别 ID {class_id} 超出类别范围",
                        label_path,
                    )
                )

        values: tuple[float, float, float, float] | None = None
        try:
            values = tuple(float(value) for value in fields[1:])
        except ValueError:
            line_issues.append(
                _label_issue(
                    "label.invalid_number",
                    f"第 {line_number} 行坐标或尺寸不是数字",
                    label_path,
                )
            )
        else:
            center_x, center_y, width, height = values
            named_values = {
                "中心点 x": center_x,
                "中心点 y": center_y,
                "宽度": width,
                "高度": height,
            }
            invalid_names = [
                name
                for name, value in named_values.items()
                if not math.isfinite(value) or not 0.0 <= value <= 1.0
            ]
            if invalid_names:
                line_issues.append(
                    _label_issue(
                        "label.coordinate_out_of_range",
                        f"第 {line_number} 行以下字段必须在 [0, 1] 内: "
                        + "、".join(invalid_names),
                        label_path,
                    )
                )
            if math.isfinite(width) and math.isfinite(height) and (
                width <= 0.0 or height <= 0.0
            ):
                line_issues.append(
                    _label_issue(
                        "label.non_positive_size",
                        f"第 {line_number} 行宽度和高度必须大于 0",
                        label_path,
                    )
                )

        if line_issues:
            issues.extend(line_issues)
        elif class_id is not None and values is not None:
            boxes.append(NormalizedBox(class_id, *values))

    return AnnotationDocument(label_path, tuple(boxes), tuple(issues))


def load_pixel_boxes(
    label_path: Path,
    image_size: tuple[int, int],
    class_count: int | None = None,
) -> tuple[tuple[PixelBox, ...], tuple[Issue, ...]]:
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽度和高度必须大于 0")

    document = load_annotation_document(label_path, class_count=class_count)
    boxes: list[PixelBox] = []
    for box in document.boxes:
        x1 = int(round((box.center_x - box.width / 2.0) * image_width))
        y1 = int(round((box.center_y - box.height / 2.0) * image_height))
        x2 = int(round((box.center_x + box.width / 2.0) * image_width))
        y2 = int(round((box.center_y + box.height / 2.0) * image_height))
        boxes.append(
            (
                max(0, min(x1, image_width)),
                max(0, min(y1, image_height)),
                max(0, min(x2, image_width)),
                max(0, min(y2, image_height)),
                box.class_id,
            )
        )
    return tuple(boxes), document.issues


def _atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
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


def _legalize_pixel_box(
    box: Sequence[int | float], image_size: tuple[int, int]
) -> PixelBox | None:
    if len(box) != 5:
        raise ValueError("像素框必须包含 x1、y1、x2、y2、class_id")
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽度和高度必须大于 0")

    raw_x1, raw_y1, raw_x2, raw_y2, raw_class_id = box
    class_id = int(raw_class_id)
    if class_id < 0:
        raise ValueError("class_id 不能为负数")
    left = max(0, min(int(round(min(raw_x1, raw_x2))), image_width))
    top = max(0, min(int(round(min(raw_y1, raw_y2))), image_height))
    right = max(0, min(int(round(max(raw_x1, raw_x2))), image_width))
    bottom = max(0, min(int(round(max(raw_y1, raw_y2))), image_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom, class_id


def save_pixel_boxes_atomic(
    label_path: Path,
    boxes: Iterable[Sequence[int | float]],
    image_size: tuple[int, int],
) -> tuple[PixelBox, ...]:
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽度和高度必须大于 0")

    saved_boxes: list[PixelBox] = []
    lines: list[str] = []
    for raw_box in boxes:
        box = _legalize_pixel_box(raw_box, image_size)
        if box is None:
            continue
        x1, y1, x2, y2, class_id = box
        center_x = ((x1 + x2) / 2.0) / image_width
        center_y = ((y1 + y2) / 2.0) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )
        saved_boxes.append(box)

    content = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write_text(Path(label_path), content)
    return tuple(saved_boxes)


def load_classes_file(
    classes_path: Path,
) -> tuple[tuple[str, ...], tuple[Issue, ...]]:
    classes_path = Path(classes_path)
    if not classes_path.exists():
        return (), ()
    try:
        names = tuple(
            line.strip()
            for line in classes_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as exc:
        issue = Issue(
            code="classes.read_error",
            severity=IssueSeverity.ERROR,
            message=f"无法读取类别文件: {exc}",
            path=classes_path,
            suggested_action="检查 classes.txt 的权限和 UTF-8 编码",
        )
        return (), (issue,)
    return names, ()


def save_classes_file_atomic(
    classes_path: Path, names: Iterable[str]
) -> tuple[str, ...]:
    normalized_names = _normalize_class_names(names)
    content = "\n".join(normalized_names) + ("\n" if normalized_names else "")
    _atomic_write_text(Path(classes_path), content)
    return normalized_names


def _normalize_class_names(names: Iterable[str]) -> tuple[str, ...]:
    normalized_names = tuple(str(name).strip() for name in names if str(name).strip())
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("类别名称不能重复")
    return normalized_names


def build_class_id_mapping(
    old_names: Sequence[str], new_names: Sequence[str]
) -> dict[int, int]:
    new_ids = {name: index for index, name in enumerate(new_names)}
    return {
        old_id: new_ids[name]
        for old_id, name in enumerate(old_names)
        if name in new_ids and old_id != new_ids[name]
    }


def rewrite_label_file_atomic(
    label_path: Path,
    id_mapping: Mapping[int, int] | None = None,
    deleted_ids: Iterable[int] = (),
) -> bool:
    label_path = Path(label_path)
    mapping = dict(id_mapping or {})
    deleted = set(deleted_ids)
    content = label_path.read_text(encoding="utf-8-sig")
    output, changed = _rewrite_label_content(content, mapping, deleted)
    if not changed:
        return False
    _atomic_write_text(label_path, output)
    return True


def _rewrite_label_content(
    content: str,
    id_mapping: Mapping[int, int],
    deleted_ids: set[int],
) -> tuple[str, bool]:
    output_lines: list[str] = []
    changed = False

    for raw_line in content.splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            output_lines.append(raw_line.rstrip())
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            output_lines.append(raw_line.rstrip())
            continue
        if old_id in deleted_ids:
            changed = True
            continue
        new_id = id_mapping.get(old_id, old_id)
        if new_id != old_id:
            changed = True
        output_lines.append(
            f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
        )

    output = "\n".join(output_lines) + ("\n" if output_lines else "")
    return output, changed


def apply_class_changes_atomic(
    classes_path: Path,
    new_names: Iterable[str],
    label_paths: Iterable[Path],
    id_mapping: Mapping[int, int] | None = None,
    deleted_ids: Iterable[int] = (),
) -> int:
    classes_path = Path(classes_path)
    normalized_names = _normalize_class_names(new_names)
    mapping = dict(id_mapping or {})
    deleted = set(deleted_ids)
    planned_labels: list[tuple[Path, bytes, str]] = []

    for raw_path in label_paths:
        label_path = Path(raw_path)
        original = label_path.read_bytes()
        content = original.decode("utf-8-sig")
        output, changed = _rewrite_label_content(content, mapping, deleted)
        if changed:
            planned_labels.append((label_path, original, output))

    classes_existed = classes_path.exists()
    original_classes = classes_path.read_bytes() if classes_existed else b""
    written_labels: list[tuple[Path, bytes]] = []
    classes_written = False
    try:
        for label_path, original, output in planned_labels:
            _atomic_write_text(label_path, output)
            written_labels.append((label_path, original))
        save_classes_file_atomic(classes_path, normalized_names)
        classes_written = True
    except Exception as exc:
        rollback_errors: list[str] = []
        for label_path, original in reversed(written_labels):
            try:
                _atomic_write_bytes(label_path, original)
            except Exception as rollback_exc:
                rollback_errors.append(f"{label_path}: {rollback_exc}")
        if classes_written:
            try:
                if classes_existed:
                    _atomic_write_bytes(classes_path, original_classes)
                else:
                    classes_path.unlink(missing_ok=True)
            except Exception as rollback_exc:
                rollback_errors.append(f"{classes_path}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "类别变更失败且回滚不完整: " + "; ".join(rollback_errors)
            ) from exc
        raise
    return len(planned_labels)


def _box_area(box: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_overlap_stats(
    box_a: Sequence[int | float], box_b: Sequence[int | float]
) -> tuple[float, float, float]:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection_area = intersection_width * intersection_height
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    union = area_a + area_b - intersection_area
    iou = intersection_area / union if union > 0 else 0.0
    minimum_area = min(area_a, area_b)
    maximum_area = max(area_a, area_b)
    containment = intersection_area / minimum_area if minimum_area > 0 else 0.0
    area_ratio = minimum_area / maximum_area if maximum_area > 0 else 0.0
    return iou, containment, area_ratio


def is_duplicate_auto_box(
    box_a: Sequence[int | float], box_b: Sequence[int | float]
) -> bool:
    iou, containment, area_ratio = _box_overlap_stats(box_a, box_b)
    if iou >= AUTO_DEDUPE_IOU_THRESHOLD:
        return True
    return (
        containment >= AUTO_DEDUPE_CONTAINMENT_THRESHOLD
        and area_ratio >= AUTO_DEDUPE_MIN_AREA_RATIO
    )


def dedupe_auto_annotation_candidates(
    candidates: Iterable[dict], existing_boxes: Iterable[Sequence[int | float]]
) -> tuple[list[dict], int, int]:
    existing = tuple(existing_boxes)
    kept: list[dict] = []
    skipped_existing = 0
    skipped_duplicate = 0
    sorted_candidates = sorted(
        candidates,
        key=lambda item: item.get("conf", 0.0),
        reverse=True,
    )
    for candidate in sorted_candidates:
        box = candidate["box"]
        if any(is_duplicate_auto_box(box, existing_box) for existing_box in existing):
            skipped_existing += 1
            continue
        if any(is_duplicate_auto_box(box, kept_item["box"]) for kept_item in kept):
            skipped_duplicate += 1
            continue
        kept.append(candidate)
    return kept, skipped_existing, skipped_duplicate

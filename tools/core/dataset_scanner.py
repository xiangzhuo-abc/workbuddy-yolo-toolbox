from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .annotation_service import load_annotation_document
from .dataset_state import (
    DatasetSnapshot,
    ImageAnnotationState,
    ImageRecord,
    SplitSummary,
)
from .issues import Issue, IssueSeverity
from .paths import ProjectPaths


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DATASET_SPLITS = ("train", "val", "test", "unlabeled")
TRAINING_SPLITS = frozenset({"train", "val", "test"})


def parse_yolo_label(
    label_path: Path, class_count: int | None
) -> tuple[int, tuple[Issue, ...]]:
    document = load_annotation_document(label_path, class_count=class_count)
    return len(document.boxes), document.issues


class DatasetScanner:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    @staticmethod
    def _files(directory: Path, suffixes: set[str]) -> tuple[Path, ...]:
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                (
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.casefold() in suffixes
                ),
                key=lambda path: path.as_posix().casefold(),
            )
        )

    @staticmethod
    def _relative_stem(path: Path, parent: Path) -> str:
        return path.relative_to(parent).with_suffix("").as_posix().casefold()

    def _class_count(self) -> tuple[int | None, tuple[Issue, ...]]:
        classes_path = self.paths.dataset_dir / "classes.txt"
        if not classes_path.is_file():
            return None, ()
        try:
            lines = classes_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            issue = Issue(
                code="dataset.classes_read_error",
                severity=IssueSeverity.ERROR,
                message=f"无法读取类别文件: {exc}",
                path=classes_path,
                suggested_action="检查 classes.txt 的权限和 UTF-8 编码",
            )
            return None, (issue,)
        return sum(1 for line in lines if line.strip()), ()

    def scan(self) -> DatasetSnapshot:
        class_count, class_issues = self._class_count()
        records: list[ImageRecord] = []
        orphan_labels: list[ImageRecord] = []
        issues: list[Issue] = list(class_issues)
        summaries: list[SplitSummary] = []
        training_names: dict[str, set[str]] = defaultdict(set)
        training_paths: dict[str, list[Path]] = defaultdict(list)

        for split in DATASET_SPLITS:
            image_dir = self.paths.image_dir(split)
            label_dir = self.paths.label_dir(split)
            image_files = self._files(image_dir, IMAGE_SUFFIXES)
            label_files = self._files(label_dir, {".txt"})
            label_by_stem = {
                self._relative_stem(path, label_dir): path for path in label_files
            }
            image_stems = {
                self._relative_stem(path, image_dir) for path in image_files
            }
            state_counts: Counter[str] = Counter()

            for image_path in image_files:
                if split in TRAINING_SPLITS:
                    image_name = image_path.name.casefold()
                    training_names[image_name].add(split)
                    training_paths[image_name].append(image_path)

                stem = self._relative_stem(image_path, image_dir)
                label_path = label_by_stem.get(stem)
                record_issues: list[Issue] = []
                if label_path is None:
                    label_path = label_dir / image_path.relative_to(image_dir).with_suffix(
                        ".txt"
                    )
                    state = ImageAnnotationState.UNREVIEWED
                    box_count = 0
                else:
                    box_count, parsed_issues = parse_yolo_label(
                        label_path, class_count=class_count
                    )
                    record_issues.extend(parsed_issues)
                    if parsed_issues:
                        state = ImageAnnotationState.INVALID_LABEL
                    elif box_count:
                        state = ImageAnnotationState.ANNOTATED
                    else:
                        state = ImageAnnotationState.EMPTY_CONFIRMED

                    if split == "unlabeled":
                        record_issues.append(
                            Issue(
                                code="dataset.label_in_unlabeled_split",
                                severity=IssueSeverity.WARNING,
                                message="unlabeled 子集中存在标签文件",
                                path=label_path,
                                suggested_action="确认该图片是否应移动到正式数据子集",
                            )
                        )

                record = ImageRecord(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    state=state,
                    box_count=box_count,
                    issues=tuple(record_issues),
                )
                records.append(record)
                issues.extend(record_issues)
                state_counts[state.value] += 1

            for label_path in label_files:
                stem = self._relative_stem(label_path, label_dir)
                if stem in image_stems:
                    continue

                record_issues: list[Issue] = []
                if split == "unlabeled":
                    record_issues.append(
                        Issue(
                            code="dataset.label_in_unlabeled_split",
                            severity=IssueSeverity.WARNING,
                            message="unlabeled 子集中存在标签文件",
                            path=label_path,
                            suggested_action="确认该标签是否应移动到正式数据子集",
                        )
                    )
                record_issues.append(
                    Issue(
                        code="dataset.orphan_label",
                        severity=IssueSeverity.ERROR,
                        message="标签文件没有对应图片",
                        path=label_path,
                        suggested_action="恢复对应图片或移除无效标签",
                    )
                )
                missing_record = ImageRecord(
                    image_path=image_dir / label_path.relative_to(label_dir).with_suffix(""),
                    label_path=label_path,
                    split=split,
                    state=ImageAnnotationState.MISSING_IMAGE,
                    box_count=0,
                    issues=tuple(record_issues),
                )
                orphan_labels.append(missing_record)
                issues.extend(record_issues)
                state_counts[ImageAnnotationState.MISSING_IMAGE.value] += 1

            summaries.append(
                SplitSummary(
                    split=split,
                    image_count=len(image_files),
                    label_count=len(label_files),
                    state_counts=tuple(
                        (state.value, state_counts[state.value])
                        for state in ImageAnnotationState
                    ),
                )
            )

        for image_name, splits in sorted(training_names.items()):
            if len(splits) < 2:
                continue
            overlap_issue = Issue(
                code="dataset.split_overlap",
                severity=IssueSeverity.ERROR,
                message=f"图片 {image_name} 同时出现在子集: {', '.join(sorted(splits))}",
                path=sorted(
                    training_paths[image_name],
                    key=lambda path: path.as_posix().casefold(),
                )[0],
                suggested_action="重新执行数据集划分，确保 train、val、test 互斥",
            )
            issues.append(overlap_issue)

        return DatasetSnapshot(
            dataset_dir=self.paths.dataset_dir,
            records=tuple(records),
            orphan_labels=tuple(orphan_labels),
            issues=tuple(issues),
            split_summaries=tuple(summaries),
        )

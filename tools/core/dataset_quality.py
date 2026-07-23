from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

from .annotation_service import load_annotation_document, load_classes_file
from .dataset_scanner import DatasetScanner, TRAINING_SPLITS
from .dataset_split import ClassCoveragePolicy
from .issues import Issue, IssueSeverity
from .paths import ProjectPaths


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class QualityFinding:
    issue: Issue
    image_path: Path | None = None


@dataclass(frozen=True)
class DuplicateGroup:
    digest: str
    image_paths: tuple[Path, ...]
    splits: tuple[str, ...]
    cross_split: bool


@dataclass(frozen=True)
class ClassDistribution:
    class_id: int
    name: str
    image_count: int
    box_count: int
    train_boxes: int
    val_boxes: int
    test_boxes: int
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0


@dataclass(frozen=True)
class DatasetQualityReport:
    dataset_dir: Path
    image_count: int
    findings: tuple[QualityFinding, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    class_distributions: tuple[ClassDistribution, ...]

    def count(self, severity: IssueSeverity) -> int:
        return sum(
            finding.issue.severity is severity for finding in self.findings
        )

    @property
    def error_count(self) -> int:
        return self.count(IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return self.count(IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return self.count(IssueSeverity.INFO)


class DatasetQualityScanner:
    def __init__(
        self,
        paths: ProjectPaths,
        coverage_policy: ClassCoveragePolicy | None = None,
    ):
        self.paths = paths
        self.coverage_policy = coverage_policy or ClassCoveragePolicy()

    @staticmethod
    def _digest(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _path_key(path: Path) -> str:
        return path.as_posix().casefold()

    def scan(
        self,
        progress: ProgressCallback | None = None,
    ) -> DatasetQualityReport:
        snapshot = DatasetScanner(self.paths).scan()
        findings: list[QualityFinding] = []
        image_by_related_path: dict[Path, Path] = {}
        training_records = []

        for record in snapshot.records:
            image_by_related_path[record.image_path] = record.image_path
            image_by_related_path[record.label_path] = record.image_path
            if record.split in TRAINING_SPLITS and record.image_path.is_file():
                training_records.append(record)

        for issue in snapshot.issues:
            image_path = (
                image_by_related_path.get(issue.path)
                if issue.path is not None
                else None
            )
            findings.append(QualityFinding(issue, image_path))

        duplicate_groups, duplicate_findings = self._scan_duplicates(
            training_records,
            progress,
        )
        findings.extend(duplicate_findings)

        class_distributions, class_findings = self._scan_class_distribution(
            snapshot,
        )
        findings.extend(class_findings)

        findings.sort(
            key=lambda finding: (
                {
                    IssueSeverity.ERROR: 0,
                    IssueSeverity.WARNING: 1,
                    IssueSeverity.INFO: 2,
                }[finding.issue.severity],
                finding.issue.code,
                self._path_key(finding.issue.path)
                if finding.issue.path is not None
                else "",
            )
        )
        return DatasetQualityReport(
            dataset_dir=self.paths.dataset_dir,
            image_count=len(training_records),
            findings=tuple(findings),
            duplicate_groups=duplicate_groups,
            class_distributions=class_distributions,
        )

    def _scan_duplicates(
        self,
        records,
        progress: ProgressCallback | None,
    ) -> tuple[tuple[DuplicateGroup, ...], tuple[QualityFinding, ...]]:
        grouped: dict[str, list] = defaultdict(list)
        findings: list[QualityFinding] = []
        total = len(records)

        for index, record in enumerate(records, start=1):
            try:
                grouped[self._digest(record.image_path)].append(record)
            except OSError as exc:
                findings.append(
                    QualityFinding(
                        Issue(
                            code="dataset.image_read_error",
                            severity=IssueSeverity.ERROR,
                            message=f"无法读取图片内容: {exc}",
                            path=record.image_path,
                            suggested_action="检查图片文件权限和完整性",
                        ),
                        record.image_path,
                    )
                )
            if progress is not None:
                progress(index, total)

        duplicate_groups: list[DuplicateGroup] = []
        split_order = {"train": 0, "val": 1, "test": 2}
        for digest, duplicate_records in grouped.items():
            if len(duplicate_records) < 2:
                continue
            ordered_records = sorted(
                duplicate_records,
                key=lambda record: self._path_key(record.image_path),
            )
            image_paths = tuple(record.image_path for record in ordered_records)
            splits = tuple(
                sorted(
                    {record.split for record in ordered_records},
                    key=lambda split: split_order.get(split, 99),
                )
            )
            cross_split = len(splits) > 1
            duplicate_groups.append(
                DuplicateGroup(
                    digest=digest,
                    image_paths=image_paths,
                    splits=splits,
                    cross_split=cross_split,
                )
            )
            if cross_split:
                code = "dataset.content_split_overlap"
                severity = IssueSeverity.ERROR
                message = (
                    f"{len(image_paths)} 张内容相同的图片跨分组出现: "
                    f"{', '.join(splits)}"
                )
                action = "保留一个分组中的样本并重新检查数据划分"
            else:
                code = "dataset.exact_duplicate"
                severity = IssueSeverity.WARNING
                message = (
                    f"{len(image_paths)} 张内容相同的图片位于 {splits[0]} 分组"
                )
                action = "检查是否需要保留重复样本"
            findings.append(
                QualityFinding(
                    Issue(
                        code=code,
                        severity=severity,
                        message=message,
                        path=image_paths[0],
                        suggested_action=action,
                    ),
                    image_paths[0],
                )
            )

        duplicate_groups.sort(
            key=lambda group: (
                not group.cross_split,
                -len(group.image_paths),
                group.digest,
            )
        )
        return tuple(duplicate_groups), tuple(findings)

    def _scan_class_distribution(
        self,
        snapshot,
    ) -> tuple[tuple[ClassDistribution, ...], tuple[QualityFinding, ...]]:
        classes_path = self.paths.dataset_dir / "classes.txt"
        class_names, _issues = load_classes_file(classes_path)
        box_counts: Counter[int] = Counter()
        split_counts: dict[str, Counter[int]] = defaultdict(Counter)
        image_paths_by_class: dict[int, set[Path]] = defaultdict(set)
        split_image_paths_by_class: dict[
            str, dict[int, set[Path]]
        ] = defaultdict(lambda: defaultdict(set))

        for record in snapshot.records:
            if record.split not in TRAINING_SPLITS or not record.label_path.is_file():
                continue
            document = load_annotation_document(
                record.label_path,
                class_count=len(class_names) if class_names else None,
            )
            image_classes = set()
            for box in document.boxes:
                if not (0 <= box.class_id < len(class_names)):
                    continue
                box_counts[box.class_id] += 1
                split_counts[record.split][box.class_id] += 1
                image_classes.add(box.class_id)
            for class_id in image_classes:
                image_paths_by_class[class_id].add(record.image_path)
                split_image_paths_by_class[record.split][class_id].add(
                    record.image_path
                )

        distributions = tuple(
            ClassDistribution(
                class_id=class_id,
                name=name,
                image_count=len(image_paths_by_class[class_id]),
                box_count=box_counts[class_id],
                train_boxes=split_counts["train"][class_id],
                val_boxes=split_counts["val"][class_id],
                test_boxes=split_counts["test"][class_id],
                train_images=len(
                    split_image_paths_by_class["train"][class_id]
                ),
                val_images=len(
                    split_image_paths_by_class["val"][class_id]
                ),
                test_images=len(
                    split_image_paths_by_class["test"][class_id]
                ),
            )
            for class_id, name in enumerate(class_names)
        )

        findings: list[QualityFinding] = []
        val_has_images = any(
            summary.split == "val" and summary.image_count > 0
            for summary in snapshot.split_summaries
        )
        test_has_images = any(
            summary.split == "test" and summary.image_count > 0
            for summary in snapshot.split_summaries
        )
        for item in distributions:
            non_train_images = item.val_images + item.test_images
            if item.train_images == 0 and non_train_images > 0:
                findings.append(
                    QualityFinding(
                        Issue(
                            code="class.missing_in_train",
                            severity=IssueSeverity.ERROR,
                            message=f"类别 {item.class_id}: {item.name} 未出现在训练集",
                            path=classes_path,
                            suggested_action="补充训练样本或检查类别映射",
                        )
                    )
                )
            elif item.box_count == 0:
                findings.append(
                    QualityFinding(
                        Issue(
                            code="class.unused",
                            severity=IssueSeverity.INFO,
                            message=f"类别 {item.class_id}: {item.name} 没有有效标注框",
                            path=classes_path,
                            suggested_action="确认是否仍需要该类别",
                        )
                    )
                )
            else:
                _required_train, required_val, _required_test = (
                    self.coverage_policy.required_counts(
                        item.image_count,
                        val_enabled=val_has_images,
                        test_enabled=test_has_images,
                    )
                )
                if (
                    required_val > 0
                    and item.train_images > 0
                    and item.val_images == 0
                ):
                    findings.append(
                        QualityFinding(
                            Issue(
                                code="class.missing_in_val",
                                severity=IssueSeverity.WARNING,
                                message=f"类别 {item.class_id}: {item.name} 未进入验证集",
                                path=classes_path,
                                suggested_action="调整划分以覆盖该类别",
                            )
                        )
                    )
                if (
                    0
                    < item.train_images
                    < self.coverage_policy.min_train_images
                ):
                    findings.append(
                        QualityFinding(
                            Issue(
                                code="class.low_sample_count",
                                severity=IssueSeverity.WARNING,
                                message=(
                                    f"类别 {item.class_id}: {item.name} "
                                    f"训练图片仅 {item.train_images} 张"
                                    f"（{item.train_boxes} 框）"
                                ),
                                path=classes_path,
                                suggested_action=(
                                    "补充至少 "
                                    f"{self.coverage_policy.min_train_images} "
                                    "张不同训练图片"
                                ),
                            )
                        )
                    )

        positive_train_counts = [
            item.train_images
            for item in distributions
            if item.train_images > 0
        ]
        if positive_train_counts:
            minimum = min(positive_train_counts)
            maximum = max(positive_train_counts)
            if maximum >= 20 and maximum >= minimum * 10:
                findings.append(
                    QualityFinding(
                        Issue(
                            code="class.imbalance",
                            severity=IssueSeverity.INFO,
                            message=(
                                "训练类别图片数差异较大: "
                                f"最少 {minimum}，最多 {maximum}"
                            ),
                            path=classes_path,
                            suggested_action="优先补充低样本类别",
                        )
                    )
                )
        return distributions, tuple(findings)

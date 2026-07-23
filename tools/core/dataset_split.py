from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path

from .annotation_service import load_annotation_document, load_classes_file
from .dataset_scanner import DatasetScanner, TRAINING_SPLITS
from .dataset_state import ImageAnnotationState
from .issues import Issue, IssueSeverity
from .paths import ProjectPaths


SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class ClassCoveragePolicy:
    min_train_images: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.min_train_images <= 20:
            raise ValueError("最低训练图片数必须在 1 到 20 之间")

    def required_counts(
        self,
        total_images: int,
        *,
        val_enabled: bool,
        test_enabled: bool,
    ) -> tuple[int, int, int]:
        if total_images < 0:
            raise ValueError("类别图片数不能为负数")

        train = min(total_images, self.min_train_images)
        remaining = total_images - train
        val = 1 if val_enabled and remaining > 0 else 0
        remaining -= val
        test = 1 if test_enabled and remaining > 0 else 0
        return train, val, test


class SplitMode(str, Enum):
    REPAIR = "repair"
    FULL = "full"


@dataclass(frozen=True)
class SplitPolicy:
    train_ratio: float = 0.80
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    seed: int = 42
    mode: SplitMode = SplitMode.REPAIR
    coverage: ClassCoveragePolicy = field(
        default_factory=ClassCoveragePolicy
    )

    def __post_init__(self) -> None:
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(not math.isfinite(value) or value < 0 for value in ratios):
            raise ValueError("划分比例必须是非负有限数")
        if abs(sum(ratios) - 1.0) > 0.001:
            raise ValueError("train、val、test 比例之和必须为 1")
        if self.train_ratio <= 0:
            raise ValueError("训练集比例必须大于 0")
        if self.seed < 0:
            raise ValueError("随机种子不能为负数")

    @property
    def ratios(self) -> dict[str, float]:
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }


@dataclass(frozen=True)
class SplitSample:
    key: str
    current_split: str
    image_path: Path
    label_path: Path
    relative_image_path: Path
    relative_label_path: Path
    class_ids: tuple[int, ...]
    box_count: int
    image_sha256: str
    label_sha256: str
    image_size: int
    label_size: int

    @property
    def is_negative(self) -> bool:
        return not self.class_ids


@dataclass(frozen=True)
class SplitMove:
    key: str
    source_split: str
    target_split: str
    image_path: Path
    label_path: Path
    relative_image_path: Path
    relative_label_path: Path
    class_ids: tuple[int, ...]


@dataclass(frozen=True)
class ClassCoverage:
    class_id: int
    name: str
    total_images: int
    total_boxes: int
    required_train: int
    required_val: int
    required_test: int
    current_train: int
    current_val: int
    current_test: int
    planned_train: int
    planned_val: int
    planned_test: int

    @property
    def requirements_met(self) -> bool:
        return (
            self.planned_train >= self.required_train
            and self.planned_val >= self.required_val
            and self.planned_test >= self.required_test
        )


@dataclass(frozen=True)
class SplitRisk:
    code: str
    message: str
    class_id: int | None = None


@dataclass(frozen=True)
class SplitPlan:
    dataset_dir: Path
    policy: SplitPolicy
    samples: tuple[SplitSample, ...]
    assignments: tuple[tuple[str, str], ...]
    moves: tuple[SplitMove, ...]
    class_coverages: tuple[ClassCoverage, ...]
    current_counts: tuple[tuple[str, int], ...]
    target_counts: tuple[tuple[str, int], ...]
    planned_counts: tuple[tuple[str, int], ...]
    blocking_issues: tuple[Issue, ...]
    risks: tuple[SplitRisk, ...]
    dataset_fingerprint: str
    plan_id: str

    @property
    def is_executable(self) -> bool:
        return bool(self.samples) and not self.blocking_issues

    def assignment_map(self) -> dict[str, str]:
        return dict(self.assignments)

    def count_map(self, name: str) -> dict[str, int]:
        return dict(getattr(self, name))


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_fingerprint(paths: ProjectPaths) -> str:
    dataset_dir = paths.dataset_dir
    entries: list[str] = []
    candidates: list[Path] = []
    for directory_name in ("images", "labels"):
        directory = dataset_dir / directory_name
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.casefold() != ".cache"
            )
    for file_name in ("classes.txt", "data.yaml"):
        path = dataset_dir / file_name
        if path.is_file():
            candidates.append(path)

    for path in sorted(
        set(candidates),
        key=lambda item: item.relative_to(dataset_dir).as_posix().casefold(),
    ):
        relative = path.relative_to(dataset_dir).as_posix()
        entries.append(f"{relative}|{path.stat().st_size}|{_digest_file(path)}")
    return sha256("\n".join(entries).encode("utf-8")).hexdigest()


class SplitPlanner:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    @staticmethod
    def _issue_key(issue: Issue) -> tuple[str, str, str]:
        return (
            issue.code,
            str(issue.path) if issue.path is not None else "",
            issue.message,
        )

    @staticmethod
    def _stable_rank(seed: int, key: str, target: str) -> int:
        payload = f"{seed}|{target}|{key}".encode("utf-8")
        return int.from_bytes(sha256(payload).digest()[:8], "big")

    @staticmethod
    def _target_counts(count: int, policy: SplitPolicy) -> dict[str, int]:
        raw = {
            split: count * ratio
            for split, ratio in policy.ratios.items()
        }
        result = {split: int(math.floor(value)) for split, value in raw.items()}
        remaining = count - sum(result.values())
        ranked = sorted(
            SPLIT_ORDER,
            key=lambda split: (
                raw[split] - result[split],
                policy.ratios[split],
                -SPLIT_ORDER.index(split),
            ),
            reverse=True,
        )
        for split in ranked[:remaining]:
            result[split] += 1

        enabled = [split for split in SPLIT_ORDER if policy.ratios[split] > 0]
        if count >= len(enabled):
            for split in enabled:
                if result[split] > 0:
                    continue
                donor = max(
                    enabled,
                    key=lambda item: (result[item], policy.ratios[item]),
                )
                if result[donor] > 1:
                    result[donor] -= 1
                    result[split] += 1
        return result

    @staticmethod
    def _class_counts(
        samples: tuple[SplitSample, ...],
        assignments: dict[str, str],
    ) -> dict[int, Counter[str]]:
        counts: dict[int, Counter[str]] = defaultdict(Counter)
        for sample in samples:
            split = assignments[sample.key]
            for class_id in sample.class_ids:
                counts[class_id][split] += 1
        return counts

    @staticmethod
    def _split_counts(assignments: dict[str, str]) -> Counter[str]:
        counts = Counter(assignments.values())
        for split in SPLIT_ORDER:
            counts.setdefault(split, 0)
        return counts

    @staticmethod
    def _requirements(
        samples: tuple[SplitSample, ...],
        policy: SplitPolicy,
    ) -> dict[int, dict[str, int]]:
        totals: Counter[int] = Counter()
        for sample in samples:
            totals.update(sample.class_ids)
        requirements: dict[int, dict[str, int]] = {}
        for class_id, total in totals.items():
            train, val, test = policy.coverage.required_counts(
                total,
                val_enabled=policy.val_ratio > 0,
                test_enabled=policy.test_ratio > 0,
            )
            requirements[class_id] = {
                "train": train,
                "val": val,
                "test": test,
            }
        return requirements

    @staticmethod
    def _desired_class_counts(
        samples: tuple[SplitSample, ...],
        policy: SplitPolicy,
        requirements: dict[int, dict[str, int]],
    ) -> dict[int, dict[str, int]]:
        totals: Counter[int] = Counter()
        for sample in samples:
            totals.update(sample.class_ids)
        desired: dict[int, dict[str, int]] = {}
        for class_id, total in totals.items():
            values = dict(requirements[class_id])
            remaining = total - sum(values.values())
            while remaining > 0:
                target = max(
                    SPLIT_ORDER,
                    key=lambda split: (
                        total * policy.ratios[split] - values[split],
                        policy.ratios[split],
                        -SPLIT_ORDER.index(split),
                    ),
                )
                values[target] += 1
                remaining -= 1
            desired[class_id] = values
        return desired

    @staticmethod
    def _can_move(
        sample: SplitSample,
        source: str,
        target: str,
        class_counts: dict[int, Counter[str]],
        requirements: dict[int, dict[str, int]],
    ) -> bool:
        if source == target:
            return False
        return all(
            class_counts[class_id][source] - 1
            >= requirements[class_id][source]
            for class_id in sample.class_ids
        )

    def _choose_requirement_move(
        self,
        samples: tuple[SplitSample, ...],
        assignments: dict[str, str],
        class_counts: dict[int, Counter[str]],
        split_counts: Counter[str],
        target_counts: dict[str, int],
        requirements: dict[int, dict[str, int]],
        desired: dict[int, dict[str, int]],
        policy: SplitPolicy,
    ) -> tuple[SplitSample, str] | None:
        deficits = []
        for class_id, required in requirements.items():
            total = sum(class_counts[class_id].values())
            for target in SPLIT_ORDER:
                deficit = required[target] - class_counts[class_id][target]
                if deficit > 0:
                    deficits.append(
                        (SPLIT_ORDER.index(target), total, class_id, target)
                    )
        for _priority, _total, class_id, target in sorted(deficits):
            candidates: list[tuple[tuple[int, ...], SplitSample]] = []
            for sample in samples:
                if class_id not in sample.class_ids:
                    continue
                source = assignments[sample.key]
                if not self._can_move(
                    sample,
                    source,
                    target,
                    class_counts,
                    requirements,
                ):
                    continue
                deficit_classes = sum(
                    class_counts[item][target] < requirements[item][target]
                    for item in sample.class_ids
                )
                newly_satisfied = sum(
                    0 < requirements[item][target] - class_counts[item][target] <= 1
                    for item in sample.class_ids
                )
                desired_gain = sum(
                    max(0, desired[item][target] - class_counts[item][target])
                    for item in sample.class_ids
                )
                score = (
                    newly_satisfied,
                    deficit_classes,
                    desired_gain,
                    max(0, target_counts[target] - split_counts[target]),
                    max(0, split_counts[source] - target_counts[source]),
                    -self._stable_rank(policy.seed, sample.key, target),
                )
                candidates.append((score, sample))
            if candidates:
                return max(candidates, key=lambda item: item[0])[1], target
        return None

    def _choose_balance_move(
        self,
        samples: tuple[SplitSample, ...],
        assignments: dict[str, str],
        class_counts: dict[int, Counter[str]],
        split_counts: Counter[str],
        target_counts: dict[str, int],
        requirements: dict[int, dict[str, int]],
        desired: dict[int, dict[str, int]],
        policy: SplitPolicy,
    ) -> tuple[SplitSample, str] | None:
        targets = sorted(
            (
                split
                for split in SPLIT_ORDER
                if split_counts[split] < target_counts[split]
            ),
            key=lambda split: (
                target_counts[split] - split_counts[split],
                -SPLIT_ORDER.index(split),
            ),
            reverse=True,
        )
        sources = sorted(
            (
                split
                for split in SPLIT_ORDER
                if split_counts[split] > target_counts[split]
            ),
            key=lambda split: (
                split_counts[split] - target_counts[split],
                -SPLIT_ORDER.index(split),
            ),
            reverse=True,
        )
        candidates: list[tuple[tuple[int, ...], SplitSample, str]] = []
        for target in targets:
            for source in sources:
                for sample in samples:
                    if assignments[sample.key] != source:
                        continue
                    if not self._can_move(
                        sample,
                        source,
                        target,
                        class_counts,
                        requirements,
                    ):
                        continue
                    gain = sum(
                        max(
                            0,
                            desired[class_id][target]
                            - class_counts[class_id][target],
                        )
                        for class_id in sample.class_ids
                    )
                    loss = sum(
                        max(
                            0,
                            desired[class_id][source]
                            - (class_counts[class_id][source] - 1),
                        )
                        for class_id in sample.class_ids
                    )
                    score = (
                        gain - loss,
                        gain,
                        -len(sample.class_ids),
                        -self._stable_rank(policy.seed, sample.key, target),
                    )
                    candidates.append((score, sample, target))
        if not candidates:
            return None
        _score, sample, target = max(candidates, key=lambda item: item[0])
        return sample, target

    def _choose_requirement_swap(
        self,
        samples: tuple[SplitSample, ...],
        assignments: dict[str, str],
        class_counts: dict[int, Counter[str]],
        requirements: dict[int, dict[str, int]],
        desired: dict[int, dict[str, int]],
        policy: SplitPolicy,
    ) -> tuple[SplitSample, SplitSample, str, str] | None:
        before_deficit = sum(
            max(0, required[split] - class_counts[class_id][split])
            for class_id, required in requirements.items()
            for split in SPLIT_ORDER
        )
        deficits = sorted(
            (
                SPLIT_ORDER.index(target),
                sum(class_counts[class_id].values()),
                class_id,
                target,
            )
            for class_id, required in requirements.items()
            for target in SPLIT_ORDER
            if class_counts[class_id][target] < required[target]
        )
        candidates = []
        for _priority, _total, deficit_class, target in deficits:
            for incoming in samples:
                if deficit_class not in incoming.class_ids:
                    continue
                source = assignments[incoming.key]
                if source == target:
                    continue
                for outgoing in samples:
                    if assignments[outgoing.key] != target:
                        continue
                    affected = set(incoming.class_ids) | set(outgoing.class_ids)
                    valid = True
                    after_values: dict[tuple[int, str], int] = {}
                    for class_id in affected:
                        source_after = (
                            class_counts[class_id][source]
                            - int(class_id in incoming.class_ids)
                            + int(class_id in outgoing.class_ids)
                        )
                        target_after = (
                            class_counts[class_id][target]
                            + int(class_id in incoming.class_ids)
                            - int(class_id in outgoing.class_ids)
                        )
                        after_values[(class_id, source)] = source_after
                        after_values[(class_id, target)] = target_after
                        if (
                            source_after < requirements[class_id][source]
                            or target_after < requirements[class_id][target]
                        ):
                            valid = False
                            break
                    if not valid:
                        continue

                    after_deficit = before_deficit
                    desired_gain = 0
                    for class_id in affected:
                        for split in (source, target):
                            current = class_counts[class_id][split]
                            after = after_values[(class_id, split)]
                            required = requirements[class_id][split]
                            after_deficit += max(0, required - after) - max(
                                0, required - current
                            )
                            desired_gain += abs(
                                desired[class_id][split] - current
                            ) - abs(desired[class_id][split] - after)
                    improvement = before_deficit - after_deficit
                    if improvement <= 0:
                        continue
                    score = (
                        improvement,
                        desired_gain,
                        -len(affected),
                        -self._stable_rank(
                            policy.seed,
                            f"{incoming.key}|{outgoing.key}",
                            target,
                        ),
                    )
                    candidates.append(
                        (score, incoming, outgoing, source, target)
                    )
        if not candidates:
            return None
        _score, incoming, outgoing, source, target = max(
            candidates,
            key=lambda item: item[0],
        )
        return incoming, outgoing, source, target

    @staticmethod
    def _apply_move(
        sample: SplitSample,
        target: str,
        assignments: dict[str, str],
        class_counts: dict[int, Counter[str]],
        split_counts: Counter[str],
    ) -> None:
        source = assignments[sample.key]
        assignments[sample.key] = target
        split_counts[source] -= 1
        split_counts[target] += 1
        for class_id in sample.class_ids:
            class_counts[class_id][source] -= 1
            class_counts[class_id][target] += 1

    def _build_samples(
        self,
    ) -> tuple[
        tuple[SplitSample, ...], tuple[str, ...], tuple[Issue, ...]
    ]:
        snapshot = DatasetScanner(self.paths).scan()
        class_names, class_issues = load_classes_file(
            self.paths.dataset_dir / "classes.txt"
        )
        blocking = [
            issue
            for issue in (*snapshot.issues, *class_issues)
            if issue.severity is IssueSeverity.ERROR
        ]
        if not class_names:
            blocking.append(
                Issue(
                    code="dataset.classes_empty",
                    severity=IssueSeverity.ERROR,
                    message="classes.txt 不存在或没有有效类别",
                    path=self.paths.dataset_dir / "classes.txt",
                    suggested_action="先配置类别列表",
                )
            )

        samples: list[SplitSample] = []
        sample_by_key: dict[str, SplitSample] = {}
        digest_paths: dict[str, list[Path]] = defaultdict(list)
        for record in snapshot.records:
            if record.split not in TRAINING_SPLITS:
                continue
            if record.state not in {
                ImageAnnotationState.ANNOTATED,
                ImageAnnotationState.EMPTY_CONFIRMED,
            }:
                continue
            try:
                relative_image = record.image_path.relative_to(
                    self.paths.image_dir(record.split)
                )
                relative_label = record.label_path.relative_to(
                    self.paths.label_dir(record.split)
                )
                key = relative_image.with_suffix("").as_posix().casefold()
                document = load_annotation_document(
                    record.label_path,
                    class_count=len(class_names) if class_names else None,
                )
                if document.has_errors:
                    blocking.extend(document.issues)
                    continue
                image_hash = _digest_file(record.image_path)
                label_hash = _digest_file(record.label_path)
                sample = SplitSample(
                    key=key,
                    current_split=record.split,
                    image_path=record.image_path,
                    label_path=record.label_path,
                    relative_image_path=relative_image,
                    relative_label_path=relative_label,
                    class_ids=tuple(
                        sorted({box.class_id for box in document.boxes})
                    ),
                    box_count=len(document.boxes),
                    image_sha256=image_hash,
                    label_sha256=label_hash,
                    image_size=record.image_path.stat().st_size,
                    label_size=record.label_path.stat().st_size,
                )
            except (OSError, ValueError) as exc:
                blocking.append(
                    Issue(
                        code="dataset.split_inventory_error",
                        severity=IssueSeverity.ERROR,
                        message=f"无法建立划分清单: {exc}",
                        path=record.image_path,
                        suggested_action="检查文件权限和数据集目录结构",
                    )
                )
                continue

            if key in sample_by_key:
                blocking.append(
                    Issue(
                        code="dataset.ambiguous_image_stem",
                        severity=IssueSeverity.ERROR,
                        message=f"同一相对 stem 对应多张图片: {key}",
                        path=record.image_path,
                        suggested_action="重命名冲突图片并保持标签同名",
                    )
                )
                continue
            sample_by_key[key] = sample
            samples.append(sample)
            digest_paths[image_hash].append(record.image_path)

        for digest, image_paths in digest_paths.items():
            if len(image_paths) < 2:
                continue
            blocking.append(
                Issue(
                    code="dataset.exact_duplicate",
                    severity=IssueSeverity.ERROR,
                    message=f"发现 {len(image_paths)} 张内容完全相同的图片",
                    path=sorted(
                        image_paths,
                        key=lambda path: path.as_posix().casefold(),
                    )[0],
                    suggested_action="先在数据质量检查中处理重复图片",
                )
            )

        unique_issues = {
            self._issue_key(issue): issue for issue in blocking
        }
        return (
            tuple(sorted(samples, key=lambda item: item.key)),
            tuple(class_names),
            tuple(unique_issues[key] for key in sorted(unique_issues)),
        )

    def _plan_id(
        self,
        fingerprint: str,
        policy: SplitPolicy,
        assignments: tuple[tuple[str, str], ...],
    ) -> str:
        payload = {
            "fingerprint": fingerprint,
            "policy": {
                "train": policy.train_ratio,
                "val": policy.val_ratio,
                "test": policy.test_ratio,
                "seed": policy.seed,
                "mode": policy.mode.value,
                "min_train": policy.coverage.min_train_images,
            },
            "assignments": assignments,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def plan(self, policy: SplitPolicy | None = None) -> SplitPlan:
        policy = policy or SplitPolicy()
        samples, class_names, blocking_issues = self._build_samples()
        try:
            fingerprint = compute_dataset_fingerprint(self.paths)
        except OSError as exc:
            fingerprint = ""
            blocking_issues = blocking_issues + (
                Issue(
                    code="dataset.fingerprint_error",
                    severity=IssueSeverity.ERROR,
                    message=f"无法计算数据集指纹: {exc}",
                    path=self.paths.dataset_dir,
                    suggested_action="检查数据集读取权限",
                ),
            )

        current_assignments = {
            sample.key: sample.current_split for sample in samples
        }
        current_counts = self._split_counts(current_assignments)
        target_counts = self._target_counts(len(samples), policy)
        if blocking_issues or not samples:
            assignments_tuple = tuple(sorted(current_assignments.items()))
            return SplitPlan(
                dataset_dir=self.paths.dataset_dir,
                policy=policy,
                samples=samples,
                assignments=assignments_tuple,
                moves=(),
                class_coverages=(),
                current_counts=tuple(
                    (split, current_counts[split]) for split in SPLIT_ORDER
                ),
                target_counts=tuple(
                    (split, target_counts[split]) for split in SPLIT_ORDER
                ),
                planned_counts=tuple(
                    (split, current_counts[split]) for split in SPLIT_ORDER
                ),
                blocking_issues=blocking_issues,
                risks=(),
                dataset_fingerprint=fingerprint,
                plan_id=self._plan_id(
                    fingerprint, policy, assignments_tuple
                ),
            )

        assignments = (
            dict(current_assignments)
            if policy.mode is SplitMode.REPAIR
            else {sample.key: "train" for sample in samples}
        )
        requirements = self._requirements(samples, policy)
        desired = self._desired_class_counts(
            samples, policy, requirements
        )
        class_counts = self._class_counts(samples, assignments)
        split_counts = self._split_counts(assignments)

        max_steps = max(1, len(samples) * max(3, len(requirements) * 2))
        for _ in range(max_steps):
            selected = self._choose_requirement_move(
                samples,
                assignments,
                class_counts,
                split_counts,
                target_counts,
                requirements,
                desired,
                policy,
            )
            if selected is None:
                swapped = self._choose_requirement_swap(
                    samples,
                    assignments,
                    class_counts,
                    requirements,
                    desired,
                    policy,
                )
                if swapped is None:
                    break
                incoming, outgoing, source, target = swapped
                self._apply_move(
                    incoming,
                    target,
                    assignments,
                    class_counts,
                    split_counts,
                )
                self._apply_move(
                    outgoing,
                    source,
                    assignments,
                    class_counts,
                    split_counts,
                )
            else:
                sample, target = selected
                self._apply_move(
                    sample,
                    target,
                    assignments,
                    class_counts,
                    split_counts,
                )

        for _ in range(len(samples) * 3):
            if all(
                split_counts[split] == target_counts[split]
                for split in SPLIT_ORDER
            ):
                break
            selected = self._choose_balance_move(
                samples,
                assignments,
                class_counts,
                split_counts,
                target_counts,
                requirements,
                desired,
                policy,
            )
            if selected is None:
                break
            sample, target = selected
            self._apply_move(
                sample, target, assignments, class_counts, split_counts
            )

        risks: list[SplitRisk] = []
        for class_id, required in sorted(requirements.items()):
            for split in SPLIT_ORDER:
                if class_counts[class_id][split] >= required[split]:
                    continue
                risks.append(
                    SplitRisk(
                        code="split.class_requirement_unmet",
                        class_id=class_id,
                        message=(
                            f"类别 {class_id} 在 {split} 需要 "
                            f"{required[split]} 张，计划仅 "
                            f"{class_counts[class_id][split]} 张"
                        ),
                    )
                )
        if any(
            split_counts[split] != target_counts[split]
            for split in SPLIT_ORDER
        ):
            risks.append(
                SplitRisk(
                    code="split.ratio_deviation",
                    message=(
                        "类别保护限制了目标比例，计划数量为 "
                        + ", ".join(
                            f"{split}={split_counts[split]}"
                            for split in SPLIT_ORDER
                        )
                    ),
                )
            )

        assignments_tuple = tuple(sorted(assignments.items()))
        sample_by_key = {sample.key: sample for sample in samples}
        moves = tuple(
            SplitMove(
                key=key,
                source_split=sample_by_key[key].current_split,
                target_split=target,
                image_path=sample_by_key[key].image_path,
                label_path=sample_by_key[key].label_path,
                relative_image_path=sample_by_key[key].relative_image_path,
                relative_label_path=sample_by_key[key].relative_label_path,
                class_ids=sample_by_key[key].class_ids,
            )
            for key, target in assignments_tuple
            if sample_by_key[key].current_split != target
        )

        current_class_counts = self._class_counts(
            samples, current_assignments
        )
        box_totals: Counter[int] = Counter()
        image_totals: Counter[int] = Counter()
        for sample in samples:
            image_totals.update(sample.class_ids)
            document = load_annotation_document(sample.label_path)
            box_totals.update(box.class_id for box in document.boxes)
        coverages = tuple(
            ClassCoverage(
                class_id=class_id,
                name=(
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"类别 {class_id}"
                ),
                total_images=image_totals[class_id],
                total_boxes=box_totals[class_id],
                required_train=requirements[class_id]["train"],
                required_val=requirements[class_id]["val"],
                required_test=requirements[class_id]["test"],
                current_train=current_class_counts[class_id]["train"],
                current_val=current_class_counts[class_id]["val"],
                current_test=current_class_counts[class_id]["test"],
                planned_train=class_counts[class_id]["train"],
                planned_val=class_counts[class_id]["val"],
                planned_test=class_counts[class_id]["test"],
            )
            for class_id in sorted(requirements)
        )

        return SplitPlan(
            dataset_dir=self.paths.dataset_dir,
            policy=policy,
            samples=samples,
            assignments=assignments_tuple,
            moves=moves,
            class_coverages=coverages,
            current_counts=tuple(
                (split, current_counts[split]) for split in SPLIT_ORDER
            ),
            target_counts=tuple(
                (split, target_counts[split]) for split in SPLIT_ORDER
            ),
            planned_counts=tuple(
                (split, split_counts[split]) for split in SPLIT_ORDER
            ),
            blocking_issues=(),
            risks=tuple(risks),
            dataset_fingerprint=fingerprint,
            plan_id=self._plan_id(
                fingerprint, policy, assignments_tuple
            ),
        )

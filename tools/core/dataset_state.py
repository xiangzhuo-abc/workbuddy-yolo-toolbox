from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .issues import Issue, IssueSeverity


class ImageAnnotationState(str, Enum):
    UNREVIEWED = "unreviewed"
    ANNOTATED = "annotated"
    EMPTY_CONFIRMED = "empty_confirmed"
    INVALID_LABEL = "invalid_label"
    MISSING_IMAGE = "missing_image"


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    split: str
    state: ImageAnnotationState
    box_count: int
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class SplitSummary:
    split: str
    image_count: int
    label_count: int
    state_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_dir: Path
    records: tuple[ImageRecord, ...]
    orphan_labels: tuple[ImageRecord, ...]
    issues: tuple[Issue, ...]
    split_summaries: tuple[SplitSummary, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is IssueSeverity.ERROR for issue in self.issues)

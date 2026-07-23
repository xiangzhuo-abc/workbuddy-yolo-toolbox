from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import TestCase

from tools.core.dataset_state import (
    DatasetSnapshot,
    ImageAnnotationState,
    ImageRecord,
    SplitSummary,
)
from tools.core.issues import Issue, IssueSeverity


class DatasetStateTests(TestCase):
    def test_state_values_are_stable(self):
        self.assertEqual(
            [state.value for state in ImageAnnotationState],
            [
                "unreviewed",
                "annotated",
                "empty_confirmed",
                "invalid_label",
                "missing_image",
            ],
        )

    def test_models_preserve_tuple_fields(self):
        issue = Issue("label.invalid", IssueSeverity.WARNING, "标签格式异常")
        record = ImageRecord(
            image_path=Path("images/train/a.png"),
            label_path=Path("labels/train/a.txt"),
            split="train",
            state=ImageAnnotationState.INVALID_LABEL,
            box_count=0,
            issues=(issue,),
        )
        summary = SplitSummary(
            split="train",
            image_count=1,
            label_count=1,
            state_counts=((ImageAnnotationState.INVALID_LABEL.value, 1),),
        )
        snapshot = DatasetSnapshot(
            dataset_dir=Path("D:/dataset"),
            records=(record,),
            orphan_labels=(),
            issues=(issue,),
            split_summaries=(summary,),
        )

        self.assertIsInstance(record.issues, tuple)
        self.assertIsInstance(summary.state_counts, tuple)
        self.assertEqual(snapshot.records, (record,))

    def test_models_are_frozen(self):
        record = ImageRecord(
            Path("a.png"),
            Path("a.txt"),
            "train",
            ImageAnnotationState.UNREVIEWED,
            0,
            (),
        )
        with self.assertRaises(FrozenInstanceError):
            record.box_count = 1

    def test_snapshot_reports_only_error_severity(self):
        warning_snapshot = DatasetSnapshot(
            Path("D:/dataset"),
            (),
            (),
            (Issue("warning", IssueSeverity.WARNING, "警告"),),
            (),
        )
        error_snapshot = DatasetSnapshot(
            Path("D:/dataset"),
            (),
            (),
            (Issue("error", IssueSeverity.ERROR, "错误"),),
            (),
        )

        self.assertFalse(warning_snapshot.has_errors)
        self.assertTrue(error_snapshot.has_errors)

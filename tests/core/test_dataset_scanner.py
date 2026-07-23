from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.core.dataset_scanner import DatasetScanner, parse_yolo_label
from tools.core.dataset_state import ImageAnnotationState
from tools.core.issues import IssueSeverity
from tools.core.paths import ProjectPaths


class DatasetScannerTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.dataset_dir = self.project_dir / "dataset"
        self.paths = ProjectPaths.from_project_dir(self.project_dir)
        (self.dataset_dir / "classes.txt").parent.mkdir(parents=True)
        (self.dataset_dir / "classes.txt").write_text(
            "按钮\n图标\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_image(self, split: str, name: str) -> Path:
        path = self.dataset_dir / "images" / split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return path

    def _write_label(self, split: str, name: str, content: str) -> Path:
        path = self.dataset_dir / "labels" / split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_classifies_all_states(self):
        self._write_image("train", "annotated.png")
        self._write_label("train", "annotated.txt", "0 0.5 0.5 0.2 0.3\n")
        self._write_image("train", "empty.png")
        self._write_label("train", "empty.txt", "")
        self._write_image("train", "broken.png")
        self._write_label("train", "broken.txt", "0 0.5 0.5 0.2\n")
        self._write_image("train", "new.png")
        self._write_label("train", "orphan.txt", "1 0.5 0.5 0.2 0.3\n")

        snapshot = DatasetScanner(self.paths).scan()
        states = {record.image_path.name: record.state for record in snapshot.records}

        self.assertEqual(states["annotated.png"], ImageAnnotationState.ANNOTATED)
        self.assertEqual(states["empty.png"], ImageAnnotationState.EMPTY_CONFIRMED)
        self.assertEqual(states["broken.png"], ImageAnnotationState.INVALID_LABEL)
        self.assertEqual(states["new.png"], ImageAnnotationState.UNREVIEWED)
        self.assertEqual(len(snapshot.orphan_labels), 1)
        self.assertEqual(
            snapshot.orphan_labels[0].state, ImageAnnotationState.MISSING_IMAGE
        )
        self.assertTrue(snapshot.has_errors)

    def test_parse_yolo_label_reports_every_invalid_rule(self):
        label_path = self._write_label(
            "train",
            "invalid.txt",
            "0 0.5 0.5 0.2\n"
            "x 0.5 0.5 0.2 0.2\n"
            "2 0.5 0.5 0.2 0.2\n"
            "0 abc 0.5 0.2 0.2\n"
            "0 nan 0.5 0.2 0.2\n"
            "0 1.1 0.5 0.2 0.2\n"
            "0 0.5 0.5 0 0.2\n",
        )

        box_count, issues = parse_yolo_label(label_path, class_count=2)
        codes = {issue.code for issue in issues}

        self.assertEqual(box_count, 0)
        self.assertEqual(
            codes,
            {
                "label.field_count",
                "label.invalid_class_id",
                "label.class_out_of_range",
                "label.invalid_number",
                "label.coordinate_out_of_range",
                "label.non_positive_size",
            },
        )
        self.assertTrue(all(issue.severity is IssueSeverity.ERROR for issue in issues))
        self.assertTrue(all(issue.path == label_path for issue in issues))

    def test_reports_overlap_unlabeled_labels_and_stable_summaries(self):
        self._write_image("train", "same.png")
        self._write_label("train", "same.txt", "0 0.5 0.5 0.2 0.2\n")
        self._write_image("val", "same.png")
        self._write_image("unlabeled", "pending.jpg")
        self._write_label("unlabeled", "pending.txt", "")

        snapshot = DatasetScanner(self.paths).scan()
        overlaps = [
            issue for issue in snapshot.issues if issue.code == "dataset.split_overlap"
        ]
        unlabeled_warnings = [
            issue
            for issue in snapshot.issues
            if issue.code == "dataset.label_in_unlabeled_split"
        ]
        summaries = {summary.split: summary for summary in snapshot.split_summaries}
        train_counts = dict(summaries["train"].state_counts)
        unlabeled_counts = dict(summaries["unlabeled"].state_counts)

        self.assertEqual(len(overlaps), 1)
        self.assertIs(overlaps[0].severity, IssueSeverity.ERROR)
        self.assertEqual(len(unlabeled_warnings), 1)
        self.assertIs(unlabeled_warnings[0].severity, IssueSeverity.WARNING)
        self.assertEqual(summaries["train"].image_count, 1)
        self.assertEqual(summaries["train"].label_count, 1)
        self.assertEqual(train_counts[ImageAnnotationState.ANNOTATED.value], 1)
        self.assertEqual(
            unlabeled_counts[ImageAnnotationState.EMPTY_CONFIRMED.value], 1
        )

    def test_scan_does_not_modify_dataset_files(self):
        self._write_image("train", "a.png")
        self._write_label("train", "a.txt", "0 0.5 0.5 0.2 0.2\n")
        self._write_image("train", "new.png")
        files = [path for path in self.dataset_dir.rglob("*") if path.is_file()]
        before = {
            path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files
        }

        DatasetScanner(self.paths).scan()

        after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}
        self.assertEqual(before, after)

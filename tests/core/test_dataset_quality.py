from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.core.dataset_quality import DatasetQualityScanner
from tools.core.issues import IssueSeverity
from tools.core.paths import ProjectPaths


class DatasetQualityScannerTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.dataset_dir = self.project_dir / "dataset"
        self.paths = ProjectPaths.from_project_dir(self.project_dir)
        self.dataset_dir.mkdir()
        (self.dataset_dir / "classes.txt").write_text(
            "按钮\n图标\n未使用\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_pair(
        self,
        split: str,
        name: str,
        image_content: bytes,
        label_content: str,
    ):
        image = self.dataset_dir / "images" / split / name
        label = self.dataset_dir / "labels" / split / f"{Path(name).stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(image_content)
        label.write_text(label_content, encoding="utf-8")
        return image, label

    def _build_quality_fixture(self):
        train_a, _ = self._write_pair(
            "train", "a.png", b"cross-duplicate", "0 0.5 0.5 0.2 0.2\n"
        )
        self._write_pair(
            "train", "b.png", b"cross-duplicate", "0 0.5 0.5 0.2 0.2\n"
        )
        val_c, _ = self._write_pair(
            "val", "c.png", b"cross-duplicate", "0 0.5 0.5 0.2 0.2\n"
        )
        self._write_pair(
            "train", "d.png", b"same-split", "1 0.5 0.5 0.2 0.2\n"
        )
        self._write_pair(
            "train", "e.png", b"same-split", "1 0.5 0.5 0.2 0.2\n"
        )
        return train_a, val_c

    def test_reports_exact_duplicates_leakage_and_class_distribution(self):
        train_a, _val_c = self._build_quality_fixture()

        report = DatasetQualityScanner(self.paths).scan()

        self.assertEqual(len(report.duplicate_groups), 2)
        cross_group = next(group for group in report.duplicate_groups if group.cross_split)
        same_group = next(group for group in report.duplicate_groups if not group.cross_split)
        self.assertEqual(len(cross_group.image_paths), 3)
        self.assertEqual(cross_group.splits, ("train", "val"))
        self.assertEqual(len(same_group.image_paths), 2)

        findings = {finding.issue.code: finding for finding in report.findings}
        self.assertIs(
            findings["dataset.content_split_overlap"].issue.severity,
            IssueSeverity.ERROR,
        )
        self.assertEqual(
            findings["dataset.content_split_overlap"].image_path,
            train_a,
        )
        self.assertIs(
            findings["dataset.exact_duplicate"].issue.severity,
            IssueSeverity.WARNING,
        )
        self.assertNotIn("class.missing_in_val", findings)
        self.assertIn("class.low_sample_count", findings)
        self.assertIn("class.unused", findings)

        distributions = {
            item.class_id: item for item in report.class_distributions
        }
        self.assertEqual(distributions[0].box_count, 3)
        self.assertEqual(distributions[0].image_count, 3)
        self.assertEqual(distributions[0].train_boxes, 2)
        self.assertEqual(distributions[0].val_boxes, 1)
        self.assertEqual(distributions[0].train_images, 2)
        self.assertEqual(distributions[0].val_images, 1)
        self.assertEqual(distributions[1].train_boxes, 2)
        self.assertEqual(distributions[1].val_boxes, 0)
        self.assertEqual(distributions[1].train_images, 2)
        self.assertEqual(distributions[1].val_images, 0)
        self.assertEqual(distributions[2].box_count, 0)

    def test_missing_val_requires_enough_distinct_images(self):
        for index in range(5):
            self._write_pair(
                "train",
                f"train-{index}.png",
                f"train-{index}".encode(),
                "0 0.5 0.5 0.2 0.2\n",
            )
        self._write_pair(
            "train",
            "train-extra.png",
            b"train-extra",
            "0 0.5 0.5 0.2 0.2\n",
        )
        self._write_pair(
            "val",
            "val-other.png",
            b"val-other",
            "1 0.5 0.5 0.2 0.2\n",
        )

        report = DatasetQualityScanner(self.paths).scan()
        codes = [finding.issue.code for finding in report.findings]

        self.assertIn("class.missing_in_val", codes)
        self.assertNotIn("class.low_sample_count", [
            finding.issue.code
            for finding in report.findings
            if "类别 0:" in finding.issue.message
        ])

    def test_many_boxes_in_one_image_still_reports_low_sample(self):
        boxes = "".join(
            f"0 0.{index + 1} 0.5 0.1 0.1\n" for index in range(6)
        )
        self._write_pair("train", "many.png", b"many", boxes)

        report = DatasetQualityScanner(self.paths).scan()
        distribution = report.class_distributions[0]
        findings = [
            finding
            for finding in report.findings
            if finding.issue.code == "class.low_sample_count"
            and "类别 0:" in finding.issue.message
        ]

        self.assertEqual(distribution.train_boxes, 6)
        self.assertEqual(distribution.train_images, 1)
        self.assertEqual(len(findings), 1)

    def test_reuses_label_integrity_findings(self):
        self._write_pair("train", "bad.png", b"bad", "0 0.5 0.5 0.2\n")

        report = DatasetQualityScanner(self.paths).scan()
        codes = {finding.issue.code for finding in report.findings}

        self.assertIn("label.field_count", codes)

    def test_scan_does_not_modify_dataset_files(self):
        self._build_quality_fixture()
        files = [path for path in self.dataset_dir.rglob("*") if path.is_file()]
        before = {
            path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files
        }

        DatasetQualityScanner(self.paths).scan()

        after = {
            path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files
        }
        self.assertEqual(before, after)

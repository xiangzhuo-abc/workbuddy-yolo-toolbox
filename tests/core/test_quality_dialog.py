import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication

from core.dataset_quality import (
    ClassDistribution,
    DatasetQualityReport,
    DuplicateGroup,
    QualityFinding,
)
from core.issues import Issue, IssueSeverity
from yolo_quality_dialog import DatasetQualityDialog


class DatasetQualityDialogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name) / "dataset"
        self.image_path = self.dataset_dir / "images" / "train" / "a.png"
        self.image_path.parent.mkdir(parents=True)
        self.image_path.write_bytes(b"image")
        issue = Issue(
            code="dataset.exact_duplicate",
            severity=IssueSeverity.WARNING,
            message="发现重复图片",
            path=self.image_path,
            suggested_action="检查重复样本",
        )
        self.report = DatasetQualityReport(
            dataset_dir=self.dataset_dir,
            image_count=2,
            findings=(QualityFinding(issue, self.image_path),),
            duplicate_groups=(
                DuplicateGroup(
                    digest="abc",
                    image_paths=(self.image_path, self.image_path.with_name("b.png")),
                    splits=("train",),
                    cross_split=False,
                ),
            ),
            class_distributions=(
                ClassDistribution(
                    class_id=0,
                    name="按钮",
                    image_count=2,
                    box_count=3,
                    train_boxes=3,
                    val_boxes=0,
                    test_boxes=0,
                ),
            ),
        )
        self.dialog = DatasetQualityDialog(
            self.dataset_dir,
            auto_start=False,
        )

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def test_dialog_builds_three_views_and_applies_report(self):
        self.dialog._apply_report(self.report)

        self.assertEqual(self.dialog.tabs.count(), 3)
        self.assertEqual(
            [self.dialog.tabs.tabText(i) for i in range(3)],
            ["问题", "类别分布", "重复图片"],
        )
        self.assertEqual(self.dialog.issues_table.rowCount(), 1)
        self.assertEqual(self.dialog.classes_table.rowCount(), 1)
        self.assertEqual(self.dialog.duplicates_table.rowCount(), 1)
        self.assertEqual(self.dialog.classes_table.columnCount(), 10)
        self.assertEqual(self.dialog.summary_badge.property("tone"), "warning")
        self.assertIn("警告 1", self.dialog.summary_badge.text())
        self.assertFalse(self.dialog.optimize_button.isEnabled())

    def test_selected_issue_emits_annotator_location(self):
        received = []
        self.dialog.locate_requested.connect(received.append)
        self.dialog._apply_report(self.report)

        self.dialog.issues_table.selectRow(0)
        self.dialog._update_action_state()
        self.dialog.locate_button.click()

        self.assertEqual(received, [str(self.image_path)])

    def test_fixable_distribution_emits_optimize_request(self):
        issue = Issue(
            code="class.missing_in_val",
            severity=IssueSeverity.WARNING,
            message="类别未进入验证集",
            path=self.dataset_dir / "classes.txt",
            suggested_action="优化划分",
        )
        report = DatasetQualityReport(
            dataset_dir=self.dataset_dir,
            image_count=6,
            findings=(QualityFinding(issue),),
            duplicate_groups=(),
            class_distributions=(),
        )
        received = []
        self.dialog.optimize_split_requested.connect(received.append)

        self.dialog._apply_report(report)
        self.dialog.optimize_button.click()

        self.assertTrue(self.dialog.optimize_button.isEnabled())
        self.assertEqual(received, ["repair"])

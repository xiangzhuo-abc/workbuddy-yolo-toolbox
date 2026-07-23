import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication

from core.dataset_split import SplitMode
from yolo_split_dialog import DatasetSplitDialog


class DatasetSplitDialogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name) / "dataset"
        self.dataset_dir.mkdir()
        self.dialog = DatasetSplitDialog(
            self.dataset_dir,
            initial_mode=SplitMode.REPAIR,
            auto_start=False,
        )

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def _plan(self, *, executable=True, moves=1):
        coverage = SimpleNamespace(
            class_id=0,
            name="按钮",
            total_images=6,
            total_boxes=8,
            current_train=6,
            current_val=0,
            current_test=0,
            planned_train=5,
            planned_val=1,
            planned_test=0,
            requirements_met=True,
        )
        move = SimpleNamespace(
            image_path=Path("D:/dataset/images/train/a.png"),
            source_split="train",
            target_split="val",
            class_ids=(0,),
        )
        return SimpleNamespace(
            is_executable=executable,
            plan_id="a" * 64,
            samples=(object(),) * 6,
            moves=(move,) * moves,
            class_coverages=(coverage,),
            current_counts=(("train", 6), ("val", 0), ("test", 0)),
            target_counts=(("train", 5), ("val", 1), ("test", 0)),
            planned_counts=(("train", 5), ("val", 1), ("test", 0)),
            blocking_issues=() if executable else (
                SimpleNamespace(code="dataset.error", message="数据错误"),
            ),
            risks=(),
        )

    def test_builds_settings_and_three_preview_tabs(self):
        self.assertEqual(self.dialog.preview_tabs.count(), 3)
        self.assertEqual(
            [
                self.dialog.preview_tabs.tabText(index)
                for index in range(self.dialog.preview_tabs.count())
            ],
            ["摘要", "类别覆盖", "移动清单"],
        )
        self.assertTrue(self.dialog.repair_mode_button.isChecked())
        self.assertFalse(self.dialog.execute_button.isEnabled())

    def test_valid_preview_enables_execute_and_populates_tables(self):
        self.dialog._apply_plan(self._plan())

        self.assertEqual(self.dialog.pages.currentIndex(), 1)
        self.assertEqual(self.dialog.summary_table.rowCount(), 3)
        self.assertEqual(self.dialog.classes_table.rowCount(), 1)
        self.assertEqual(self.dialog.moves_table.rowCount(), 1)
        self.assertTrue(self.dialog.execute_button.isEnabled())
        self.assertIn("移动 1", self.dialog.status_badge.text())

    def test_blocking_preview_keeps_execute_disabled(self):
        self.dialog._apply_plan(self._plan(executable=False))

        self.assertFalse(self.dialog.execute_button.isEnabled())
        self.assertEqual(self.dialog.status_badge.property("tone"), "danger")

    def test_full_mode_button_changes_selected_mode(self):
        self.dialog.full_mode_button.click()

        self.assertIs(self.dialog.selected_mode(), SplitMode.FULL)

    def test_completed_or_failed_execution_requires_new_preview(self):
        self.dialog._apply_plan(self._plan())
        result = SimpleNamespace(moved_pairs=1, backup_dir=Path("D:/backup"))

        self.dialog._show_success(result)
        self.dialog._finish_worker()
        self.assertFalse(self.dialog.execute_button.isEnabled())

        self.dialog._apply_plan(self._plan())
        self.dialog._show_failure("计划已过期")
        self.dialog._finish_worker()
        self.assertFalse(self.dialog.execute_button.isEnabled())

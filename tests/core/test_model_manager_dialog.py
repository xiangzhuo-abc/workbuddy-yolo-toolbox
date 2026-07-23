from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication, QMessageBox

from core.model_registry import get_model
from yolo_model_manager import ModelDownloadThread, ModelManagerDialog


class ModelManagerDialogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name) / "models"
        self.dialog = ModelManagerDialog(self.models_dir)
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def test_dialog_lists_detection_models_and_controls(self):
        self.assertEqual(self.dialog.models_table.rowCount(), 10)
        self.assertEqual(self.dialog.models_table.columnCount(), 5)
        self.assertEqual(self.dialog.models_table.item(0, 0).text(), "YOLOv8")
        self.assertEqual(self.dialog.models_table.item(5, 0).text(), "YOLO11")
        self.assertTrue(self.dialog.download_button.isEnabled())
        self.assertFalse(self.dialog.cancel_button.isVisible())

    def test_running_state_exposes_cancel_and_locks_selection(self):
        self.dialog._set_running(True)
        self.dialog.show()
        self.app.processEvents()

        self.assertFalse(self.dialog.models_table.isEnabled())
        self.assertFalse(self.dialog.download_button.isEnabled())
        self.assertTrue(self.dialog.cancel_button.isVisible())
        self.assertTrue(self.dialog.cancel_button.isEnabled())

        self.dialog._set_running(False)
        self.assertTrue(self.dialog.models_table.isEnabled())
        self.assertFalse(self.dialog.cancel_button.isVisible())

    def test_existing_file_requires_overwrite_confirmation(self):
        self.models_dir.mkdir(parents=True)
        selected = self.dialog._selected_model()
        (self.models_dir / selected.filename).write_bytes(b"invalid")
        with patch(
            "yolo_model_manager.QMessageBox.question",
            return_value=QMessageBox.No,
        ) as question:
            self.dialog._download_selected()

        question.assert_called_once()
        self.assertIsNone(self.dialog._thread)

    def test_selection_validates_only_current_model(self):
        with patch("yolo_model_manager.installed_models") as scan_all, patch(
            "yolo_model_manager.validate_model_file", return_value=False
        ) as validate_one:
            self.dialog.models_table.selectRow(1)
            self.app.processEvents()

        scan_all.assert_not_called()
        validate_one.assert_called_with(
            self.models_dir / "yolov8s.pt",
            get_model("yolov8s.pt"),
        )

    def test_download_thread_keeps_result_until_finished_signal(self):
        expected = Mock()
        thread = ModelDownloadThread(get_model("yolo11n.pt"), self.models_dir)
        with patch("yolo_model_manager.download_model", return_value=expected):
            thread.run()

        self.assertIs(thread.result, expected)
        self.assertEqual(thread.error_message, "")
        self.assertFalse(thread.cancelled)

    def test_invalid_existing_file_is_reported(self):
        self.models_dir.mkdir(parents=True)
        selected = self.dialog._selected_model()
        (self.models_dir / selected.filename).write_bytes(b"invalid")

        self.dialog._refresh_table()

        self.assertEqual(self.dialog.models_table.item(0, 4).text(), "文件异常")
        self.assertIn("校验失败", self.dialog.detail_label.text())
        self.assertEqual(self.dialog.download_button.text(), "重新下载")

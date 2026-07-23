import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication

import yolo_detect_gui as gui


class DetectGuiIntegrationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir()
        self.image_a = self.base_dir / "a.png"
        self.image_b = self.base_dir / "b.png"
        self._write_image(self.image_a, "white")
        self._write_image(self.image_b, "lightgray")
        self.patches = (
            patch.object(gui, "setup_detect_logging", return_value=None),
            patch.object(gui.YoloDetectGUI, "_scan_models", return_value=None),
        )
        for item in self.patches:
            item.start()
        self.window = None

    def tearDown(self):
        if self.window is not None:
            self.window._stop_detect_worker()
            self.window.deleteLater()
            self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def _write_image(self, path: Path, color: str):
        image = QImage(160, 100, QImage.Format_RGB32)
        image.fill(QColor(color))
        self.assertTrue(image.save(str(path)))

    def _create_window(self):
        self.window = gui.YoloDetectGUI(runs_dir=self.runs_dir)
        self.app.processEvents()
        return self.window

    def _load_directory_images(self, window):
        window.image_paths = [self.image_a, self.image_b]
        window.current_idx = 0
        window._refresh_image_combo()
        window._load_and_show()

    def test_detect_window_builds_setup_workspace_and_inspector(self):
        window = self._create_window()

        self.assertEqual(window.setup_panel.objectName(), "DetectionSetup")
        self.assertEqual(window.main_splitter.count(), 2)
        self.assertEqual(window.workspace_panel.objectName(), "DetectionWorkspace")
        self.assertEqual(window.inspector_panel.objectName(), "DetectionInspector")
        for name in (
            "model_combo",
            "btn_load_model",
            "btn_browse_model",
            "btn_select_image",
            "btn_select_dir",
            "btn_detect",
            "result_list",
            "stats_label",
            "status_label",
        ):
            self.assertTrue(hasattr(window, name), name)
        self.assertFalse(window.btn_save_result.isEnabled())

    def test_image_combo_locates_directory_images(self):
        window = self._create_window()
        self._load_directory_images(window)

        self.assertEqual(window.image_combo.count(), 2)
        window.image_combo.setCurrentIndex(1)
        self.app.processEvents()

        self.assertEqual(window.current_idx, 1)
        self.assertIn("b.png", window.img_label.text())

    def test_image_combo_cannot_switch_during_detection(self):
        window = self._create_window()
        self._load_directory_images(window)
        window._detect_active_request = {"id": 7}

        window.image_combo.setCurrentIndex(1)
        self.app.processEvents()

        self.assertEqual(window.current_idx, 0)
        self.assertEqual(window.image_combo.currentIndex(), 0)

    def test_detect_result_updates_elapsed_and_summary(self):
        window = self._create_window()
        window._detect_started_at = 100.0
        with patch.object(gui.time, "perf_counter", return_value=100.125):
            window._apply_detect_result({
                "names": {"0": "player"},
                "detections": [{
                    "class_id": 0,
                    "name": "player",
                    "conf": 0.91,
                    "xyxy": [10, 10, 50, 60],
                }],
            })

        self.assertIn("125 ms", window.elapsed_label.text())
        self.assertIn("1 个目标", window.stats_label.text())
        self.assertEqual(window.result_list.count(), 1)
        self.assertTrue(window.btn_save_result.isEnabled())

    def test_detection_does_not_start_worker_without_runtime(self):
        window = self._create_window()
        with patch.object(
            gui,
            "ensure_ml_runtime",
            return_value=False,
        ) as ensure, patch.object(gui, "QProcess") as process_class:
            result = window._ensure_detect_worker()

        self.assertFalse(result)
        ensure.assert_called_once_with(window, "模型测试")
        process_class.assert_not_called()

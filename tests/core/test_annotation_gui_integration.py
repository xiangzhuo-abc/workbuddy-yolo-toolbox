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

import yolo_annotate_gui as gui
import yolo_annotate as legacy_gui


class AnnotationGuiIntegrationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name) / "dataset"
        self.image_dir = self.dataset_dir / "images" / "train"
        self.label_dir = self.dataset_dir / "labels" / "train"
        self.image_dir.mkdir(parents=True)
        self.label_dir.mkdir(parents=True)
        (self.dataset_dir / "classes.txt").write_text("按钮\n", encoding="utf-8")
        self._write_image("a.png")
        self._write_image("b.png")
        self.config_patches = (
            patch.object(gui.dataset_tools, "load_config", return_value={}),
            patch.object(gui.dataset_tools, "save_config", return_value=None),
        )
        for config_patch in self.config_patches:
            config_patch.start()
        self.window = None

    def tearDown(self):
        if self.window is not None:
            self.window._stop_detect_worker()
            self.window.deleteLater()
            self.app.processEvents()
        for config_patch in reversed(self.config_patches):
            config_patch.stop()
        self.temp_dir.cleanup()

    def _write_image(self, name: str):
        image = QImage(100, 80, QImage.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(self.image_dir / name)))

    def _create_window(self):
        self.window = gui.YoloAnnotatorGUI(dataset_dir=self.dataset_dir)
        self.app.processEvents()
        return self.window

    def test_damaged_label_loads_valid_boxes_and_can_be_saved(self):
        label_path = self.label_dir / "a.txt"
        label_path.write_text(
            "0 0.5 0.5 0.2 0.4\ninvalid line\n",
            encoding="utf-8",
        )

        window = self._create_window()

        self.assertEqual(window.image_label.boxes, [(40, 24, 60, 56, 0)])
        self.assertIn("标签问题 1 处", window.status_label.text())
        self.assertTrue(window.save_labels())
        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "0 0.500000 0.500000 0.200000 0.400000\n",
        )

    def test_failed_save_blocks_navigation_and_preserves_current_index(self):
        (self.label_dir / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n",
            encoding="utf-8",
        )
        window = self._create_window()
        self.assertEqual(window.current_idx, 0)

        with patch.object(
            gui,
            "save_pixel_boxes_atomic",
            side_effect=OSError("injected save failure"),
        ), patch.object(gui.QMessageBox, "critical", return_value=gui.QMessageBox.Ok):
            window.next_image()

        self.assertEqual(window.current_idx, 0)
        self.assertIn("保存失败", window.status_label.text())

    def test_class_reorder_updates_classes_and_labels_together(self):
        (self.dataset_dir / "classes.txt").write_text(
            "按钮\n图标\n", encoding="utf-8"
        )
        label_path = self.label_dir / "a.txt"
        label_path.write_text(
            "0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.1\n",
            encoding="utf-8",
        )
        window = self._create_window()

        with patch.object(window, "_backup_label_state", return_value=True):
            result = window._apply_class_reorder(["图标", "按钮"])

        self.assertTrue(result)
        self.assertEqual(window.class_names, ["图标", "按钮"])
        self.assertEqual(
            (self.dataset_dir / "classes.txt").read_text(encoding="utf-8"),
            "图标\n按钮\n",
        )
        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "1 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.1\n",
        )

    def test_legacy_annotator_uses_the_same_atomic_label_service(self):
        annotator = object.__new__(legacy_gui.YoloAnnotator)
        annotator.image_paths = [self.image_dir / "a.png"]
        annotator.current_idx = 0
        annotator.label_dir = self.label_dir
        annotator.class_names = ["按钮"]
        annotator.boxes = [(10, 20, 30, 40, 0)]

        self.assertTrue(annotator._save_labels())
        self.assertEqual(
            (self.label_dir / "a.txt").read_text(encoding="utf-8"),
            "0 0.200000 0.375000 0.200000 0.250000\n",
        )

        (self.label_dir / "a.txt").write_text(
            "0 0.2 0.375 0.2 0.25\nbroken line\n",
            encoding="utf-8",
        )
        annotator._load_labels(self.image_dir / "a.png")
        self.assertEqual(annotator.boxes, [(10, 20, 30, 40, 0)])

    def test_preannotation_adds_new_classes_in_one_atomic_update(self):
        window = self._create_window()
        data = {
            "detections": [
                {"name": "按钮", "xyxy": [10, 10, 30, 30], "conf": 0.9},
                {"name": "图标", "xyxy": [60, 40, 90, 70], "conf": 0.8},
            ]
        }

        with patch.object(window, "_save_classes", wraps=window._save_classes) as save:
            window._apply_auto_annotations(data)

        self.assertEqual(save.call_count, 1)
        self.assertEqual(window.class_names, ["按钮", "图标"])
        self.assertEqual(
            window.image_label.boxes,
            [(10, 10, 30, 30, 0), (60, 40, 90, 70, 1)],
        )

    def test_preannotation_does_not_start_worker_without_runtime(self):
        window = self._create_window()
        with patch.object(
            gui,
            "ensure_ml_runtime",
            return_value=False,
        ) as ensure, patch.object(gui, "QProcess") as process_class:
            result = window._ensure_detect_worker()

        self.assertFalse(result)
        ensure.assert_called_once_with(window, "自动预标注")
        process_class.assert_not_called()

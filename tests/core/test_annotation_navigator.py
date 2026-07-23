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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication

import yolo_annotate_gui as gui


class AnnotationNavigatorTests(TestCase):
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

    def _write_image(self, name: str, split: str = "train"):
        image_dir = self.dataset_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        image = QImage(120, 80, QImage.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(image_dir / name)))
        return image_dir / name

    def _create_window(self):
        self.window = gui.YoloAnnotatorGUI(dataset_dir=self.dataset_dir)
        self.app.processEvents()
        return self.window

    def _item_for_index(self, window, index):
        return next(
            window.image_list.item(row)
            for row in range(window.image_list.count())
            if window.image_list.item(row).data(Qt.UserRole) == index
        )

    def test_navigator_lists_all_images_and_filters_unlabeled(self):
        (self.label_dir / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        window = self._create_window()

        self.assertEqual(window.image_list.count(), 2)
        self.assertIn("已标注", self._item_for_index(window, 0).text())
        self.assertIn("未标注", self._item_for_index(window, 1).text())
        self.assertIn("[已标注]", self._item_for_index(window, 0).text())

        window.image_filter_combo.setCurrentText("未标注")
        window._refresh_image_navigator()

        self.assertEqual(window.image_list.count(), 1)
        self.assertIn("b.png", window.image_list.item(0).text())
        self.assertEqual(window.navigator_summary_label.text(), "显示 1 / 2")

    def test_navigator_blocks_switch_when_current_save_fails(self):
        window = self._create_window()
        target = self._item_for_index(window, 1)

        with patch.object(window, "save_labels", return_value=False):
            window.on_image_item_selected(target)

        self.assertEqual(window.current_idx, 0)
        self.assertEqual(window.image_list.currentItem().data(Qt.UserRole), 0)

    def test_navigator_switches_after_current_image_is_saved(self):
        window = self._create_window()
        target = self._item_for_index(window, 1)

        with patch.object(window, "save_labels", return_value=True) as save:
            window.on_image_item_selected(target)

        save.assert_called_once_with()
        self.assertEqual(window.current_idx, 1)
        self.assertEqual(window.image_list.currentItem().data(Qt.UserRole), 1)

    def test_successful_save_refreshes_current_item_status(self):
        window = self._create_window()
        self.assertIn("未标注", self._item_for_index(window, 0).text())
        window.image_label.boxes = [(20, 20, 80, 60, 0)]

        self.assertTrue(window.save_labels())

        self.assertIn("已标注", self._item_for_index(window, 0).text())
        self.assertEqual(window.image_list.currentItem().data(Qt.UserRole), 0)

    def test_unsaved_preannotation_does_not_change_navigation_status(self):
        window = self._create_window()
        data = {
            "detections": [
                {"name": "按钮", "xyxy": [10, 10, 40, 40], "conf": 0.9},
            ]
        }

        window._apply_auto_annotations(data)

        self.assertIn("未标注", self._item_for_index(window, 0).text())

    def test_initial_image_opens_its_split_and_location(self):
        target = self._write_image("target.png", split="val")
        self.window = gui.YoloAnnotatorGUI(
            dataset_dir=self.dataset_dir,
            initial_image=target,
        )
        self.app.processEvents()

        self.assertEqual(self.window.image_dir, self.dataset_dir / "images" / "val")
        self.assertEqual(self.window.image_paths[self.window.current_idx], target)

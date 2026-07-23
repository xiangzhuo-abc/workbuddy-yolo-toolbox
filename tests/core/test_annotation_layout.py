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


class AnnotationLayoutTests(TestCase):
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
        image = QImage(160, 100, QImage.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(self.image_dir / "a.png")))
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

    def _create_window(self):
        self.window = gui.YoloAnnotatorGUI(dataset_dir=self.dataset_dir)
        self.app.processEvents()
        return self.window

    def test_annotator_builds_three_columns_and_preserves_controls(self):
        window = self._create_window()

        self.assertEqual(window.main_splitter.count(), 3)
        self.assertEqual(window.navigator_panel.objectName(), "ImageNavigator")
        self.assertEqual(window.workspace_panel.objectName(), "AnnotationWorkspace")
        self.assertEqual(window.inspector_panel.objectName(), "AnnotationInspector")
        self.assertEqual(window.inspector_tabs.count(), 3)
        self.assertEqual(
            [window.inspector_tabs.tabText(i) for i in range(3)],
            ["标注框", "类别", "预标注"],
        )
        for name in (
            "btn_prev",
            "btn_next",
            "btn_next_unlabeled",
            "btn_save",
            "btn_fit",
            "btn_open",
            "class_list",
            "box_list",
            "model_combo",
            "btn_auto_annotate",
            "status_label",
        ):
            self.assertTrue(hasattr(window, name), name)
        self.assertEqual(window.btn_prev.text(), "")
        self.assertEqual(window.btn_next.text(), "")
        self.assertEqual(window.btn_go_to_image.text(), "")
        self.assertEqual(window.btn_next_unlabeled.text(), "")
        self.assertTrue(window.btn_prev.toolTip())
        self.assertTrue(window.btn_next_unlabeled.toolTip())

    def test_box_list_and_canvas_selection_stay_synchronized(self):
        window = self._create_window()
        window.image_label.boxes = [
            (10, 10, 90, 80, 0),
            (20, 20, 40, 35, 0),
        ]

        window.refresh_box_list()
        window.select_box(1)

        self.assertEqual(window.box_list.currentRow(), 1)
        self.assertEqual(window.image_label.selected_box_index, 1)

    def test_preannotation_model_list_includes_models_directory_weights(self):
        window = self._create_window()
        model_dir = Path(self.temp_dir.name) / "models"
        model_dir.mkdir()
        downloaded = model_dir / "yolo11n.pt"
        downloaded.write_bytes(b"model")

        with patch.object(window, "_model_search_dirs", return_value=[model_dir]):
            window._populate_model_combo()

        values = [
            window.model_combo.itemData(index)
            for index in range(window.model_combo.count())
        ]
        self.assertIn(str(downloaded), values)

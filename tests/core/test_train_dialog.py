import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication, QDialogButtonBox

import yolo_tool_launcher as launcher


class TrainDialogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        launcher.theme.apply_app_theme(cls.app)

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.dataset_dir = self.base_dir / "dataset"
        self.runs_dir = self.base_dir / "runs"
        self.dataset_dir.mkdir()
        self.runs_dir.mkdir()
        self.data_yaml = self.dataset_dir / "data.yaml"
        self.data_yaml.write_text("path: .\n", encoding="utf-8")
        self.model_path = self.base_dir / "yolov8n.pt"
        self.model_path.write_bytes(b"test-model")
        self.summary = {
            "train_images": 32,
            "val_images": 8,
            "test_images": 4,
            "n_classes": 3,
        }
        self.summary_error = None
        self.patches = (
            patch.object(
                launcher.tools,
                "find_data_configs",
                return_value=[self.data_yaml],
            ),
            patch.object(
                launcher.tools,
                "detect_devices",
                return_value=[("CPU（测试）", "cpu"), ("自动选择", "")],
            ),
            patch.object(
                launcher.tools,
                "data_yaml_summary",
                side_effect=self._summary_result,
            ),
        )
        for item in self.patches:
            item.start()
        self.dialog = None

    def tearDown(self):
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()
            self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def _summary_result(self, _path):
        if self.summary_error is not None:
            raise self.summary_error
        return dict(self.summary)

    def _create_dialog(self, models=None):
        self.dialog = launcher.TrainDialog(
            None,
            models or [self.model_path],
            dataset_dir=self.dataset_dir,
            runs_dir=str(self.runs_dir),
        )
        self.dialog.show()
        self.app.processEvents()
        return self.dialog

    def test_dialog_builds_source_tabs_footer_and_preserves_contracts(self):
        dialog = self._create_dialog()

        self.assertEqual(dialog.source_panel.objectName(), "TrainingSource")
        self.assertEqual(dialog.settings_tabs.objectName(), "TrainingTabs")
        self.assertEqual(dialog.settings_tabs.count(), 3)
        self.assertEqual(
            [dialog.settings_tabs.tabText(i) for i in range(3)],
            ["基础参数", "高级参数", "数据检查"],
        )
        self.assertEqual(dialog.basic_tab.objectName(), "TrainingBasic")
        self.assertEqual(dialog.advanced_tab.objectName(), "TrainingAdvanced")
        self.assertEqual(dialog.data_check_tab.objectName(), "TrainingDataCheck")
        self.assertEqual(dialog.footer_panel.objectName(), "TrainingFooter")

        for name in (
            "model_combo",
            "data_combo",
            "epochs_spin",
            "batch_spin",
            "imgsz_spin",
            "device_combo",
            "name_edit",
            "resume_check",
            "summary_label",
            "training_estimate_label",
        ):
            self.assertTrue(hasattr(dialog, name), name)

        self.assertEqual(dialog.model_combo.currentData(), str(self.model_path))
        self.assertEqual(dialog.data_combo.currentData(), str(self.data_yaml))
        self.assertEqual(dialog.epochs_spin.value(), 20)
        self.assertEqual(dialog.batch_spin.value(), 4)
        self.assertEqual(dialog.imgsz_spin.value(), 640)
        self.assertEqual(dialog.device_combo.currentData(), "cpu")
        self.assertEqual(dialog.name_edit.text(), "coc_detect")
        self.assertFalse(dialog.resume_check.isChecked())
        self.assertIn(str(self.runs_dir), dialog.output_label.text())
        self.assertEqual(
            dialog.button_box.button(QDialogButtonBox.Ok).text(),
            "开始训练",
        )
        self.assertIn("训练 8", dialog.training_estimate_label.text())
        self.assertIn("验证 2", dialog.training_estimate_label.text())
        self.assertIn("共 10", dialog.training_estimate_label.text())

    def test_gpu_defaults_to_batch_16_and_preserves_custom_batch(self):
        launcher.tools.detect_devices.return_value = [
            ("GPU 0: Test GPU（推荐）", "0"),
            ("自动选择（由 Ultralytics 决定）", ""),
            ("CPU（兼容模式，速度较慢）", "cpu"),
        ]
        dialog = self._create_dialog()

        self.assertEqual(dialog.device_combo.currentData(), "0")
        self.assertEqual(dialog.batch_spin.value(), 16)
        self.assertIn("训练 2", dialog.training_estimate_label.text())
        self.assertIn("验证 1", dialog.training_estimate_label.text())

        dialog.device_combo.setCurrentIndex(2)
        self.app.processEvents()
        self.assertEqual(dialog.batch_spin.value(), 4)

        dialog.device_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(dialog.batch_spin.value(), 16)

        dialog.batch_spin.setValue(12)
        dialog.device_combo.setCurrentIndex(2)
        self.app.processEvents()
        self.assertEqual(dialog.batch_spin.value(), 12)

    def test_prefers_lightweight_model_and_warns_when_large_model_is_selected(self):
        large_model = self.base_dir / "yolov8l.pt"
        large_model.write_bytes(b"large-model")
        dialog = self._create_dialog([large_model, self.model_path])

        self.assertEqual(dialog.model_combo.currentData(), str(self.model_path))
        self.assertEqual(dialog.model_hint_label.property("tone"), "success")
        self.assertIn("快速", dialog.model_hint_label.text())

        large_index = dialog.model_combo.findData(str(large_model))
        dialog.model_combo.setCurrentIndex(large_index)
        self.app.processEvents()

        self.assertEqual(dialog.model_hint_label.property("tone"), "warning")
        self.assertIn("明显增加", dialog.model_hint_label.text())

    def test_summary_status_tracks_success_warning_and_danger(self):
        dialog = self._create_dialog()

        self.assertEqual(dialog.summary_label.property("tone"), "success")
        self.assertEqual(dialog.data_status_badge.property("tone"), "success")
        self.assertEqual(dialog.data_status_badge.text(), "数据可用")
        self.assertTrue(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

        self.summary["n_classes"] = 0
        dialog._refresh_summary()
        self.assertEqual(dialog.summary_label.property("tone"), "danger")
        self.assertEqual(dialog.data_status_badge.property("tone"), "danger")
        self.assertEqual(dialog.data_status_badge.text(), "类别不可用")
        self.assertFalse(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

        self.summary["n_classes"] = 3
        self.summary["val_images"] = 0
        dialog._refresh_summary()
        self.assertEqual(dialog.summary_label.property("tone"), "warning")
        self.assertEqual(dialog.data_status_badge.property("tone"), "warning")
        self.assertEqual(dialog.data_status_badge.text(), "建议划分验证集")
        self.assertTrue(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

        self.summary["train_images"] = 0
        dialog._refresh_summary()
        self.assertEqual(dialog.summary_label.property("tone"), "danger")
        self.assertEqual(dialog.data_status_badge.property("tone"), "danger")
        self.assertEqual(dialog.data_status_badge.text(), "训练集不可用")
        self.assertFalse(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

        self.summary_error = ValueError("配置损坏")
        dialog._refresh_summary()
        self.assertEqual(dialog.summary_label.property("tone"), "danger")
        self.assertEqual(dialog.data_status_badge.property("tone"), "danger")
        self.assertEqual(dialog.data_status_badge.text(), "读取失败")
        self.assertFalse(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

    def test_browse_adds_external_config_and_refreshes_summary(self):
        dialog = self._create_dialog()
        external_yaml = self.base_dir / "external.yaml"
        external_yaml.write_text("path: dataset\n", encoding="utf-8")

        with patch.object(
            launcher.QFileDialog,
            "getOpenFileName",
            return_value=(str(external_yaml), "YAML 配置文件"),
        ):
            dialog._browse_data_config()

        self.assertEqual(dialog.data_combo.count(), 2)
        self.assertEqual(dialog.data_combo.currentData(), str(external_yaml))
        self.assertIn("训练集 32 张", dialog.summary_label.text())


if __name__ == "__main__":
    unittest.main()

import os
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

import yolo_tool_launcher as launcher


class LauncherUiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.config = {
            "dataset_dir": str(base / "dataset"),
            "runs_dir": str(base / "runs"),
            "source_dir": "",
        }
        self.window = None

    def tearDown(self):
        if self.window is not None:
            self.window.close()
            self.window.deleteLater()
            self.app.processEvents()
        self.temp_dir.cleanup()

    def _create_window(self):
        patches = (
            patch.object(launcher.tools, "setup_logging"),
            patch.object(
                launcher.tools,
                "check_dependencies",
                return_value=([], {"PyQt5": "已安装"}),
            ),
            patch.object(launcher.tools, "load_config", return_value=self.config),
        )
        for config_patch in patches:
            config_patch.start()
            self.addCleanup(config_patch.stop)

        self.window = launcher.YoloToolLauncher()
        self.app.processEvents()
        return self.window

    def test_launcher_builds_workflow_console_and_preserves_contracts(self):
        window = self._create_window()

        self.assertEqual(window.workflow_panel.objectName(), "WorkflowNav")
        self.assertEqual(window.console_panel.objectName(), "ConsolePanel")
        self.assertIs(window.dataset_edit, window.dataset_path_bar.line_edit)
        self.assertIs(window.runs_edit, window.runs_path_bar.line_edit)
        self.assertEqual(
            set(window.func_buttons),
            {
                "数据集准备",
                "数据集划分",
                "启动标注工具",
                "格式校验",
                "数据质量检查",
                "数据集统计",
                "环境体检",
                "发布前自检",
                "备份恢复",
                "查看日志",
                "模型训练",
                "模型测试",
                "模型评估",
                "模型管理",
                "运行环境管理",
                "TensorBoard",
                "导出诊断",
                "关于",
            },
        )
        self.assertEqual(window.main_splitter.count(), 2)
        self.assertIn("PyQt5: 正常", window.dep_label.text())
        self.assertEqual(window.log_text.lineWrapMode(), window.log_text.NoWrap)
        self.assertEqual(window.stop_train_button.objectName(), "StopTrainingButton")
        self.assertFalse(window.stop_train_button.isEnabled())

    def test_stop_training_control_tracks_state_and_confirms_before_stopping(self):
        window = self._create_window()

        window._set_train_control_state(True)
        self.assertTrue(window.stop_train_button.isEnabled())
        self.assertEqual(window.stop_train_button.text(), "停止训练")

        window._set_train_control_state(True, stopping=True)
        self.assertFalse(window.stop_train_button.isEnabled())
        self.assertEqual(window.stop_train_button.text(), "正在停止")

        window._set_train_control_state(False)
        self.assertFalse(window.stop_train_button.isEnabled())
        self.assertEqual(window.stop_train_button.text(), "停止训练")

        window.train_process = object()
        try:
            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.Yes,
            ), patch.object(window, "_stop_train_process") as stop_process:
                window._on_stop_train_clicked()

            stop_process.assert_called_once_with()
        finally:
            window.train_process = None

    def test_stop_training_dispatches_terminate_once_and_finishes_as_cancelled(self):
        window = self._create_window()
        process = MagicMock()
        task_id = "training-stop-test"
        window.train_process = process
        window._train_task_id = task_id
        window._train_finished = False
        window.task_manager.create_task(
            "training",
            task_id=task_id,
            cancel_callback=lambda: window._terminate_train_process(process),
        )

        with patch.object(window, "_terminate_train_process") as terminate:
            window._stop_train_process()

        terminate.assert_called_once_with(process)
        self.assertIsNone(window.train_process)
        task = window.task_manager.get(task_id)
        self.assertEqual(task.final_event.type, launcher.TaskEventType.CANCELLED)

    def test_launcher_navigation_buttons_keep_existing_callbacks(self):
        window = self._create_window()

        expected_callbacks = {
            "数据集准备": window._on_prepare,
            "启动标注工具": window._on_launch_annotator,
            "数据质量检查": window._on_quality_check,
            "模型训练": window._on_train,
            "模型测试": window._on_launch_detect,
            "模型评估": window._on_model_evaluation,
            "模型管理": window._on_model_manager,
            "TensorBoard": window._on_tensorboard,
            "导出诊断": window._on_export_diagnostics,
            "关于": window._on_about,
        }
        for name, callback in expected_callbacks.items():
            receivers = window.func_buttons[name].receivers(
                window.func_buttons[name].clicked
            )
            self.assertGreaterEqual(receivers, 1, f"{name} 未连接回调 {callback}")

    def test_export_diagnostics_uses_current_workspace_without_user_data(self):
        window = self._create_window()
        output = Path(self.temp_dir.name) / "诊断.zip"
        with patch.object(
            QFileDialog,
            "getSaveFileName",
            return_value=(str(output), "ZIP (*.zip)"),
        ), patch.object(
            launcher,
            "build_diagnostic_archive",
            return_value=output,
        ) as build_archive, patch.object(QMessageBox, "information"):
            window._on_export_diagnostics()

        build_archive.assert_called_once()
        call = build_archive.call_args
        self.assertEqual(call.args[0], output)
        self.assertFalse(call.kwargs["include_user_data"])
        self.assertEqual(
            call.kwargs["runtime_paths"].workspace_dir,
            window.workspace_dir,
        )

    def test_about_opens_product_dialog(self):
        window = self._create_window()
        with patch.object(launcher, "AboutDialog") as dialog_class:
            window._on_about()

        dialog_class.assert_called_once_with(window, window.runtime_paths)
        dialog_class.return_value.exec_.assert_called_once_with()

    def test_model_evaluation_opens_shared_dialog_with_current_paths(self):
        window = self._create_window()
        data_yaml = Path(self.config["dataset_dir"]) / "data.yaml"
        data_yaml.parent.mkdir(parents=True)
        data_yaml.write_text("path: .\nnames: []\n", encoding="utf-8")
        with patch.object(
            launcher.tools,
            "find_data_configs",
            return_value=[data_yaml],
        ), patch.object(window, "_show_child_window") as show_window:
            window._on_model_evaluation()

        show_window.assert_called_once()
        dialog = show_window.call_args.args[0]
        self.assertEqual(dialog.data_edit.text(), str(data_yaml))
        self.assertEqual(dialog._runs_dir, Path(self.config["runs_dir"]))

    def test_model_manager_opens_with_workspace_models_directory(self):
        window = self._create_window()
        with patch.object(window, "_show_child_window") as show_window:
            window._on_model_manager()

        show_window.assert_called_once()
        dialog = show_window.call_args.args[0]
        self.assertEqual(dialog.models_dir, window.models_dir)

    def test_training_stops_before_model_scan_when_runtime_is_unavailable(self):
        window = self._create_window()
        with patch.object(
            launcher,
            "ensure_ml_runtime",
            return_value=False,
        ) as ensure, patch.object(
            launcher.tools,
            "find_pretrained_models",
        ) as find_models:
            window._on_train()

        ensure.assert_called_once_with(window, "模型训练")
        find_models.assert_not_called()

    def test_training_scans_current_workspace_and_configured_models_directory(self):
        base = Path(self.temp_dir.name)
        workspace = base / "自定义工作区"
        models_dir = base / "外部模型仓库"
        models_dir.mkdir()
        model = models_dir / "yolov8n.pt"
        model.write_bytes(b"model")
        self.config.update(
            {
                "workspace_dir": str(workspace),
                "models_dir": str(models_dir),
            }
        )
        window = self._create_window()

        with patch.object(
            launcher.tools,
            "find_pretrained_models",
            return_value=[model],
        ) as find_models, patch.object(launcher, "TrainDialog") as dialog_class:
            dialog_class.return_value.exec_.return_value = 0
            window._on_train()

        find_models.assert_called_once_with(
            project_dir=window.workspace_dir,
            models_dir=window.models_dir,
        )
        dialog_class.assert_called_once_with(
            window,
            [model],
            dataset_dir=window.dataset_dir,
            runs_dir=str(window.runs_dir),
        )

    def test_first_run_selects_workspace_and_saves_productized_paths(self):
        selected = Path(self.temp_dir.name) / "新工作区"
        selected.mkdir()
        config = {
            "workspace_dir": str(Path(self.temp_dir.name) / "不存在"),
            "dataset_dir": "",
            "runs_dir": "",
            "models_dir": "",
            "source_dir": "",
        }
        with patch.object(
            QFileDialog,
            "getExistingDirectory",
            return_value=str(selected),
        ), patch.object(launcher.tools, "save_config") as save_config:
            prepared = launcher.prepare_initial_workspace(config)

        self.assertEqual(prepared["workspace_dir"], str(selected.resolve()))
        self.assertEqual(prepared["dataset_dir"], str((selected / "dataset").resolve()))
        self.assertEqual(prepared["models_dir"], str((selected / "models").resolve()))
        save_config.assert_called_once_with(prepared)

    def test_first_run_cancel_does_not_save_or_create_workspace(self):
        missing = Path(self.temp_dir.name) / "未选择工作区"
        config = {"workspace_dir": str(missing)}
        with patch.object(
            QFileDialog,
            "getExistingDirectory",
            return_value="",
        ), patch.object(launcher.tools, "save_config") as save_config:
            prepared = launcher.prepare_initial_workspace(config)

        self.assertIsNone(prepared)
        self.assertFalse(missing.exists())
        save_config.assert_not_called()

    def test_quality_location_is_forwarded_to_annotator(self):
        window = self._create_window()
        image_path = "D:/dataset/images/val/target.png"

        with patch.object(window, "_open_annotator_window") as open_annotator:
            window._on_quality_locate(image_path)

        open_annotator.assert_called_once_with(initial_image=image_path)

    def test_quality_optimize_opens_repair_split_dialog(self):
        window = self._create_window()

        with patch.object(window, "_open_split_dialog") as open_dialog:
            window._on_quality_optimize("repair")

        open_dialog.assert_called_once_with("repair")

    def test_split_navigation_opens_shared_dialog(self):
        window = self._create_window()

        with patch.object(window, "_open_split_dialog") as open_dialog:
            window._on_split()

        open_dialog.assert_called_once_with("repair")

    def test_backup_dialog_distinguishes_full_and_legacy_restore_scope(self):
        dataset_dir = Path(self.temp_dir.name) / "dataset"
        smart = dataset_dir / "backups" / "20260718-120000-smart_split"
        legacy = dataset_dir / "backups" / "20260718-110000-manual"
        smart.mkdir(parents=True)
        (smart / "manifest.json").write_text(
            json.dumps({"kind": "smart_split"}),
            encoding="utf-8",
        )
        (legacy / "labels").mkdir(parents=True)
        (legacy / "labels" / "a.txt").write_text("", encoding="utf-8")

        dialog = launcher.BackupRestoreDialog(None, dataset_dir)
        self.addCleanup(dialog.deleteLater)

        scopes = {
            dialog.table.item(row, 3).text()
            for row in range(dialog.table.rowCount())
        }
        self.assertEqual(dialog.table.columnCount(), 6)
        self.assertIn("完整分组恢复", scopes)
        self.assertIn("仅标签和配置", scopes)

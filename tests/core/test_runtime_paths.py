from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.core.runtime_paths import MLRuntimePaths, RuntimePaths
import tools.yolo_dataset_tools as dataset_tools


class RuntimePathTests(TestCase):
    def test_ml_runtime_paths_use_neutral_local_app_data_directory(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            paths = MLRuntimePaths.from_environment(
                {"LOCALAPPDATA": str(base / "本地状态")}
            )

            self.assertEqual(paths.root_dir, base / "本地状态" / "YOLOToolbox")
            self.assertEqual(paths.runtimes_dir, paths.root_dir / "runtimes")
            self.assertEqual(paths.cache_dir, paths.root_dir / "cache" / "runtimes")
            self.assertEqual(paths.staging_dir, paths.root_dir / "staging" / "runtimes")
            self.assertEqual(
                paths.state_file,
                paths.root_dir / "config" / "runtime_state.json",
            )
            self.assertFalse(paths.root_dir.exists())

    def test_source_resources_and_user_state_are_separate(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            install_dir = base / "源码目录"
            local_app_data = base / "用户状态"
            workspace = base / "工作区"
            paths = RuntimePaths.from_environment(
                workspace_dir=workspace,
                install_dir=install_dir,
                frozen=False,
                environ={
                    "LOCALAPPDATA": str(local_app_data),
                    "USERPROFILE": str(base / "用户"),
                },
            )

            self.assertEqual(paths.install_dir, install_dir)
            self.assertEqual(paths.resource_dir, install_dir)
            self.assertEqual(paths.state_dir, local_app_data / "WorkBuddyYoloTool")
            self.assertEqual(paths.workspace_dir, workspace)
            self.assertEqual(paths.config_file, paths.state_dir / "config" / "tool_config.json")
            self.assertEqual(paths.logs_dir, paths.state_dir / "logs")
            self.assertEqual(paths.tensorboard_logdirs, paths.state_dir / "tensorboard_logdirs")
            self.assertEqual(paths.dataset_dir, workspace / "dataset")
            self.assertEqual(paths.models_dir, workspace / "models")
            self.assertEqual(paths.runs_dir, workspace / "runs")

    def test_frozen_resources_use_internal_directory(self):
        install_dir = Path(r"C:\Program Files\WorkBuddy\YOLO")
        paths = RuntimePaths.from_environment(
            install_dir=install_dir,
            frozen=True,
            environ={
                "LOCALAPPDATA": r"C:\Users\tester\AppData\Local",
                "USERPROFILE": r"C:\Users\tester",
            },
        )

        self.assertEqual(paths.resource_dir, install_dir / "_internal")
        self.assertNotEqual(paths.workspace_dir, paths.install_dir)
        self.assertFalse(str(paths.config_file).startswith(str(install_dir)))

    def test_environment_overrides_state_and_workspace(self):
        paths = RuntimePaths.from_environment(
            install_dir=Path(r"C:\app"),
            frozen=False,
            environ={
                "WORKBUDDY_STATE_DIR": r"D:\状态",
                "WORKBUDDY_WORKSPACE_DIR": r"E:\训练工作区",
            },
        )

        self.assertEqual(paths.state_dir, Path(r"D:\状态"))
        self.assertEqual(paths.workspace_dir, Path(r"E:\训练工作区"))

    def test_path_resolution_does_not_create_directories(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            state_dir = base / "尚未创建的状态目录"
            workspace = base / "尚未创建的工作区"

            RuntimePaths.from_environment(
                install_dir=base / "程序",
                workspace_dir=workspace,
                frozen=False,
                environ={"WORKBUDDY_STATE_DIR": str(state_dir)},
            )

            self.assertFalse(state_dir.exists())
            self.assertFalse(workspace.exists())

    def test_legacy_config_is_loaded_readonly_from_original_workspace(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            install_dir = base / "旧工具箱"
            state_dir = base / "用户状态"
            legacy_config = install_dir / "config" / "tool_config.json"
            legacy_config.parent.mkdir(parents=True)
            legacy_text = json.dumps(
                {
                    "dataset_dir": "dataset",
                    "runs_dir": "runs",
                    "source_dir": "D:/截图",
                },
                ensure_ascii=False,
            )
            legacy_config.write_text(legacy_text, encoding="utf-8")
            paths = RuntimePaths.from_environment(
                install_dir=install_dir,
                frozen=False,
                environ={"WORKBUDDY_STATE_DIR": str(state_dir)},
            )

            with patch.object(dataset_tools, "get_runtime_paths", return_value=paths):
                loaded = dataset_tools.load_config()

            self.assertEqual(loaded["workspace_dir"], str(install_dir.resolve()))
            self.assertEqual(loaded["dataset_dir"], str((install_dir / "dataset").resolve()))
            self.assertEqual(loaded["runs_dir"], str((install_dir / "runs").resolve()))
            self.assertEqual(legacy_config.read_text(encoding="utf-8"), legacy_text)
            self.assertFalse(paths.config_file.exists())

    def test_new_config_is_saved_relative_to_workspace_in_state_directory(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            workspace = base / "工作区"
            paths = RuntimePaths.from_environment(
                install_dir=base / "程序",
                workspace_dir=workspace,
                frozen=False,
                environ={"WORKBUDDY_STATE_DIR": str(base / "用户状态")},
            )
            config = {
                "workspace_dir": str(workspace),
                "dataset_dir": str(workspace / "dataset"),
                "runs_dir": str(workspace / "runs"),
                "models_dir": str(workspace / "models"),
                "source_dir": "D:/截图",
            }

            with patch.object(dataset_tools, "get_runtime_paths", return_value=paths):
                dataset_tools.save_config(config)
                saved = json.loads(paths.config_file.read_text(encoding="utf-8"))
                loaded = dataset_tools.load_config()

            self.assertEqual(saved["workspace_dir"], str(workspace.resolve()))
            self.assertEqual(saved["dataset_dir"], "dataset")
            self.assertEqual(saved["runs_dir"], "runs")
            self.assertEqual(saved["models_dir"], "models")
            self.assertEqual(loaded["models_dir"], str((workspace / "models").resolve()))

    def test_training_model_scan_uses_workspace_models_directory(self):
        with TemporaryDirectory() as temp_name:
            workspace = Path(temp_name) / "工作区"
            models_dir = workspace / "models"
            models_dir.mkdir(parents=True)
            model = models_dir / "yolo11n.pt"
            model.write_bytes(b"model")
            paths = RuntimePaths.from_environment(
                install_dir=Path(temp_name) / "程序",
                workspace_dir=workspace,
                frozen=False,
                environ={"WORKBUDDY_STATE_DIR": str(Path(temp_name) / "状态")},
            )

            with patch.object(dataset_tools, "get_runtime_paths", return_value=paths):
                models = dataset_tools.find_pretrained_models()

            self.assertIn(model, models)

    def test_training_model_scan_accepts_explicit_models_directory(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            workspace = base / "工作区"
            models_dir = base / "外部模型仓库"
            workspace.mkdir()
            models_dir.mkdir()
            model = models_dir / "yolov8n.pt"
            model.write_bytes(b"model")

            models = dataset_tools.find_pretrained_models(
                project_dir=workspace,
                models_dir=models_dir,
            )

            self.assertEqual(models, [model])

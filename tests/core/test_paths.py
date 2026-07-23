from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import TestCase

from tools.core.paths import ProjectPaths
from tools.core.runtime_paths import RuntimePaths


class ProjectPathsTests(TestCase):
    def test_resolves_standard_directories(self):
        project = Path("C:/工具/项目")
        paths = ProjectPaths.from_project_dir(project)

        self.assertEqual(paths.project_dir, project)
        self.assertEqual(paths.dataset_dir, project / "dataset")
        self.assertEqual(paths.runs_dir, project / "runs")
        self.assertEqual(paths.models_dir, project / "models")
        self.assertEqual(paths.logs_dir, project / "logs")
        self.assertEqual(paths.config_file, project / "config" / "tool_config.json")
        self.assertEqual(paths.image_dir("train"), project / "dataset" / "images" / "train")
        self.assertEqual(paths.label_dir("val"), project / "dataset" / "labels" / "val")

    def test_accepts_external_dataset_and_runs(self):
        paths = ProjectPaths.from_project_dir(
            Path("C:/project"),
            dataset_dir=Path("D:/数据集"),
            runs_dir=Path("E:/训练结果"),
        )

        self.assertEqual(paths.dataset_dir, Path("D:/数据集"))
        self.assertEqual(paths.runs_dir, Path("E:/训练结果"))

    def test_is_immutable(self):
        paths = ProjectPaths.from_project_dir(Path("C:/project"))

        with self.assertRaises(FrozenInstanceError):
            paths.project_dir = Path("D:/other")

    def test_builds_from_productized_runtime_paths(self):
        runtime = RuntimePaths.from_environment(
            install_dir=Path("C:/Program Files/WorkBuddy/YOLO"),
            workspace_dir=Path("D:/YOLO工作区"),
            frozen=True,
            environ={"WORKBUDDY_STATE_DIR": "C:/Users/tester/AppData/Local/WorkBuddyYoloTool"},
        )

        paths = ProjectPaths.from_runtime_paths(runtime)

        self.assertEqual(paths.project_dir, runtime.resource_dir)
        self.assertEqual(paths.dataset_dir, runtime.dataset_dir)
        self.assertEqual(paths.models_dir, runtime.models_dir)
        self.assertEqual(paths.logs_dir, runtime.logs_dir)
        self.assertEqual(paths.config_file, runtime.config_file)

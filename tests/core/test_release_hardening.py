import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools import build_release
from tools import preflight_check
from tools.build_runtime import package_runtime_directory, write_runtime_catalog
import tools.yolo_dataset_tools as dataset_tools
from tools.core.ml_runtime import RuntimeArtifact
from tools.core.python_support import describe_python, selection_text


class ReleaseHardeningTests(TestCase):
    @staticmethod
    def _write(path: Path, content: bytes = b"MZ") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _package_runtime(self, root: Path, output: Path, profile: str):
        gpu = profile == "cuda118"
        artifact = RuntimeArtifact.from_dict({
            "runtime_id": "ml-cu118-win-x64-r1" if gpu else "ml-cpu-win-x64-r1",
            "profile": profile,
            "platform": "windows",
            "architecture": "x86_64",
            "worker_protocol": 1,
            "torch_version": "2.7.1+cu118" if gpu else "2.7.1+cpu",
            "torchvision_version": "0.22.1+cu118" if gpu else "0.22.1+cpu",
            "ultralytics_version": "8.4.90",
            "cuda_version": "11.8" if gpu else None,
        })
        self._write(root / "YOLO工具箱Worker.exe")
        self._write(root / "_internal" / "torch" / "lib" / "torch_cpu.dll")
        if gpu:
            self._write(root / "_internal" / "torch" / "lib" / "torch_cuda.dll")
            self._write(root / "_internal" / "torch" / "lib" / "cudnn64_9.dll")
            self._write(root / "_internal" / "torch" / "lib" / "cublas64_11.dll")
        (root / "runtime.json").write_text(
            json.dumps(artifact.to_dict()),
            encoding="utf-8",
        )
        (root / "LICENSE").write_text("license", encoding="utf-8")
        (root / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
        _archive, completed = package_runtime_directory(
            root,
            artifact,
            output,
            base_url="https://example.invalid/runtime-r1",
        )
        return completed

    def test_generated_batch_files_use_their_own_directory(self):
        generated = build_release.generated_files()

        install_bat = generated["安装依赖.bat"].decode("utf-8-sig")
        launch_bat = generated["启动YOLO工具箱.bat"].decode("utf-8-sig")
        for content in (install_bat, launch_bat):
            self.assertIn('cd /d "%~dp0"', content)
            self.assertIn("exit /b %RC%", content)
            self.assertNotIn("C:\\Users\\", content)
            self.assertIn('set "VENV_DIR=%~dp0.venv"', install_bat)
            self.assertIn('set "PYTHON_EXE=%VENV_DIR%\\Scripts\\python.exe"', install_bat)
            self.assertIn('set "PYTHON_EXE=%~dp0.venv\\Scripts\\python.exe"', launch_bat)
            self.assertIn('"%PYTHON_EXE%" -u tools\\', content)
            self.assertNotIn("python -u tools\\", content)

        self.assertIn("py -%%V", install_bat)
        for version in ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(version, install_bat)
        self.assertIn("包内 .venv", generated["README_RELEASE.txt"].decode("utf-8"))

    def test_python_support_levels_and_selection_order(self):
        self.assertEqual(describe_python((3, 8)).status, "unsupported")
        self.assertEqual(describe_python((3, 9)).status, "compatible")
        self.assertEqual(describe_python((3, 11)).status, "recommended")
        self.assertEqual(describe_python((3, 13)).status, "recommended")
        self.assertEqual(describe_python((3, 14)).status, "experimental")
        self.assertEqual(describe_python((3, 15)).status, "unsupported")
        self.assertEqual(selection_text(), "3.11 3.12 3.13 3.10 3.9 3.14")

    def test_preflight_skips_core_tests_when_release_package_has_no_tests(self):
        with tempfile.TemporaryDirectory() as temp_name:
            project = Path(temp_name)
            dataset = project / "dataset"
            runs = project / "runs"
            dataset.mkdir()
            runs.mkdir()
            with patch.object(preflight_check, "PROJECT_DIR", project), \
                 patch.object(preflight_check, "REQUIRED_SOURCE_FILES", []), \
                 patch.object(build_release, "PROJECT_DIR", project), \
                 patch.object(build_release, "collect_sources", return_value=[]), \
                 patch.object(build_release, "generated_files", return_value={}):
                report = preflight_check.run_preflight(
                    dataset_dir=dataset,
                    runs_dir=runs,
                    include_pip=False,
                    include_env=False,
                    include_core_tests=True,
                )

        self.assertEqual(report["errors"], 0)
        self.assertGreaterEqual(report["warnings"], 1)
        self.assertTrue(
            any("测试未随发布包附带" in message for _level, message in report["lines"])
        )

    def test_release_artifact_preflight_compares_base_and_runtime_catalogs(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime_output = root / "runtime-output"
            cpu = self._package_runtime(root / "cpu", runtime_output, "cpu")
            gpu = self._package_runtime(root / "gpu", runtime_output, "cuda118")
            catalog_path = write_runtime_catalog(
                (cpu, gpu),
                runtime_output / "runtime_catalog.json",
            )

            base = root / "base"
            self._write(base / "YOLO工具箱.exe")
            self._write(
                base / "_internal" / "PyQt5" / "Qt5" / "plugins" / "platforms" / "qwindows.dll"
            )
            for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"):
                (base / name).write_text(name, encoding="utf-8")
            (base / "BUILD_INFO.json").write_text("{}", encoding="utf-8")
            (base / "_internal" / "runtime_catalog.json").write_bytes(
                catalog_path.read_bytes()
            )
            messages = []

            preflight_check._check_release_artifacts(
                exe_dir=base,
                runtime_dir=runtime_output,
                add=lambda level, message: messages.append((level, message)),
            )

        self.assertFalse([item for item in messages if item[0] == "error"])
        self.assertTrue(any("清单一致" in message for _level, message in messages))

    def test_release_sources_exclude_runtime_data_and_models_by_default(self):
        sources = build_release.collect_sources(include_models=False)
        source_text = {path.as_posix() for path in sources}

        self.assertFalse(any(path.startswith("dataset/") for path in source_text))
        self.assertFalse(any(path.startswith("runs/") for path in source_text))
        self.assertFalse(any(path.startswith("logs/") for path in source_text))
        self.assertFalse(any(path.startswith("debug/") for path in source_text))
        self.assertFalse(any(path.lower().endswith(".pt") for path in source_text))
        self.assertIn("tools/core/task_protocol.py", source_text)
        self.assertIn("tools/core/model_registry.py", source_text)
        self.assertIn("tools/yolo_model_manager.py", source_text)
        self.assertIn("tools/yolo_ui_widgets.py", source_text)
        self.assertIn("tests/core/test_release_hardening.py", source_text)
        self.assertFalse(any(path.startswith("docs/superpowers/") for path in source_text))
        self.assertIn(
            "tools/yolo_ui_widgets.py",
            preflight_check.REQUIRED_SOURCE_FILES,
        )
        self.assertIn(
            "tools/core/python_support.py",
            preflight_check.REQUIRED_SOURCE_FILES,
        )
        self.assertIn(
            "tools/core/model_registry.py",
            preflight_check.REQUIRED_SOURCE_FILES,
        )
        self.assertIn(
            "tools/yolo_model_manager.py",
            preflight_check.REQUIRED_SOURCE_FILES,
        )
        for required in (
            "requirements-base-windows.txt",
            "requirements-runtime-windows-cpu.txt",
            "packaging/runtime_profiles.json",
            "packaging/workbuddy_yolo_runtime.spec",
            "packaging/smoke_test.ps1",
            "tools/build_runtime.py",
            "tools/core/ml_runtime.py",
            "tools/core/runtime_installer.py",
            "tools/core/external_python.py",
            "tools/yolo_runtime_dialog.py",
        ):
            self.assertIn(required, source_text)
            self.assertIn(required, preflight_check.REQUIRED_SOURCE_FILES)
        self.assertEqual(
            set(preflight_check.PIP_REQUIREMENT_FILES),
            {
                "requirements.txt",
                "requirements-base-windows.txt",
                "requirements-runtime-windows-cpu.txt",
                "requirements-release-windows-cu118.txt",
            },
        )

    def test_release_requires_smart_split_modules(self):
        source_text = {
            path.as_posix()
            for path in build_release.collect_sources(include_models=False)
        }
        required = {
            "tools/core/dataset_split.py",
            "tools/core/dataset_split_executor.py",
            "tools/yolo_split_dialog.py",
        }

        self.assertTrue(required <= source_text)
        self.assertTrue(required <= set(preflight_check.REQUIRED_SOURCE_FILES))

    def test_project_paths_are_saved_relative_and_external_paths_stay_absolute(self):
        with tempfile.TemporaryDirectory() as temp_name:
            project = Path(temp_name) / "工具箱"
            config_path = project / "config" / "tool_config.json"
            project_dataset = project / "dataset"
            project_runs = project / "runs"
            external_dataset = Path(temp_name) / "外部数据"
            project_dataset.mkdir(parents=True)
            project_runs.mkdir()

            with patch.object(dataset_tools, "get_project_dir", return_value=project), \
                 patch.object(dataset_tools, "get_config_path", return_value=config_path):
                dataset_tools.save_config({
                    "dataset_dir": str(project_dataset),
                    "runs_dir": str(project_runs),
                    "source_dir": str(external_dataset),
                    "annotator_last_images": {
                        str(project_dataset / "images" / "train"): str(
                            project_dataset / "images" / "train" / "a.png"
                        )
                    },
                })
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                loaded = dataset_tools.load_config()

            self.assertEqual(saved["dataset_dir"], "dataset")
            self.assertEqual(saved["runs_dir"], "runs")
            self.assertEqual(saved["source_dir"], str(external_dataset))
            self.assertEqual(loaded["dataset_dir"], str(project_dataset.resolve()))
            self.assertEqual(loaded["runs_dir"], str(project_runs.resolve()))
            image_dir = project_dataset / "images" / "train"
            self.assertIn(str(image_dir.resolve()), loaded["annotator_last_images"])

    def test_old_missing_project_path_is_migrated_after_directory_move(self):
        with tempfile.TemporaryDirectory() as temp_name:
            project = Path(temp_name) / "新位置" / "工具箱"
            config_path = project / "config" / "tool_config.json"
            (project / "dataset").mkdir(parents=True)
            old_dataset = Path(temp_name) / "旧位置" / "工具箱" / "dataset"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps({
                    "dataset_dir": str(old_dataset),
                    "runs_dir": str(Path(temp_name) / "旧位置" / "工具箱" / "runs"),
                }),
                encoding="utf-8",
            )

            with patch.object(dataset_tools, "get_project_dir", return_value=project), \
                 patch.object(dataset_tools, "get_config_path", return_value=config_path):
                loaded = dataset_tools.load_config()

            self.assertEqual(loaded["dataset_dir"], str((project / "dataset").resolve()))
            self.assertEqual(loaded["runs_dir"], str((project / "runs").resolve()))

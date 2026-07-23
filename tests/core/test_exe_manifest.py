from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.build_exe import (
    MAX_BASE_BYTES,
    MIN_FREE_BUILD_BYTES,
    check_build_disk_space,
    check_exe_manifest,
    select_entry_scripts,
)


class ExeManifestTests(TestCase):
    @staticmethod
    def _write(path: Path, content: bytes = b"MZ") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _make_valid_candidate(self, root: Path) -> None:
        self._write(root / "YOLO工具箱.exe")
        (root / "LICENSE").write_text("AGPL-3.0-only", encoding="utf-8")
        (root / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
        (root / "CHANGELOG.md").write_text("changes", encoding="utf-8")
        (root / "BUILD_INFO.json").write_text(
            json.dumps({"version": "0.9.0-beta.1"}),
            encoding="utf-8",
        )
        self._write(root / "_internal" / "PyQt5" / "Qt5" / "plugins" / "platforms" / "qwindows.dll")
        (root / "_internal" / "runtime_catalog.json").write_text(
            json.dumps(
                {
                    "catalog_version": 1,
                    "artifacts": [
                        {
                            "runtime_id": "ml-cpu-win-x64-r1",
                            "profile": "cpu",
                            "platform": "windows",
                            "architecture": "x86_64",
                            "worker_protocol": 1,
                            "torch_version": "2.7.1+cpu",
                            "torchvision_version": "0.22.1+cpu",
                            "ultralytics_version": "8.4.90",
                            "cuda_version": None,
                            "archive_size": 123,
                            "installed_size": 456,
                            "sha256": "a" * 64,
                            "url": "https://example.invalid/ml-cpu-win-x64-r1.zip",
                        },
                        {
                            "runtime_id": "ml-cu118-win-x64-r1",
                            "profile": "cuda118",
                            "platform": "windows",
                            "architecture": "x86_64",
                            "worker_protocol": 1,
                            "torch_version": "2.7.1+cu118",
                            "torchvision_version": "0.22.1+cu118",
                            "ultralytics_version": "8.4.90",
                            "cuda_version": "11.8",
                            "archive_size": 789,
                            "installed_size": 999,
                            "sha256": "b" * 64,
                            "url": "https://example.invalid/ml-cu118-win-x64-r1.zip",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_missing_release_components_are_reported(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            errors = check_exe_manifest(root)

        self.assertIn("缺少主程序: YOLO工具箱.exe", errors)
        self.assertIn("缺少共享依赖目录: _internal", errors)
        self.assertIn("缺少许可证: LICENSE", errors)
        self.assertIn("缺少构建元数据: BUILD_INFO.json", errors)
        self.assertIn("缺少 Qt Windows 平台插件: qwindows.dll", errors)
        self.assertIn("缺少可信运行时清单: runtime_catalog.json", errors)

    def test_valid_candidate_has_no_manifest_errors(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_valid_candidate(root)

            self.assertEqual(check_exe_manifest(root), [])

    def test_runtime_data_and_model_files_are_rejected(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_valid_candidate(root)
            self._write(root / "weights.pt", b"model")
            self._write(root / "dataset" / "images" / "new.jpg", b"image")
            self._write(root / "logs" / "tool.log", b"log")

            errors = check_exe_manifest(root)

        self.assertTrue(any("禁止发布物包含模型文件" in item for item in errors))
        self.assertTrue(any("禁止发布物包含用户运行目录" in item for item in errors))

    def test_machine_learning_runtime_is_rejected_from_base_candidate(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_valid_candidate(root)
            self._write(root / "YOLO工具箱Worker.exe")
            self._write(root / "_internal" / "torch" / "lib" / "torch_cpu.dll")
            self._write(root / "_internal" / "_polars_runtime.pyd")
            self._write(root / "_internal" / "cudnn64_9.dll")

            errors = check_exe_manifest(root)

        self.assertTrue(any("禁止基础程序包含 Worker" in item for item in errors))
        self.assertTrue(any("禁止基础程序包含机器学习运行时" in item for item in errors))

    def test_entry_script_selection_keeps_runtime_hooks(self):
        scripts = [
            ("pyi_rth_inspect", "rthooks/pyi_rth_inspect.py", "PYSOURCE"),
            ("pyi_rth_pyqt5", "rthooks/pyi_rth_pyqt5.py", "PYSOURCE"),
            ("yolo_tool_launcher", "tools/yolo_tool_launcher.py", "PYSOURCE"),
            ("yolo_worker_entry", "tools/yolo_worker_entry.py", "PYSOURCE"),
        ]

        gui_scripts = select_entry_scripts(scripts, "yolo_tool_launcher")
        self.assertEqual([item[0] for item in gui_scripts], [
            "pyi_rth_inspect",
            "pyi_rth_pyqt5",
            "yolo_tool_launcher",
        ])

    def test_spec_selects_entry_scripts_by_name(self):
        root = Path(__file__).resolve().parents[2]
        spec = (root / "packaging" / "workbuddy_yolo.spec").read_text(encoding="utf-8")

        self.assertIn('select_entry_scripts(analysis.scripts, "yolo_tool_launcher")', spec)
        self.assertNotIn('select_entry_scripts(analysis.scripts, "yolo_worker_entry")', spec)
        self.assertNotIn("analysis.scripts[0", spec)

    def test_build_disk_space_check_has_stable_boundary(self):
        root = Path(r"C:\build")

        self.assertIsNone(
            check_build_disk_space(root, free_bytes=MIN_FREE_BUILD_BYTES)
        )
        error = check_build_disk_space(
            root,
            free_bytes=MIN_FREE_BUILD_BYTES - 1,
        )

        self.assertIsNotNone(error)
        self.assertIn("构建磁盘空间不足", str(error))
        self.assertIn("12.0 GiB", str(error))

    def test_spec_embeds_version_resource_for_gui_only(self):
        root = Path(__file__).resolve().parents[2]
        spec = (root / "packaging" / "workbuddy_yolo.spec").read_text(encoding="utf-8")

        self.assertIn("def windows_version_info", spec)
        self.assertIn('version=windows_version_info("YOLO工具箱.exe"', spec)
        self.assertNotIn('version=windows_version_info("YOLO工具箱Worker.exe"', spec)

    def test_base_spec_explicitly_excludes_machine_learning_packages(self):
        root = Path(__file__).resolve().parents[2]
        spec_path = root / "packaging" / "workbuddy_yolo.spec"
        text = spec_path.read_text(encoding="utf-8")

        for package in ("torch", "torchvision", "ultralytics", "tensorboard", "polars"):
            self.assertIn(f'"{package}"', text)
        self.assertNotIn("collect_all", text)

    def test_base_size_limit_is_enforced(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_valid_candidate(root)
            with (root / "oversized.bin").open("wb") as stream:
                stream.truncate(MAX_BASE_BYTES + 1)

            errors = check_exe_manifest(root)

        self.assertTrue(any("基础程序体积超过" in item for item in errors))

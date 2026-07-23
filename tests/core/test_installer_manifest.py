from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import json

from tools.build_installer import (
    INSTALLER_FILE_NAME,
    build_iscc_command,
    check_installer_manifest,
    find_iscc,
    resolve_exe_dir,
)


class InstallerManifestTests(TestCase):
    @staticmethod
    def _write(path: Path, content: bytes = b"MZ") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _make_valid_candidate(self, root: Path) -> None:
        self._write(root / "YOLO工具箱.exe")
        (root / "LICENSE").write_text("AGPL-3.0-only", encoding="utf-8")
        (root / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
        (root / "CHANGELOG.md").write_text("changes", encoding="utf-8")
        (root / "BUILD_INFO.json").write_text("{}", encoding="utf-8")
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

    def test_valid_candidate_is_accepted_and_nested_candidate_is_resolved(self):
        with TemporaryDirectory() as temp_name:
            parent = Path(temp_name)
            candidate = parent / "YOLO数据标注工具箱"
            self._make_valid_candidate(candidate)

            self.assertEqual(check_installer_manifest(candidate), [])
            self.assertEqual(resolve_exe_dir(parent), candidate.resolve())

    def test_models_and_user_data_are_rejected(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_valid_candidate(root)
            self._write(root / "weights.pt", b"model")
            self._write(root / "dataset" / "images" / "new.jpg", b"image")
            self._write(root / "runs" / "train" / "results.csv", b"run")

            errors = check_installer_manifest(root)

        self.assertTrue(any("模型文件" in error for error in errors))
        self.assertTrue(any("用户运行目录" in error for error in errors))

    def test_installer_script_does_not_delete_user_state(self):
        script = (Path(__file__).resolve().parents[2] / "packaging" / "installer.iss").read_text(
            encoding="utf-8"
        )

        self.assertIn("DefaultDirName={autopf}\\WorkBuddy\\YOLO数据标注工具箱", script)
        self.assertIn("Uninstallable=yes", script)
        self.assertIn("PrivilegesRequiredOverridesAllowed=commandline", script)
        self.assertNotIn("[UninstallDelete]", script)
        self.assertNotIn("WorkBuddyYoloTool", script)
        self.assertNotIn("{userappdata}", script)
        self.assertIn("{autoprograms}", script)
        self.assertIn("VersionInfoProductVersion={#FileVersion}", script)
        self.assertNotIn("VersionInfoProductVersion={#AppVersion}", script)
        self.assertIn(
            'MessagesFile: "{#SourcePath}\\languages\\ChineseSimplified.isl"',
            script,
        )
        root = Path(__file__).resolve().parents[2]
        self.assertTrue(
            (root / "packaging" / "languages" / "ChineseSimplified.isl").is_file()
        )
        notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("Inno Setup 简体中文翻译", notices)
        self.assertIn("6da09d23e14443d4cf8f07b1c5fd821bfe459788", notices)

    def test_iscc_command_contains_absolute_paths_and_version(self):
        command = build_iscc_command(
            Path(r"C:\build\YOLO数据标注工具箱"),
            Path(r"D:\output"),
            source_url="https://example.invalid/source",
            iscc=Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        )

        self.assertEqual(command[0], str(Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")))
        self.assertIn("/DSourceDir=C:\\build\\YOLO数据标注工具箱", command)
        self.assertIn("/DOutputDir=D:\\output", command)
        self.assertIn("/DSourceUrl=https://example.invalid/source", command)
        self.assertTrue(command[-1].endswith("packaging\\installer.iss"))

    def test_find_iscc_prefers_explicit_path_and_environment(self):
        with TemporaryDirectory() as temp_name:
            explicit = Path(temp_name) / "ISCC.exe"
            explicit.write_bytes(b"MZ")
            self.assertEqual(find_iscc(explicit), explicit.resolve())

            env_path = Path(temp_name) / "env-iscc.exe"
            env_path.write_bytes(b"MZ")
            with patch.dict(os.environ, {"WORKBUDDY_ISCC": str(env_path)}, clear=False):
                self.assertEqual(find_iscc(), env_path.resolve())

    def test_find_iscc_detects_current_user_install(self):
        with TemporaryDirectory() as temp_name:
            local_app_data = Path(temp_name)
            iscc = local_app_data / "Programs" / "Inno Setup 6" / "ISCC.exe"
            iscc.parent.mkdir(parents=True)
            iscc.write_bytes(b"MZ")
            with patch.dict(
                os.environ,
                {"LOCALAPPDATA": str(local_app_data), "WORKBUDDY_ISCC": ""},
                clear=False,
            ), patch("tools.build_installer.shutil.which", return_value=None):
                self.assertEqual(find_iscc(), iscc.resolve())

    def test_installer_output_name_is_versioned(self):
        self.assertIn("0.9.0-beta.1", INSTALLER_FILE_NAME)
        self.assertTrue(INSTALLER_FILE_NAME.endswith("windows-x64-setup.exe"))

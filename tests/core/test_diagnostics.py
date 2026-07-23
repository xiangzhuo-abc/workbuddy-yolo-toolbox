from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools.core.diagnostics import (
    build_diagnostic_archive,
    install_global_exception_hook,
    write_crash_report,
)
from tools.core.runtime_paths import RuntimePaths


class DiagnosticsTests(TestCase):
    @staticmethod
    def _runtime(root: Path) -> RuntimePaths:
        state = root / "用户" / "状态"
        workspace = root / "用户" / "工作区"
        install = root / "程序"
        install.mkdir(parents=True)
        return RuntimePaths.from_environment(
            workspace_dir=workspace,
            environ={
                "USERPROFILE": str(root / "用户"),
                "LOCALAPPDATA": str(root / "用户" / "AppData" / "Local"),
                "WORKBUDDY_STATE_DIR": str(state),
            },
            install_dir=install,
            frozen=True,
        )

    def test_archive_contains_metadata_and_redacted_log_tail_only(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime = self._runtime(root)
            runtime.logs_dir.mkdir(parents=True)
            runtime.logs_dir.joinpath("yolo_tool.log").write_text(
                "early\n" + "x" * 300000 + f"\nlatest {runtime.workspace_dir}\n",
                encoding="utf-8",
            )
            secret_files = {
                runtime.dataset_dir / "images" / "train" / "secret.jpg": b"image-secret",
                runtime.dataset_dir / "labels" / "train" / "secret.txt": b"label-secret",
                runtime.models_dir / "secret.pt": b"model-secret",
                runtime.runs_dir / "train" / "results.csv": b"run-secret",
            }
            for path, content in secret_files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            output = build_diagnostic_archive(
                root / "diagnostics.zip",
                runtime_paths=runtime,
            )

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                report = json.loads(archive.read("diagnostics.json"))
                log_text = archive.read("logs/yolo_tool.log.tail.txt").decode("utf-8")
                all_bytes = b"".join(archive.read(name) for name in names)

        self.assertEqual(report["product"]["version"], "0.9.0-beta.1")
        self.assertIn("latest", log_text)
        self.assertIn("%WORKSPACE%", log_text)
        self.assertNotIn(str(runtime.workspace_dir), log_text)
        self.assertFalse(any("dataset" in name for name in names))
        self.assertFalse(any(name.endswith((".pt", ".jpg", ".csv")) for name in names))
        for secret in (b"image-secret", b"label-secret", b"model-secret", b"run-secret"):
            self.assertNotIn(secret, all_bytes)

    def test_user_data_opt_in_is_rejected(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime = self._runtime(root)
            with self.assertRaises(ValueError):
                build_diagnostic_archive(
                    root / "diagnostics.zip",
                    runtime_paths=runtime,
                    include_user_data=True,
                )

    def test_missing_paths_still_produce_status_report(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime = self._runtime(root)
            output = build_diagnostic_archive(
                root / "diagnostics.zip",
                runtime_paths=runtime,
            )
            with zipfile.ZipFile(output) as archive:
                report = json.loads(archive.read("diagnostics.json"))

        self.assertFalse(report["paths"]["dataset_dir"]["exists"])
        self.assertFalse(report["paths"]["logs_dir"]["exists"])

    def test_crash_report_is_written_to_state_directory(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime = self._runtime(root)
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                report = write_crash_report(runtime, *sys.exc_info())

            text = report.read_text(encoding="utf-8")

        self.assertEqual(report.parent, runtime.crash_reports_dir)
        self.assertIn("RuntimeError: boom", text)
        self.assertIn("0.9.0-beta.1", text)

    def test_installed_exception_hook_writes_report_and_calls_previous_hook(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime = self._runtime(root)
            calls = []

            def previous(exc_type, exc_value, traceback):
                calls.append((exc_type, exc_value, traceback))

            with patch.object(sys, "excepthook", previous):
                returned = install_global_exception_hook(runtime)
                installed = sys.excepthook
                try:
                    raise ValueError("hook failure")
                except ValueError:
                    installed(*sys.exc_info())

            reports = list(runtime.crash_reports_dir.glob("crash-*.txt"))

        self.assertIs(returned, previous)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(reports), 1)

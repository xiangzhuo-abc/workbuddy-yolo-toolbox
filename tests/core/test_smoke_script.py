from pathlib import Path
from unittest import TestCase


class SmokeScriptTests(TestCase):
    def test_script_uses_isolated_workspace_and_reports_skips(self):
        root = Path(__file__).resolve().parents[2]
        script = (root / "packaging" / "smoke_test.ps1").read_text(encoding="utf-8")

        self.assertIn("WORKBUDDY_STATE_DIR", script)
        self.assertIn("WORKBUDDY_WORKSPACE_DIR", script)
        self.assertIn("YOLO_CONFIG_DIR", script)
        self.assertIn("PYTHONHOME", script)
        self.assertIn("PYTHONPATH", script)
        self.assertIn("smoke-", script)
        self.assertIn('"SKIP"', script)
        self.assertIn("dataset", script)
        self.assertIn("runs", script)
        self.assertNotIn("Remove-Item", script)

    def test_script_checks_gui_worker_tensorboard_and_orphans(self):
        root = Path(__file__).resolve().parents[2]
        script = (root / "packaging" / "smoke_test.ps1").read_text(encoding="utf-8")

        for marker in (
            "YOLO工具箱.exe",
            "YOLO工具箱Worker.exe",
            "tensorboard",
            "MISSING_MODEL_EXIT",
            "RemainingCandidateProcesses",
            "SHA256",
        ):
            self.assertIn(marker, script)

        self.assertIn("RuntimeDir", script)
        self.assertIn("Base portable layout", script)
        self.assertIn("Worker runtime layout", script)
        self.assertIn(
            '(-not (Test-Path -LiteralPath (Join-Path $Candidate "YOLO工具箱Worker.exe")))',
            script,
        )

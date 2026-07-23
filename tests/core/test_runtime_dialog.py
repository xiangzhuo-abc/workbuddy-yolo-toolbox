from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDialog

from tools.core.ml_runtime import RuntimeCatalog
from tools.core.external_python import ExternalPythonCandidate
from tools.core.ml_runtime import RuntimeSelection, RuntimeStateStore
from tools.core.runtime_paths import MLRuntimePaths
from tools import yolo_runtime_dialog as runtime_ui


def catalog() -> RuntimeCatalog:
    return RuntimeCatalog.from_dict(
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
                }
            ],
        }
    )


class RuntimeDialogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_source_mode_is_ready_without_opening_dialog(self):
        resumed = []
        with patch.object(runtime_ui.RuntimeInstallDialog, "exec_") as execute:
            result = runtime_ui.ensure_ml_runtime(
                None,
                "模型训练",
                resume_callback=lambda: resumed.append("done"),
                frozen=False,
            )

        self.assertTrue(result)
        self.assertEqual(resumed, ["done"])
        execute.assert_not_called()

    def test_existing_managed_runtime_is_selected_without_dialog(self):
        with TemporaryDirectory() as temp_name:
            paths = MLRuntimePaths.from_environment(
                {"YOLO_TOOLBOX_RUNTIME_DIR": str(Path(temp_name) / "runtime")}
            )
            artifact = catalog().artifacts[0]
            runtime_dir = paths.runtimes_dir / artifact.runtime_id
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "YOLO工具箱Worker.exe").write_bytes(b"worker")
            (runtime_dir / "runtime.json").write_text(
                json.dumps(artifact.to_dict()),
                encoding="utf-8",
            )

            with patch.object(runtime_ui.RuntimeInstallDialog, "exec_") as execute:
                result = runtime_ui.ensure_ml_runtime(
                    None,
                    "模型测试",
                    catalog_value=catalog(),
                    runtime_paths=paths,
                    frozen=True,
                )

            self.assertTrue(result)
            execute.assert_not_called()
            self.assertIn(artifact.runtime_id, paths.state_file.read_text(encoding="utf-8"))

    def test_rejected_install_dialog_does_not_resume_action(self):
        with TemporaryDirectory() as temp_name:
            paths = MLRuntimePaths.from_environment(
                {"YOLO_TOOLBOX_RUNTIME_DIR": str(Path(temp_name) / "runtime")}
            )
            resumed = []
            with patch.object(
                runtime_ui.RuntimeInstallDialog,
                "exec_",
                return_value=QDialog.Rejected,
            ):
                result = runtime_ui.ensure_ml_runtime(
                    None,
                    "模型训练",
                    resume_callback=lambda: resumed.append("done"),
                    catalog_value=catalog(),
                    runtime_paths=paths,
                    frozen=True,
                )

            self.assertFalse(result)
            self.assertEqual(resumed, [])

    def test_accepted_install_dialog_resumes_action_once(self):
        with TemporaryDirectory() as temp_name:
            paths = MLRuntimePaths.from_environment(
                {"YOLO_TOOLBOX_RUNTIME_DIR": str(Path(temp_name) / "runtime")}
            )
            resumed = []
            with patch.object(
                runtime_ui.RuntimeInstallDialog,
                "exec_",
                return_value=QDialog.Accepted,
            ):
                result = runtime_ui.ensure_ml_runtime(
                    None,
                    "自动预标注",
                    resume_callback=lambda: resumed.append("done"),
                    catalog_value=catalog(),
                    runtime_paths=paths,
                    frozen=True,
                )

            self.assertTrue(result)
            self.assertEqual(resumed, ["done"])

    def test_static_profile_disables_online_download(self):
        with TemporaryDirectory() as temp_name:
            paths = MLRuntimePaths.from_environment(
                {"YOLO_TOOLBOX_RUNTIME_DIR": str(Path(temp_name) / "runtime")}
            )
            dialog = runtime_ui.RuntimeInstallDialog(catalog(), paths)
            try:
                self.assertFalse(dialog.download_button.isEnabled())
                self.assertTrue(dialog.offline_button.isEnabled())
                self.assertIn("未配置", dialog.detail_label.text())
            finally:
                dialog.close()
                dialog.deleteLater()

    def test_selected_external_python_is_revalidated_without_install_dialog(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            paths = MLRuntimePaths.from_environment(
                {"YOLO_TOOLBOX_RUNTIME_DIR": str(root / "runtime")}
            )
            executable = root / "python.exe"
            worker_source = root / "worker" / "yolo_worker_entry.py"
            executable.write_bytes(b"python")
            worker_source.parent.mkdir()
            worker_source.write_text("", encoding="utf-8")
            RuntimeStateStore(paths.state_file).save(
                RuntimeSelection(
                    runtime_id=None,
                    backend_kind="external",
                    external_python=str(executable),
                )
            )
            candidate = ExternalPythonCandidate(
                executable=executable,
                python_version="3.13.5",
                architecture="AMD64",
                worker_protocol=1,
                versions={
                    "torch": "2.7.1+cu118",
                    "torchvision": "0.22.1+cu118",
                    "ultralytics": "8.4.90",
                    "tensorboard": "2.21.0",
                },
                cuda_version="11.8",
                gpu_available=True,
                gpu_names=("Test GPU",),
                ready=True,
                errors=(),
            )

            with patch.object(
                runtime_ui,
                "external_worker_source_path",
                return_value=worker_source,
            ), patch.object(
                runtime_ui,
                "probe_external_python",
                return_value=candidate,
            ) as probe, patch.object(
                runtime_ui.RuntimeInstallDialog,
                "exec_",
            ) as execute:
                result = runtime_ui.ensure_ml_runtime(
                    None,
                    "模型训练",
                    catalog_value=catalog(),
                    runtime_paths=paths,
                    frozen=True,
                )

            self.assertTrue(result)
            probe.assert_called_once_with(executable, worker_source)
            execute.assert_not_called()

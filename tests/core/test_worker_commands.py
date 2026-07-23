from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from tools.core.ml_runtime import RuntimeSelection
from tools.core.runtime_paths import MLRuntimePaths
from tools.core.worker_commands import (
    RuntimeUnavailableError,
    WorkerBackend,
    build_worker_command,
    resolve_worker_backend,
)
from tools import yolo_worker_entry


class WorkerCommandTests(TestCase):
    def test_source_commands_keep_worker_arguments_unchanged(self):
        resource_dir = Path(r"C:\repo")
        for kind, script_name in {
            "train": "yolo_train_worker.py",
            "detect": "yolo_detect_worker.py",
            "evaluate": "yolo_evaluate_worker.py",
            "tensorboard": "launch_tensorboard.py",
        }.items():
            program, arguments = build_worker_command(
                kind,
                ["--task-id", "任务 1", "--path", r"D:\图片\a.png"],
                frozen=False,
                resource_dir=resource_dir,
            )
            self.assertEqual(Path(program), Path(sys.executable))
            self.assertEqual(
                arguments,
                [
                    str(resource_dir / "tools" / script_name),
                    "--task-id",
                    "任务 1",
                    "--path",
                    r"D:\图片\a.png",
                ],
            )

    def test_frozen_command_targets_selected_runtime_worker(self):
        backend = WorkerBackend(
            kind="managed",
            program=Path(r"C:\Users\tester\AppData\Local\YOLOToolbox\runtimes\ml-cpu-win-x64-r1\YOLO工具箱Worker.exe"),
            prefix_args=(),
            runtime_id="ml-cpu-win-x64-r1",
        )
        program, arguments = build_worker_command(
            "detect",
            ["--model", "model.pt"],
            frozen=True,
            backend=backend,
        )
        self.assertEqual(
            Path(program),
            backend.program,
        )
        self.assertEqual(arguments, ["detect", "--model", "model.pt"])

    def test_frozen_command_without_runtime_is_rejected(self):
        with patch(
            "tools.core.worker_commands.resolve_worker_backend",
            side_effect=RuntimeUnavailableError("模型环境未安装"),
        ):
            with self.assertRaisesRegex(RuntimeUnavailableError, "未安装"):
                build_worker_command("train", [], frozen=True)

    def test_resolve_managed_backend_uses_neutral_runtime_directory(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_name:
            paths = MLRuntimePaths.from_environment(
                {"LOCALAPPDATA": str(Path(temp_name) / "local")}
            )
            runtime_dir = paths.runtimes_dir / "ml-cpu-win-x64-r1"
            runtime_dir.mkdir(parents=True)
            worker = runtime_dir / "YOLO工具箱Worker.exe"
            worker.write_bytes(b"worker")
            (runtime_dir / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_id": "ml-cpu-win-x64-r1",
                        "worker_protocol": 1,
                    }
                ),
                encoding="utf-8",
            )

            backend = resolve_worker_backend(
                frozen=True,
                runtime_paths=paths,
                selection=RuntimeSelection(
                    runtime_id="ml-cpu-win-x64-r1",
                    backend_kind="managed",
                ),
            )

            self.assertEqual(backend.kind, "managed")
            self.assertEqual(backend.program, worker)
            self.assertEqual(backend.runtime_id, "ml-cpu-win-x64-r1")

    def test_external_backend_prefixes_worker_source(self):
        backend = WorkerBackend(
            kind="external",
            program=Path(r"C:\Python313\python.exe"),
            prefix_args=(r"C:\Program Files\YOLO\_internal\worker_source\yolo_worker_entry.py",),
            runtime_id="external:C:\\Python313\\python.exe",
        )

        program, arguments = build_worker_command(
            "evaluate",
            ["--model", "model.pt"],
            frozen=True,
            backend=backend,
        )

        self.assertEqual(Path(program), backend.program)
        self.assertEqual(
            arguments,
            [
                r"C:\Program Files\YOLO\_internal\worker_source\yolo_worker_entry.py",
                "evaluate",
                "--model",
                "model.pt",
            ],
        )

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            build_worker_command("unknown", [])

    def test_worker_entry_dispatches_without_importing_qt(self):
        calls = []

        class FakeModule:
            @staticmethod
            def main(argv):
                calls.append(list(argv))
                return 7

        with patch.object(
            yolo_worker_entry.importlib,
            "import_module",
            return_value=FakeModule,
        ) as import_module:
            self.assertEqual(
                yolo_worker_entry.main(
                    ["evaluate", "--model", "模型.pt", "--data", "data.yaml"]
                ),
                7,
            )
        import_module.assert_called_once_with("yolo_evaluate_worker")
        self.assertEqual(calls, [["--model", "模型.pt", "--data", "data.yaml"]])

    def test_worker_entry_preserves_argparse_exit_code(self):
        class HelpModule:
            @staticmethod
            def main(argv):
                raise SystemExit(0)

        with patch.object(
            yolo_worker_entry.importlib,
            "import_module",
            return_value=HelpModule,
        ):
            self.assertEqual(yolo_worker_entry.main(["train", "--help"]), 0)

    def test_worker_entry_unknown_kind_emits_structured_error(self):
        stream = io.StringIO()
        with patch.object(yolo_worker_entry, "sys") as fake_sys:
            fake_sys.argv = ["worker.exe"]
            with patch("builtins.print") as print_mock:
                self.assertEqual(
                    yolo_worker_entry.main(["not-a-worker"], stream=stream),
                    2,
                )
        payload = json.loads(
            stream.getvalue().strip()[len(yolo_worker_entry.WORKER_ERROR_PREFIX):]
        )
        self.assertEqual(payload["kind"], "not-a-worker")
        self.assertIn("未知 Worker 类型", payload["error"])

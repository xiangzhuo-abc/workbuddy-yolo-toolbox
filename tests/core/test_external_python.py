from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from tools.core.external_python import (
    ExternalPythonCandidate,
    build_external_worker_backend,
    discover_external_pythons,
    probe_external_python,
)


class ExternalPythonTests(TestCase):
    def test_discovery_uses_launcher_and_path_without_recursive_scan(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            python_a = root / "Python313" / "python.exe"
            python_b = root / "Python312" / "python.exe"
            python_a.parent.mkdir()
            python_b.parent.mkdir()
            python_a.write_bytes(b"python")
            python_b.write_bytes(b"python")
            result = SimpleNamespace(
                returncode=0,
                stdout=(
                    f" -V:3.13 *        {python_a}\n"
                    f" -V:3.12          {python_b}\n"
                ),
                stderr="",
            )

            with patch(
                "tools.core.external_python.shutil.which",
                side_effect=lambda name: str(python_a) if name == "python" else None,
            ) as which:
                found = discover_external_pythons(
                    runner=lambda *args, **kwargs: result,
                )

            self.assertEqual(found, [python_a.resolve(), python_b.resolve()])
            self.assertEqual([call.args[0] for call in which.call_args_list], ["python", "python3"])

    def test_probe_accepts_complete_x64_environment(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            executable = root / "python.exe"
            worker = root / "worker" / "yolo_worker_entry.py"
            executable.write_bytes(b"python")
            worker.parent.mkdir()
            worker.write_text("", encoding="utf-8")
            payload = {
                "ok": True,
                "worker_protocol": 1,
                "runtime_id": f"external:{executable}",
                "python_version": "3.13.5",
                "architecture": "AMD64",
                "versions": {
                    "torch": "2.7.1+cu118",
                    "torchvision": "0.22.1+cu118",
                    "ultralytics": "8.4.90",
                    "tensorboard": "2.21.0",
                },
                "cuda_version": "11.8",
                "gpu_available": True,
                "gpu_names": ["Test GPU"],
                "errors": [],
            }
            calls = []

            def runner(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

            candidate = probe_external_python(executable, worker, runner=runner)

            self.assertTrue(candidate.ready)
            self.assertEqual(candidate.worker_protocol, 1)
            self.assertEqual(candidate.gpu_names, ("Test GPU",))
            self.assertEqual(
                calls[0],
                [
                    str(executable),
                    str(worker),
                    "probe",
                    "--runtime-id",
                    f"external:{executable}",
                ],
            )

    def test_probe_rejects_32_bit_or_missing_dependencies(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            executable = root / "python.exe"
            worker = root / "yolo_worker_entry.py"
            executable.write_bytes(b"python")
            worker.write_text("", encoding="utf-8")
            payload = {
                "ok": False,
                "worker_protocol": 1,
                "runtime_id": f"external:{executable}",
                "python_version": "3.13.5",
                "architecture": "x86",
                "versions": {"torch": None},
                "cuda_version": None,
                "gpu_available": False,
                "gpu_names": [],
                "errors": ["torch 导入失败"],
            }

            candidate = probe_external_python(
                executable,
                worker,
                runner=lambda *args, **kwargs: SimpleNamespace(
                    returncode=1,
                    stdout=json.dumps(payload),
                    stderr="",
                ),
            )

            self.assertFalse(candidate.ready)
            self.assertIn("x64", " ".join(candidate.errors))
            self.assertIn("torch", " ".join(candidate.errors))

    def test_candidate_builds_external_worker_backend(self):
        candidate = ExternalPythonCandidate(
            executable=Path(r"C:\Python313\python.exe"),
            python_version="3.13.5",
            architecture="AMD64",
            worker_protocol=1,
            versions={"torch": "2.7.1+cu118"},
            cuda_version="11.8",
            gpu_available=True,
            gpu_names=("Test GPU",),
            ready=True,
            errors=(),
        )
        worker = Path(r"C:\Program Files\YOLO\_internal\worker_source\yolo_worker_entry.py")

        backend = build_external_worker_backend(candidate, worker)

        self.assertEqual(backend.kind, "external")
        self.assertEqual(backend.program, candidate.executable)
        self.assertEqual(backend.prefix_args, (str(worker),))


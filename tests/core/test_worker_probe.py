from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from tools import yolo_worker_entry


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        return "Test GPU"


class WorkerProbeTests(TestCase):
    def test_probe_reports_dependency_and_gpu_state(self):
        modules = {
            "torch": SimpleNamespace(
                __version__="2.7.1+cu118",
                version=SimpleNamespace(cuda="11.8"),
                cuda=FakeCuda(),
            ),
            "torchvision": SimpleNamespace(__version__="0.22.1+cu118"),
            "ultralytics": SimpleNamespace(__version__="8.4.90"),
            "tensorboard": SimpleNamespace(__version__="2.21.0"),
        }

        with patch.object(
            yolo_worker_entry.importlib,
            "import_module",
            side_effect=lambda name: modules[name],
        ):
            payload = yolo_worker_entry.collect_runtime_probe("ml-cu118-win-x64-r1")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["worker_protocol"], 1)
        self.assertEqual(payload["runtime_id"], "ml-cu118-win-x64-r1")
        self.assertEqual(payload["cuda_version"], "11.8")
        self.assertTrue(payload["gpu_available"])
        self.assertEqual(payload["gpu_names"], ["Test GPU"])
        self.assertEqual(payload["errors"], [])

    def test_probe_failure_is_structured_and_nonzero(self):
        stream = io.StringIO()
        with patch.object(
            yolo_worker_entry.importlib,
            "import_module",
            side_effect=ImportError("torch missing"),
        ):
            exit_code = yolo_worker_entry.run_probe(
                ["--runtime-id", "ml-cpu-win-x64-r1"],
                stream=stream,
            )

        payload = json.loads(stream.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("torch", payload["errors"][0])

    def test_main_dispatches_probe_without_loading_worker_module(self):
        stream = io.StringIO()
        with patch.object(
            yolo_worker_entry,
            "run_probe",
            return_value=0,
        ) as run_probe, patch.object(
            yolo_worker_entry.importlib,
            "import_module",
        ) as import_module:
            result = yolo_worker_entry.main(
                ["probe", "--runtime-id", "ml-cpu-win-x64-r1"],
                stream=stream,
            )

        self.assertEqual(result, 0)
        run_probe.assert_called_once_with(
            ["--runtime-id", "ml-cpu-win-x64-r1"],
            stream=stream,
        )
        import_module.assert_not_called()


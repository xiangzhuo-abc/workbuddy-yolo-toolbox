import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import yolo_dataset_tools as tools


class TrainingDefaultTests(TestCase):
    def test_cuda_device_is_first_and_cpu_remains_available(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda _index: "Test GPU",
            )
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            devices = tools.detect_devices()

        self.assertEqual(devices[0], ("GPU 0: Test GPU（推荐）", "0"))
        self.assertIn(("自动选择（由 Ultralytics 决定）", ""), devices)
        self.assertIn(("CPU（兼容模式，速度较慢）", "cpu"), devices)

    def test_cpu_is_first_when_cuda_is_unavailable(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            devices = tools.detect_devices()

        self.assertEqual(devices[0], ("CPU（未检测到 GPU）", "cpu"))
        self.assertEqual(devices[1][1], "")

    def test_missing_torch_uses_nvidia_smi_without_importing_model_runtime(self):
        result = SimpleNamespace(
            returncode=0,
            stdout="NVIDIA Test GPU\nNVIDIA Second GPU\n",
        )
        with patch.dict(sys.modules, {"torch": None}), patch.object(
            tools.subprocess,
            "run",
            return_value=result,
        ) as run:
            devices = tools.detect_devices()

        self.assertEqual(devices[0], ("GPU 0: NVIDIA Test GPU（推荐）", "0"))
        self.assertEqual(devices[1], ("GPU 1: NVIDIA Second GPU", "1"))
        self.assertIn(("全部 2 块 GPU", "0,1"), devices)
        run.assert_called_once()

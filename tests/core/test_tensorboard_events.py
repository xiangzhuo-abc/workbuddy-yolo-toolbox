import io
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import launch_tensorboard
from core.task_protocol import TaskEventType, decode_task_event


class _FakeProcess:
    pid = 1234

    def __init__(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def kill(self):
        self.returncode = 1


class TensorBoardEventTests(TestCase):
    def _events(self, stream):
        return [
            event
            for line in stream.getvalue().splitlines()
            if (event := decode_task_event(line)) is not None
        ]

    def test_port_failure_emits_failed_final_event(self):
        stream = io.StringIO()
        with patch.object(launch_tensorboard, "find_available_port", side_effect=RuntimeError("无可用端口")):
            exit_code = launch_tensorboard.main(
                ["--port", "auto", "--task-id", "tb-1"],
                stream=stream,
            )

        events = self._events(stream)
        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertEqual(len([event for event in events if event.is_final]), 1)

    def test_health_check_failure_emits_failed_final_event(self):
        stream = io.StringIO()
        with patch.object(launch_tensorboard, "make_tensorboard_logdir", return_value=(Path("runs"), None)), \
             patch.object(launch_tensorboard, "ensure_tensorboard_events", return_value=(0, 0, [])), \
             patch.object(launch_tensorboard, "is_port_available", return_value=True), \
             patch.object(launch_tensorboard, "wait_until_ready", return_value=False), \
             patch.object(launch_tensorboard.subprocess, "Popen", return_value=_FakeProcess()):
            exit_code = launch_tensorboard.main(["--task-id", "tb-2"], stream=stream)

        events = self._events(stream)
        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertEqual(len([event for event in events if event.is_final]), 1)

    def test_ready_emits_result_final_event(self):
        stream = io.StringIO()
        fake = _FakeProcess()
        with patch.object(launch_tensorboard, "make_tensorboard_logdir", return_value=(Path("runs"), None)), \
             patch.object(launch_tensorboard, "ensure_tensorboard_events", return_value=(0, 0, [])), \
             patch.object(launch_tensorboard, "is_port_available", return_value=True), \
             patch.object(launch_tensorboard, "wait_until_ready", return_value=True), \
             patch.object(launch_tensorboard.subprocess, "Popen", return_value=fake):
            exit_code = launch_tensorboard.main(["--task-id", "tb-3"], stream=stream)

        events = self._events(stream)
        self.assertEqual(exit_code, 0)
        self.assertEqual(events[-1].type, TaskEventType.RESULT)
        self.assertEqual(len([event for event in events if event.is_final]), 1)

    def test_frozen_mode_uses_embedded_server_instead_of_recursive_process(self):
        stream = io.StringIO()
        with patch.object(launch_tensorboard, "make_tensorboard_logdir", return_value=(Path("runs"), None)), \
             patch.object(launch_tensorboard, "ensure_tensorboard_events", return_value=(0, 0, [])), \
             patch.object(launch_tensorboard, "is_port_available", return_value=True), \
             patch.object(launch_tensorboard, "_run_embedded_tensorboard", return_value=0) as embedded, \
             patch.object(launch_tensorboard.sys, "frozen", True, create=True), \
             patch.object(launch_tensorboard.subprocess, "Popen") as popen:
            exit_code = launch_tensorboard.main(["--task-id", "tb-frozen"], stream=stream)

        self.assertEqual(exit_code, 0)
        embedded.assert_called_once()
        popen.assert_not_called()

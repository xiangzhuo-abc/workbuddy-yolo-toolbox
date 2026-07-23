import io
import sys
from pathlib import Path
from unittest import TestCase

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import yolo_train_worker
from core.task_protocol import TaskEventType, decode_task_event


class TrainWorkerEventTests(TestCase):
    def _run(self, train_func):
        stream = io.StringIO()
        exit_code = yolo_train_worker.main(
            [
                "--task-id", "train-task",
                "--model", "model.pt",
                "--data", "data.yaml",
                "--epochs", "20",
                "--batch", "4",
                "--imgsz", "640",
                "--project", "runs",
                "--name", "test-run",
            ],
            train_func=train_func,
            stream=stream,
        )
        events = [
            decode_task_event(line)
            for line in stream.getvalue().splitlines()
        ]
        return exit_code, events

    def test_success_emits_started_log_and_one_result(self):
        received = {}

        def train_func(**kwargs):
            received.update(kwargs)
            kwargs["emit"]("info", "Epoch 1/20")
            return True

        exit_code, events = self._run(train_func)

        self.assertEqual(exit_code, 0)
        self.assertEqual(events[0].type, TaskEventType.STARTED)
        self.assertIn(TaskEventType.LOG, [event.type for event in events])
        self.assertEqual(events[-1].type, TaskEventType.RESULT)
        self.assertEqual(len([event for event in events if event.is_final]), 1)
        self.assertEqual(received["epochs"], 20)
        self.assertEqual(received["name"], "test-run")
        self.assertEqual(received["device"], "")

    def test_explicit_gpu_device_is_forwarded(self):
        received = {}
        stream = io.StringIO()

        exit_code = yolo_train_worker.main(
            [
                "--model", "model.pt",
                "--data", "data.yaml",
                "--device", "0",
            ],
            train_func=lambda **kwargs: received.update(kwargs) or True,
            stream=stream,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(received["device"], "0")

    def test_backend_failure_emits_failed_final_event(self):
        exit_code, events = self._run(lambda **kwargs: False)

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertEqual(len([event for event in events if event.is_final]), 1)

    def test_unhandled_exception_emits_failed_with_traceback(self):
        def train_func(**kwargs):
            raise RuntimeError("training exploded")

        exit_code, events = self._run(train_func)

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertIn("training exploded", events[-1].message)
        self.assertIn("traceback", events[-1].payload)

    def test_keyboard_interrupt_emits_cancelled(self):
        def train_func(**kwargs):
            raise KeyboardInterrupt

        exit_code, events = self._run(train_func)

        self.assertEqual(exit_code, 130)
        self.assertEqual(events[-1].type, TaskEventType.CANCELLED)
        self.assertEqual(len([event for event in events if event.is_final]), 1)

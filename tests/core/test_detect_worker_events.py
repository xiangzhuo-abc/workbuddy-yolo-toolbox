import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import yolo_detect_worker
from core.task_protocol import TaskEventType, decode_task_event


class DetectWorkerEventTests(TestCase):
    def _events(self, stream):
        return [
            event
            for line in stream.getvalue().splitlines()
            if (event := decode_task_event(line)) is not None
        ]

    def test_single_detection_success_has_one_result(self):
        stream = io.StringIO()
        with TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "result.json"
            exit_code = yolo_detect_worker.main(
                [
                    "--task-id", "detect-1",
                    "--model", "model.pt",
                    "--image", "image.png",
                    "--output", str(output),
                ],
                load_model_func=lambda path: (object(), {"0": "按钮"}),
                predict_func=lambda model, names, image, conf: {
                    "ok": True,
                    "names": names,
                    "detections": [{"class_id": 0, "name": "按钮"}],
                },
                stream=stream,
            )
            output_payload = json.loads(output.read_text(encoding="utf-8"))

        events = self._events(stream)
        self.assertEqual(exit_code, 0)
        self.assertEqual(events[0].type, TaskEventType.STARTED)
        self.assertEqual(events[-1].type, TaskEventType.RESULT)
        self.assertEqual(len([event for event in events if event.is_final]), 1)
        self.assertTrue(output_payload["ok"])

    def test_single_detection_exception_has_failed_final(self):
        stream = io.StringIO()
        with TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "result.json"

            def fail_predict(*args):
                raise RuntimeError("predict failed")

            exit_code = yolo_detect_worker.main(
                [
                    "--task-id", "detect-1",
                    "--model", "model.pt",
                    "--image", "image.png",
                    "--output", str(output),
                ],
                load_model_func=lambda path: (object(), {}),
                predict_func=fail_predict,
                stream=stream,
            )
            output_payload = json.loads(output.read_text(encoding="utf-8"))

        events = self._events(stream)
        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertIn("predict failed", events[-1].message)
        self.assertFalse(output_payload["ok"])

    def test_serve_emits_worker_and_request_lifecycles(self):
        stream = io.StringIO()
        stdin = io.StringIO(
            json.dumps({
                "cmd": "detect",
                "id": 7,
                "task_id": "request-7",
                "image": "image.png",
                "conf": 0.5,
            })
            + "\n"
            + json.dumps({"cmd": "quit"})
            + "\n"
        )
        exit_code = yolo_detect_worker.serve(
            Path("model.pt"),
            task_id="worker-1",
            load_model_func=lambda path: (object(), {"0": "按钮"}),
            predict_func=lambda model, names, image, conf: {
                "ok": True,
                "names": names,
                "detections": [],
            },
            stream=stream,
            stdin=stdin,
        )

        events = self._events(stream)
        worker_events = [event for event in events if event.task_id == "worker-1"]
        request_events = [event for event in events if event.task_id == "request-7"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(worker_events[0].type, TaskEventType.STARTED)
        self.assertIn(TaskEventType.PROGRESS, [event.type for event in worker_events])
        self.assertEqual(worker_events[-1].type, TaskEventType.CANCELLED)
        self.assertEqual(request_events[0].type, TaskEventType.STARTED)
        self.assertEqual(request_events[-1].type, TaskEventType.RESULT)
        self.assertEqual(len([event for event in request_events if event.is_final]), 1)

    def test_failed_request_does_not_prevent_worker_cancel_final(self):
        stream = io.StringIO()
        stdin = io.StringIO(
            json.dumps({
                "cmd": "detect",
                "id": 8,
                "task_id": "request-8",
                "image": "bad.png",
                "conf": 0.5,
            })
            + "\n"
            + json.dumps({"cmd": "quit"})
            + "\n"
        )

        def fail_predict(*args):
            raise RuntimeError("bad image")

        exit_code = yolo_detect_worker.serve(
            Path("model.pt"),
            task_id="worker-1",
            load_model_func=lambda path: (object(), {}),
            predict_func=fail_predict,
            stream=stream,
            stdin=stdin,
        )

        events = self._events(stream)
        request_final = [
            event for event in events if event.task_id == "request-8" and event.is_final
        ]
        worker_final = [
            event for event in events if event.task_id == "worker-1" and event.is_final
        ]
        self.assertEqual(exit_code, 0)
        self.assertEqual(request_final[0].type, TaskEventType.FAILED)
        self.assertEqual(worker_final[0].type, TaskEventType.CANCELLED)

    def test_model_load_failure_emits_worker_failed(self):
        stream = io.StringIO()

        def fail_load(path):
            raise RuntimeError("model broken")

        exit_code = yolo_detect_worker.serve(
            Path("model.pt"),
            task_id="worker-1",
            load_model_func=fail_load,
            stream=stream,
            stdin=io.StringIO(),
        )

        events = self._events(stream)
        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertIn("model broken", events[-1].message)

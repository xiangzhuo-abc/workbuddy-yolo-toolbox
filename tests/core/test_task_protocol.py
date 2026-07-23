import io
from unittest import TestCase

from tools.core.task_protocol import (
    TASK_EVENT_PREFIX,
    TaskEvent,
    TaskEventEmitter,
    TaskEventType,
    decode_task_event,
    encode_task_event,
)


class TaskProtocolTests(TestCase):
    def test_event_json_roundtrip_preserves_stable_fields(self):
        event = TaskEvent(
            task_id="task-1",
            type=TaskEventType.PROGRESS,
            stage="training",
            progress=0.42,
            message="Epoch 8/20",
            payload={"epoch": 8, "total": 20},
        )

        line = encode_task_event(event)
        decoded = decode_task_event(line)

        self.assertTrue(line.startswith(TASK_EVENT_PREFIX))
        self.assertEqual(decoded, event)
        self.assertEqual(decoded.to_dict()["type"], "progress")

    def test_non_protocol_line_returns_none(self):
        self.assertIsNone(decode_task_event("ordinary worker output"))

    def test_invalid_progress_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskEvent(
                task_id="task-1",
                type=TaskEventType.PROGRESS,
                stage="training",
                progress=1.5,
            )

    def test_emitter_writes_utf8_events_and_only_one_final_event(self):
        stream = io.StringIO()
        emitter = TaskEventEmitter("task-1", "training", stream=stream)

        self.assertTrue(emitter.started("开始训练"))
        self.assertTrue(emitter.log("载入模型", level="info"))
        self.assertTrue(emitter.result("训练完成", {"best": "best.pt"}))
        self.assertFalse(emitter.failed("不应重复结束"))

        events = [
            decode_task_event(line)
            for line in stream.getvalue().splitlines()
        ]
        self.assertEqual(
            [event.type for event in events],
            [TaskEventType.STARTED, TaskEventType.LOG, TaskEventType.RESULT],
        )
        self.assertEqual(events[-1].payload["best"], "best.pt")

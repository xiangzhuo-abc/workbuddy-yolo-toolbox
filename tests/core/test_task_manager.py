from unittest import TestCase

from tools.core.task_manager import TaskLifecycle, TaskManager, TaskStatus
from tools.core.task_protocol import TaskEvent, TaskEventType


class TaskLifecycleTests(TestCase):
    def test_success_path_has_exactly_one_final_event(self):
        task = TaskLifecycle("task-1", "training")

        self.assertTrue(task.accept(TaskEvent("task-1", TaskEventType.STARTED, "training")))
        self.assertTrue(
            task.accept(
                TaskEvent(
                    "task-1",
                    TaskEventType.PROGRESS,
                    "training",
                    progress=0.5,
                )
            )
        )
        self.assertTrue(task.accept(TaskEvent("task-1", TaskEventType.RESULT, "training")))
        self.assertFalse(task.accept(TaskEvent("task-1", TaskEventType.FAILED, "training")))

        self.assertIs(task.status, TaskStatus.SUCCEEDED)
        self.assertIs(task.final_event.type, TaskEventType.RESULT)
        self.assertEqual(len([event for event in task.events if event.is_final]), 1)

    def test_process_exit_without_final_event_becomes_failure(self):
        task = TaskLifecycle("task-1", "training")
        task.accept(TaskEvent("task-1", TaskEventType.STARTED, "training"))

        event = task.synthesize_process_exit(exit_code=0, crashed=False)

        self.assertIs(event.type, TaskEventType.FAILED)
        self.assertIs(task.status, TaskStatus.FAILED)
        self.assertIn("最终事件", event.message)

    def test_requested_cancel_becomes_cancelled_when_process_exits(self):
        task = TaskLifecycle("task-1", "training")
        task.accept(TaskEvent("task-1", TaskEventType.STARTED, "training"))

        self.assertTrue(task.request_cancel())
        event = task.synthesize_process_exit(exit_code=15, crashed=True)

        self.assertIs(event.type, TaskEventType.CANCELLED)
        self.assertIs(task.status, TaskStatus.CANCELLED)

    def test_final_failure_after_cancel_is_ignored_until_process_exit(self):
        task = TaskLifecycle("task-1", "training")
        task.accept(TaskEvent("task-1", TaskEventType.STARTED, "training"))

        self.assertTrue(task.request_cancel())
        accepted = task.accept(
            TaskEvent("task-1", TaskEventType.FAILED, "training", message="被终止")
        )

        self.assertFalse(accepted)
        self.assertIsNone(task.final_event)
        event = task.synthesize_process_exit(exit_code=1, crashed=True)
        self.assertIs(event.type, TaskEventType.CANCELLED)

    def test_event_for_another_task_is_rejected(self):
        task = TaskLifecycle("task-1", "training")
        with self.assertRaises(ValueError):
            task.accept(TaskEvent("task-2", TaskEventType.STARTED, "training"))


class TaskManagerTests(TestCase):
    def test_cancel_callback_runs_once_and_task_finishes_cancelled(self):
        cancelled = []
        manager = TaskManager()
        task = manager.create_task(
            "training",
            task_id="task-1",
            cancel_callback=lambda: cancelled.append("task-1"),
        )
        manager.accept(TaskEvent("task-1", TaskEventType.STARTED, "training"))

        self.assertTrue(manager.request_cancel("task-1"))
        self.assertFalse(manager.request_cancel("task-1"))
        final = manager.process_exited("task-1", exit_code=1, crashed=True)

        self.assertEqual(cancelled, ["task-1"])
        self.assertIs(final.type, TaskEventType.CANCELLED)
        self.assertEqual(manager.active_task_ids, ())
        self.assertIs(task.status, TaskStatus.CANCELLED)

    def test_unknown_event_creates_managed_lifecycle(self):
        manager = TaskManager()
        event = TaskEvent("request-7", TaskEventType.STARTED, "detection")

        self.assertTrue(manager.accept(event))

        self.assertIs(manager.get("request-7").status, TaskStatus.RUNNING)

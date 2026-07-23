from __future__ import annotations

from enum import Enum
from typing import Callable
from uuid import uuid4

from .task_protocol import TaskEvent, TaskEventType


class TaskStatus(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_final(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }


class TaskLifecycle:
    def __init__(self, task_id: str, stage: str):
        self.task_id = str(task_id)
        self.stage = str(stage)
        self.status = TaskStatus.STARTING
        self.events: list[TaskEvent] = []
        self.final_event: TaskEvent | None = None

    def accept(self, event: TaskEvent) -> bool:
        if event.task_id != self.task_id:
            raise ValueError(
                f"任务事件 ID 不匹配: {event.task_id} != {self.task_id}"
            )
        if self.final_event is not None:
            return False

        # 取消已经被界面确认后，进程在退出前可能仍吐出一个失败事件。
        # 该事件不能覆盖取消语义，最终结果交给 process_exited() 或明确的
        # cancelled 事件收口。
        if self.status is TaskStatus.CANCELLING and event.type in {
            TaskEventType.RESULT,
            TaskEventType.FAILED,
        }:
            return False

        if event.type in {
            TaskEventType.STARTED,
            TaskEventType.PROGRESS,
            TaskEventType.LOG,
        }:
            if self.status is not TaskStatus.CANCELLING:
                self.status = TaskStatus.RUNNING
        elif event.type is TaskEventType.RESULT:
            self.status = TaskStatus.SUCCEEDED
            self.final_event = event
        elif event.type is TaskEventType.FAILED:
            self.status = TaskStatus.FAILED
            self.final_event = event
        elif event.type is TaskEventType.CANCELLED:
            self.status = TaskStatus.CANCELLED
            self.final_event = event

        self.events.append(event)
        return True

    def request_cancel(self) -> bool:
        if self.final_event is not None or self.status is TaskStatus.CANCELLING:
            return False
        self.status = TaskStatus.CANCELLING
        return True

    def synthesize_process_exit(
        self,
        exit_code: int,
        crashed: bool = False,
    ) -> TaskEvent:
        if self.final_event is not None:
            return self.final_event
        payload = {"exit_code": int(exit_code), "crashed": bool(crashed)}
        if self.status is TaskStatus.CANCELLING:
            event = TaskEvent(
                self.task_id,
                TaskEventType.CANCELLED,
                self.stage,
                message="任务取消后进程已退出",
                payload=payload,
            )
        else:
            event = TaskEvent(
                self.task_id,
                TaskEventType.FAILED,
                self.stage,
                message="任务进程退出但没有发送最终事件",
                payload=payload,
            )
        self.accept(event)
        return event


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, TaskLifecycle] = {}
        self._cancel_callbacks: dict[str, Callable[[], None]] = {}

    def create_task(
        self,
        stage: str,
        task_id: str | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> TaskLifecycle:
        actual_id = task_id or str(uuid4())
        existing = self._tasks.get(actual_id)
        if existing is not None and not existing.status.is_final:
            raise ValueError(f"任务已存在: {actual_id}")
        task = TaskLifecycle(actual_id, stage)
        self._tasks[actual_id] = task
        if cancel_callback is not None:
            self._cancel_callbacks[actual_id] = cancel_callback
        return task

    def get(self, task_id: str) -> TaskLifecycle | None:
        return self._tasks.get(str(task_id))

    def accept(self, event: TaskEvent) -> bool:
        task = self._tasks.get(event.task_id)
        if task is None:
            task = self.create_task(event.stage, task_id=event.task_id)
        accepted = task.accept(event)
        if task.status.is_final:
            self._cancel_callbacks.pop(task.task_id, None)
        return accepted

    def request_cancel(self, task_id: str) -> bool:
        task = self._tasks.get(str(task_id))
        if task is None or not task.request_cancel():
            return False
        callback = self._cancel_callbacks.get(task.task_id)
        if callback is not None:
            callback()
        return True

    def process_exited(
        self,
        task_id: str,
        exit_code: int,
        crashed: bool = False,
    ) -> TaskEvent:
        task = self._tasks.get(str(task_id))
        if task is None:
            raise KeyError(f"未知任务: {task_id}")
        event = task.synthesize_process_exit(exit_code, crashed)
        self._cancel_callbacks.pop(task.task_id, None)
        return event

    def cancel_all(self) -> tuple[str, ...]:
        cancelled = []
        for task_id in tuple(self.active_task_ids):
            if self.request_cancel(task_id):
                cancelled.append(task_id)
        return tuple(cancelled)

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        return tuple(
            task_id
            for task_id, task in self._tasks.items()
            if not task.status.is_final
        )

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TextIO


TASK_EVENT_PREFIX = "__YOLO_TASK_EVENT__"


class TaskEventType(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    LOG = "log"
    RESULT = "result"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        return self in {
            TaskEventType.RESULT,
            TaskEventType.FAILED,
            TaskEventType.CANCELLED,
        }


@dataclass(frozen=True)
class TaskEvent:
    task_id: str
    type: TaskEventType
    stage: str
    progress: float | None = None
    message: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        stage = str(self.stage).strip()
        if not task_id:
            raise ValueError("task_id 不能为空")
        if not stage:
            raise ValueError("stage 不能为空")
        event_type = TaskEventType(self.type)
        progress = self.progress
        if progress is not None:
            progress = float(progress)
            if not 0.0 <= progress <= 1.0:
                raise ValueError("progress 必须在 [0, 1] 范围内")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload or {})))

    @property
    def is_final(self) -> bool:
        return self.type.is_final

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": self.type.value,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskEvent":
        if not isinstance(data, Mapping):
            raise ValueError("任务事件根节点必须是对象")
        return cls(
            task_id=data.get("task_id", ""),
            type=data.get("type", ""),
            stage=data.get("stage", ""),
            progress=data.get("progress"),
            message=data.get("message", ""),
            payload=data.get("payload") or {},
        )


def encode_task_event(event: TaskEvent) -> str:
    return TASK_EVENT_PREFIX + json.dumps(
        event.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_task_event(line: str) -> TaskEvent | None:
    text = str(line).strip()
    if not text.startswith(TASK_EVENT_PREFIX):
        return None
    payload_text = text[len(TASK_EVENT_PREFIX):]
    try:
        payload = json.loads(payload_text)
        return TaskEvent.from_dict(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"无效任务事件: {exc}") from exc


class TaskEventEmitter:
    def __init__(
        self,
        task_id: str,
        stage: str,
        stream: TextIO | None = None,
    ):
        self.task_id = str(task_id)
        self.stage = str(stage)
        self.stream = stream if stream is not None else sys.__stdout__
        self.final_emitted = False

    def emit(
        self,
        event_type: TaskEventType,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        progress: float | None = None,
        stage: str | None = None,
    ) -> bool:
        event = TaskEvent(
            task_id=self.task_id,
            type=event_type,
            stage=stage or self.stage,
            progress=progress,
            message=message,
            payload=payload or {},
        )
        if self.final_emitted:
            return False
        self.stream.write(encode_task_event(event) + "\n")
        self.stream.flush()
        if event.is_final:
            self.final_emitted = True
        return True

    def started(self, message: str = "", payload: Mapping[str, Any] | None = None) -> bool:
        return self.emit(TaskEventType.STARTED, message, payload)

    def progress(
        self,
        value: float,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.emit(TaskEventType.PROGRESS, message, payload, progress=value)

    def log(
        self,
        message: str,
        level: str = "info",
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        log_payload = dict(payload or {})
        log_payload["level"] = str(level or "info")
        return self.emit(TaskEventType.LOG, message, log_payload)

    def result(
        self,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.emit(TaskEventType.RESULT, message, payload)

    def failed(
        self,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.emit(TaskEventType.FAILED, message, payload)

    def cancelled(
        self,
        message: str = "任务已取消",
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.emit(TaskEventType.CANCELLED, message, payload)

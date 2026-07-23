from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    code: str
    severity: IssueSeverity
    message: str
    path: Path | None = None
    suggested_action: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": str(self.path) if self.path is not None else None,
            "suggested_action": self.suggested_action,
        }
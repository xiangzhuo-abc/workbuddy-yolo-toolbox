import json
from pathlib import Path

from .issues import Issue, IssueSeverity


class ConfigStore:
    def __init__(self, path: Path, defaults: dict | None = None):
        self.path = Path(path)
        self.defaults = dict(defaults or {})
        self.last_issue: Issue | None = None

    def load(self) -> dict:
        config = dict(self.defaults)
        self.last_issue = None
        if not self.path.exists():
            return config

        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("配置根节点必须是对象")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.last_issue = Issue(
                code="config.invalid_json",
                severity=IssueSeverity.ERROR,
                message=f"配置读取失败: {exc}",
                path=self.path,
                suggested_action="检查配置文件或在工具中重新保存配置",
            )
            return config

        config.update(loaded)
        return config

    def save(self, config: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temp_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.path)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
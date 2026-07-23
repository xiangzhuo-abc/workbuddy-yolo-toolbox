from dataclasses import dataclass
from pathlib import Path

from .runtime_paths import RuntimePaths


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    dataset_dir: Path
    runs_dir: Path
    models_dir: Path
    logs_dir: Path
    config_file: Path

    @classmethod
    def from_project_dir(
        cls,
        project_dir: Path,
        dataset_dir: Path | None = None,
        runs_dir: Path | None = None,
    ) -> "ProjectPaths":
        project = Path(project_dir)
        return cls(
            project_dir=project,
            dataset_dir=Path(dataset_dir) if dataset_dir is not None else project / "dataset",
            runs_dir=Path(runs_dir) if runs_dir is not None else project / "runs",
            models_dir=project / "models",
            logs_dir=project / "logs",
            config_file=project / "config" / "tool_config.json",
        )

    @classmethod
    def from_runtime_paths(
        cls,
        runtime: RuntimePaths,
        dataset_dir: Path | None = None,
        runs_dir: Path | None = None,
        models_dir: Path | None = None,
    ) -> "ProjectPaths":
        """从产品化运行路径创建核心服务路径。"""
        return cls(
            project_dir=runtime.resource_dir,
            dataset_dir=Path(dataset_dir) if dataset_dir is not None else runtime.dataset_dir,
            runs_dir=Path(runs_dir) if runs_dir is not None else runtime.runs_dir,
            models_dir=Path(models_dir) if models_dir is not None else runtime.models_dir,
            logs_dir=runtime.logs_dir,
            config_file=runtime.config_file,
        )

    def image_dir(self, split: str) -> Path:
        return self.dataset_dir / "images" / split

    def label_dir(self, split: str) -> Path:
        return self.dataset_dir / "labels" / split

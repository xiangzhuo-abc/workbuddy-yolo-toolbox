"""机器学习运行时清单、状态和本地发现契约。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse

from .runtime_paths import MLRuntimePaths


WORKER_EXE_NAME = "YOLO工具箱Worker.exe"
SUPPORTED_WORKER_PROTOCOLS = frozenset({1})
_RUNTIME_ID_PATTERN = re.compile(r"^ml-(cpu|cu118)-win-x64-r[1-9][0-9]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOWNLOAD_FIELDS = ("archive_size", "installed_size", "sha256", "url")


class RuntimeProfile(str, Enum):
    CPU = "cpu"
    CUDA118 = "cuda118"


def is_valid_runtime_id(value: str) -> bool:
    return bool(_RUNTIME_ID_PATTERN.fullmatch(str(value or "").strip()))


@dataclass(frozen=True)
class RuntimeArtifact:
    runtime_id: str
    profile: RuntimeProfile
    platform: str
    architecture: str
    worker_protocol: int
    torch_version: str
    torchvision_version: str
    ultralytics_version: str
    cuda_version: str | None
    archive_size: int | None = None
    installed_size: int | None = None
    sha256: str | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuntimeArtifact":
        if not isinstance(data, Mapping):
            raise ValueError("运行时清单项必须是对象")

        runtime_id = str(data.get("runtime_id", "")).strip()
        if not is_valid_runtime_id(runtime_id):
            raise ValueError(f"运行时编号格式无效: {runtime_id!r}")
        try:
            profile = RuntimeProfile(str(data.get("profile", "")).strip())
        except ValueError as exc:
            raise ValueError("运行时 profile 只能是 cpu 或 cuda118") from exc
        expected_profile = (
            RuntimeProfile.CPU if "-cpu-" in runtime_id else RuntimeProfile.CUDA118
        )
        if profile is not expected_profile:
            raise ValueError("运行时编号与 profile 不一致")

        platform_name = str(data.get("platform", "")).strip().lower()
        architecture = str(data.get("architecture", "")).strip().lower()
        if platform_name != "windows":
            raise ValueError("运行时平台必须为 windows")
        if architecture != "x86_64":
            raise ValueError("运行时架构必须为 x86_64")

        try:
            worker_protocol = int(data.get("worker_protocol", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Worker 协议必须是正整数") from exc
        if worker_protocol <= 0:
            raise ValueError("Worker 协议必须是正整数")

        versions = {
            name: str(data.get(name, "")).strip()
            for name in (
                "torch_version",
                "torchvision_version",
                "ultralytics_version",
            )
        }
        missing_versions = [name for name, value in versions.items() if not value]
        if missing_versions:
            raise ValueError(f"运行时依赖版本不能为空: {', '.join(missing_versions)}")

        raw_cuda = data.get("cuda_version")
        cuda_version = None if raw_cuda is None else str(raw_cuda).strip() or None
        if profile is RuntimeProfile.CPU and cuda_version is not None:
            raise ValueError("CPU 运行时不能声明 CUDA 版本")
        if profile is RuntimeProfile.CUDA118 and cuda_version != "11.8":
            raise ValueError("CUDA 11.8 运行时必须声明 CUDA 版本 11.8")

        present_download_fields = [
            name for name in _DOWNLOAD_FIELDS if data.get(name) not in (None, "")
        ]
        if present_download_fields and len(present_download_fields) != len(_DOWNLOAD_FIELDS):
            raise ValueError("下载元数据必须完整提供大小、安装大小、SHA256 和 URL")

        archive_size = None
        installed_size = None
        sha256 = None
        url = None
        if present_download_fields:
            try:
                archive_size = int(data["archive_size"])
                installed_size = int(data["installed_size"])
            except (TypeError, ValueError) as exc:
                raise ValueError("下载元数据中的大小必须是正整数") from exc
            if archive_size <= 0 or installed_size <= 0:
                raise ValueError("下载元数据中的大小必须是正整数")
            sha256 = str(data["sha256"]).strip().lower()
            if not _SHA256_PATTERN.fullmatch(sha256):
                raise ValueError("SHA256 必须是 64 位小写十六进制摘要")
            url = str(data["url"]).strip()
            parsed = urlparse(url)
            if parsed.scheme.lower() != "https" or not parsed.netloc:
                raise ValueError("运行时下载地址必须是完整 HTTPS URL")

        return cls(
            runtime_id=runtime_id,
            profile=profile,
            platform=platform_name,
            architecture=architecture,
            worker_protocol=worker_protocol,
            torch_version=versions["torch_version"],
            torchvision_version=versions["torchvision_version"],
            ultralytics_version=versions["ultralytics_version"],
            cuda_version=cuda_version,
            archive_size=archive_size,
            installed_size=installed_size,
            sha256=sha256,
            url=url,
        )

    @property
    def is_downloadable(self) -> bool:
        return all(
            value is not None
            for value in (self.archive_size, self.installed_size, self.sha256, self.url)
        )

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.runtime_id,
            self.profile,
            self.platform,
            self.architecture,
            self.worker_protocol,
            self.torch_version,
            self.torchvision_version,
            self.ultralytics_version,
            self.cuda_version,
        )

    def is_compatible_with(self, other: "RuntimeArtifact") -> bool:
        return self.compatibility_key == other.compatibility_key

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["profile"] = self.profile.value
        return {
            key: value
            for key, value in result.items()
            if key not in _DOWNLOAD_FIELDS or value is not None
        }


@dataclass(frozen=True)
class RuntimeCatalog:
    catalog_version: int
    artifacts: tuple[RuntimeArtifact, ...]

    @classmethod
    def from_file(cls, path: Path) -> "RuntimeCatalog":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取运行时清单: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuntimeCatalog":
        if not isinstance(data, Mapping):
            raise ValueError("运行时清单必须是对象")
        try:
            version = int(data.get("catalog_version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("运行时清单版本必须是正整数") from exc
        if version <= 0:
            raise ValueError("运行时清单版本必须是正整数")
        raw_artifacts = data.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ValueError("运行时清单必须包含 artifacts")
        artifacts = tuple(RuntimeArtifact.from_dict(item) for item in raw_artifacts)
        runtime_ids = [artifact.runtime_id for artifact in artifacts]
        if len(set(runtime_ids)) != len(runtime_ids):
            raise ValueError("运行时清单包含重复编号")
        return cls(version, artifacts)

    def for_profile(
        self,
        profile: RuntimeProfile | str,
    ) -> tuple[RuntimeArtifact, ...]:
        expected = profile if isinstance(profile, RuntimeProfile) else RuntimeProfile(profile)
        return tuple(item for item in self.artifacts if item.profile is expected)

    def get(self, runtime_id: str) -> RuntimeArtifact | None:
        expected = str(runtime_id)
        return next(
            (item for item in self.artifacts if item.runtime_id == expected),
            None,
        )


@dataclass(frozen=True)
class RuntimeSelection:
    runtime_id: str | None = None
    backend_kind: str = "managed"
    external_python: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuntimeSelection":
        if not isinstance(data, Mapping):
            raise ValueError("运行时状态必须是对象")
        runtime_id = str(data.get("runtime_id") or "").strip() or None
        backend_kind = str(data.get("backend_kind") or "managed").strip().lower()
        if backend_kind not in {"managed", "external"}:
            raise ValueError("运行时后端只能是 managed 或 external")
        external_python = str(data.get("external_python") or "").strip() or None
        if backend_kind == "managed" and external_python is not None:
            raise ValueError("托管运行时不能声明外部 Python")
        if backend_kind == "external" and external_python is None:
            raise ValueError("外部运行时必须声明 Python 路径")
        return cls(runtime_id, backend_kind, external_python)

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "backend_kind": self.backend_kind,
            "external_python": self.external_python,
        }


class RuntimeStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.last_error: str | None = None

    def load(self) -> RuntimeSelection:
        self.last_error = None
        if not self.path.is_file():
            return RuntimeSelection()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return RuntimeSelection.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.last_error = f"无法读取运行时状态: {exc}"
            return RuntimeSelection()

    def save(self, selection: RuntimeSelection) -> None:
        validated = RuntimeSelection.from_dict(selection.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(validated.to_dict(), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        self.last_error = None


@dataclass(frozen=True)
class RuntimeCandidate:
    artifact: RuntimeArtifact
    runtime_dir: Path
    worker_executable: Path
    selected: bool = False


class RuntimeDiscovery:
    def __init__(
        self,
        catalog: RuntimeCatalog,
        paths: MLRuntimePaths,
        state_store: RuntimeStateStore,
        *,
        supported_protocols: Iterable[int] = SUPPORTED_WORKER_PROTOCOLS,
    ):
        self.catalog = catalog
        self.paths = paths
        self.state_store = state_store
        self.supported_protocols = frozenset(int(value) for value in supported_protocols)

    def find_compatible(
        self,
        profile: RuntimeProfile | str | None = None,
    ) -> list[RuntimeCandidate]:
        expected_profile = (
            None
            if profile is None
            else profile if isinstance(profile, RuntimeProfile) else RuntimeProfile(profile)
        )
        selection = self.state_store.load()
        candidates: list[RuntimeCandidate] = []
        for artifact in self.catalog.artifacts:
            if expected_profile is not None and artifact.profile is not expected_profile:
                continue
            if artifact.worker_protocol not in self.supported_protocols:
                continue
            runtime_dir = self.paths.runtimes_dir / artifact.runtime_id
            worker = runtime_dir / WORKER_EXE_NAME
            manifest_path = runtime_dir / "runtime.json"
            if not worker.is_file() or not manifest_path.is_file():
                continue
            try:
                manifest = RuntimeArtifact.from_dict(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not artifact.is_compatible_with(manifest):
                continue
            candidates.append(
                RuntimeCandidate(
                    artifact=artifact,
                    runtime_dir=runtime_dir,
                    worker_executable=worker,
                    selected=(
                        selection.backend_kind == "managed"
                        and selection.runtime_id == artifact.runtime_id
                    ),
                )
            )
        candidates.sort(
            key=lambda item: (
                not item.selected,
                item.artifact.profile.value,
                item.artifact.runtime_id,
            )
        )
        return candidates


__all__ = [
    "RuntimeArtifact",
    "RuntimeCandidate",
    "RuntimeCatalog",
    "RuntimeDiscovery",
    "RuntimeProfile",
    "RuntimeSelection",
    "RuntimeStateStore",
    "SUPPORTED_WORKER_PROTOCOLS",
    "WORKER_EXE_NAME",
    "is_valid_runtime_id",
]

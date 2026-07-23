"""机器学习运行时下载、校验和原子安装。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping
from uuid import uuid4

from .ml_runtime import (
    RuntimeArtifact,
    RuntimeCandidate,
    RuntimeCatalog,
    RuntimeSelection,
    RuntimeStateStore,
    WORKER_EXE_NAME,
)
from .runtime_paths import MLRuntimePaths


DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT = 60
MIN_INSTALL_MARGIN = 512 * 1024**2


@dataclass(frozen=True)
class DownloadProgress:
    stage: str
    completed: int
    total: int
    bytes_per_second: float = 0.0


class RuntimeInstallError(RuntimeError):
    def __init__(self, code: str, message: str, recoverable: bool = True):
        super().__init__(str(message))
        self.code = str(code)
        self.message = str(message)
        self.recoverable = bool(recoverable)


ProgressCallback = Callable[[DownloadProgress], None]
CancelCallback = Callable[[], bool]
ProbeRunner = Callable[[Path, RuntimeArtifact], Mapping[str, object]]


class RuntimeInstaller:
    def __init__(
        self,
        paths: MLRuntimePaths | None = None,
        *,
        opener=None,
        probe_runner: ProbeRunner | None = None,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.paths = paths or MLRuntimePaths.from_environment()
        self.opener = opener or urllib.request.build_opener()
        self.probe_runner = probe_runner or self._default_probe_runner
        self.disk_usage = disk_usage
        self.chunk_size = max(1, int(chunk_size))
        self.timeout = max(1, int(timeout))

    def install_cached(
        self,
        artifact: RuntimeArtifact,
        archive_path: Path,
        progress: ProgressCallback | None = None,
    ) -> RuntimeCandidate:
        self._require_downloadable(artifact)
        with self._runtime_lock(artifact.runtime_id):
            self._validate_archive(artifact, Path(archive_path))
            return self._install_archive(artifact, Path(archive_path), progress)

    def download_and_install(
        self,
        artifact: RuntimeArtifact,
        cancel: CancelCallback | None = None,
        progress: ProgressCallback | None = None,
    ) -> RuntimeCandidate:
        self._require_downloadable(artifact)
        with self._runtime_lock(artifact.runtime_id):
            self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
            archive_path = self._cache_path(artifact)
            if archive_path.is_file():
                try:
                    self._validate_archive(artifact, archive_path)
                except RuntimeInstallError:
                    archive_path.unlink(missing_ok=True)
                else:
                    return self._install_archive(artifact, archive_path, progress)

            self._download(artifact, archive_path, cancel, progress)
            self._validate_archive(artifact, archive_path)
            return self._install_archive(artifact, archive_path, progress)

    def import_offline(
        self,
        archive_path: Path,
        catalog: RuntimeCatalog,
        progress: ProgressCallback | None = None,
    ) -> RuntimeCandidate:
        path = Path(archive_path)
        if not path.is_file():
            raise RuntimeInstallError("offline_missing", f"离线运行时包不存在: {path}")
        size = path.stat().st_size
        digest = self._sha256(path)
        artifact = next(
            (
                item
                for item in catalog.artifacts
                if item.is_downloadable
                and item.archive_size == size
                and item.sha256 == digest
            ),
            None,
        )
        if artifact is None:
            raise RuntimeInstallError(
                "offline_untrusted",
                "离线运行时包不在当前版本的可信清单中",
            )
        return self.install_cached(artifact, path, progress)

    @staticmethod
    def _require_downloadable(artifact: RuntimeArtifact) -> None:
        if not artifact.is_downloadable:
            raise RuntimeInstallError(
                "artifact_incomplete",
                f"运行时 {artifact.runtime_id} 缺少完整下载元数据",
                recoverable=False,
            )

    def _cache_path(self, artifact: RuntimeArtifact) -> Path:
        return self.paths.cache_dir / f"{artifact.runtime_id}.zip"

    @contextmanager
    def _runtime_lock(self, runtime_id: str) -> Iterator[None]:
        lock_dir = self.paths.root_dir / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{runtime_id}.lock"
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeInstallError(
                "runtime_busy",
                f"运行时 {runtime_id} 正在被另一个安装任务使用",
            ) from exc
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            descriptor = -1
            yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            lock_path.unlink(missing_ok=True)
            try:
                lock_dir.rmdir()
            except OSError:
                pass

    def _download(
        self,
        artifact: RuntimeArtifact,
        archive_path: Path,
        cancel: CancelCallback | None,
        progress: ProgressCallback | None,
        *,
        _allow_validator_restart: bool = True,
    ) -> None:
        part_path = archive_path.with_suffix(archive_path.suffix + ".part")
        metadata_path = part_path.with_suffix(part_path.suffix + ".json")
        metadata = self._load_part_metadata(metadata_path)
        offset = part_path.stat().st_size if part_path.is_file() else 0
        if self._part_matches_artifact(artifact, metadata, offset) and offset == artifact.archive_size:
            os.replace(part_path, archive_path)
            metadata_path.unlink(missing_ok=True)
            try:
                self._validate_archive(artifact, archive_path)
            except RuntimeInstallError:
                archive_path.unlink(missing_ok=True)
                offset = 0
                metadata = {}
            else:
                return
        elif not self._part_is_reusable(artifact, metadata, offset):
            part_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            metadata = {}
            offset = 0

        self._check_download_disk_space(artifact, offset)

        headers = {"User-Agent": "YOLOToolbox-RuntimeInstaller/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            validator = metadata.get("etag") or metadata.get("last_modified")
            if validator:
                headers["If-Range"] = str(validator)
        request = urllib.request.Request(str(artifact.url), headers=headers)

        started_at = time.monotonic()
        downloaded_this_request = 0
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                if status_code not in {200, 206}:
                    raise RuntimeInstallError(
                        "http_status",
                        f"运行时下载失败，HTTP 状态码: {status_code}",
                    )
                append = bool(offset and status_code == 206)
                if not append:
                    offset = 0
                response_headers = getattr(response, "headers", {})
                if append and self._response_validator_changed(metadata, response_headers):
                    part_path.unlink(missing_ok=True)
                    metadata_path.unlink(missing_ok=True)
                    if _allow_validator_restart:
                        return self._download(
                            artifact,
                            archive_path,
                            cancel,
                            progress,
                            _allow_validator_restart=False,
                        )
                    raise RuntimeInstallError(
                        "remote_changed",
                        "远端运行时文件已变化，无法继续旧断点",
                    )
                current_metadata = {
                    "url": artifact.url,
                    "etag": response_headers.get("ETag"),
                    "last_modified": response_headers.get("Last-Modified"),
                    "total": artifact.archive_size,
                    "downloaded": offset,
                }
                self._save_part_metadata(metadata_path, current_metadata)
                mode = "ab" if append else "wb"
                with part_path.open(mode) as stream:
                    while True:
                        if cancel is not None and cancel():
                            current_metadata["downloaded"] = offset + downloaded_this_request
                            self._save_part_metadata(metadata_path, current_metadata)
                            raise RuntimeInstallError("cancelled", "运行时下载已取消")
                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        stream.write(chunk)
                        downloaded_this_request += len(chunk)
                        completed = offset + downloaded_this_request
                        if completed > int(artifact.archive_size or 0):
                            stream.flush()
                            raise RuntimeInstallError(
                                "download_too_large",
                                "下载内容超过可信清单声明的大小",
                            )
                        current_metadata["downloaded"] = completed
                        self._save_part_metadata(metadata_path, current_metadata)
                        if progress is not None:
                            elapsed = max(time.monotonic() - started_at, 0.001)
                            progress(
                                DownloadProgress(
                                    "downloading",
                                    completed,
                                    int(artifact.archive_size or 0),
                                    downloaded_this_request / elapsed,
                                )
                            )
                    stream.flush()
                    os.fsync(stream.fileno())
        except RuntimeInstallError:
            if part_path.is_file() and part_path.stat().st_size > int(
                artifact.archive_size or 0
            ):
                part_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeInstallError("network", f"运行时下载失败: {exc}") from exc

        completed_size = part_path.stat().st_size if part_path.is_file() else 0
        if completed_size != artifact.archive_size:
            raise RuntimeInstallError(
                "download_incomplete",
                f"下载大小不完整: {completed_size} / {artifact.archive_size}",
            )
        os.replace(part_path, archive_path)
        metadata_path.unlink(missing_ok=True)

    @staticmethod
    def _load_part_metadata(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _save_part_metadata(path: Path, data: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(dict(data), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _part_is_reusable(
        artifact: RuntimeArtifact,
        metadata: Mapping[str, object],
        offset: int,
    ) -> bool:
        if offset <= 0 or not metadata:
            return False
        try:
            return (
                RuntimeInstaller._part_matches_artifact(artifact, metadata, offset)
                and offset < int(artifact.archive_size or 0)
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _part_matches_artifact(
        artifact: RuntimeArtifact,
        metadata: Mapping[str, object],
        offset: int,
    ) -> bool:
        try:
            return (
                bool(metadata)
                and str(metadata.get("url")) == artifact.url
                and int(metadata.get("total", -1)) == artifact.archive_size
                and int(metadata.get("downloaded", -1)) == offset
                and 0 < offset <= int(artifact.archive_size or 0)
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _response_validator_changed(
        metadata: Mapping[str, object],
        response_headers: Mapping[str, object],
    ) -> bool:
        old_etag = str(metadata.get("etag") or "")
        new_etag = str(response_headers.get("ETag") or "")
        if old_etag and new_etag:
            return old_etag != new_etag
        old_modified = str(metadata.get("last_modified") or "")
        new_modified = str(response_headers.get("Last-Modified") or "")
        return bool(old_modified and new_modified and old_modified != new_modified)

    def _validate_archive(self, artifact: RuntimeArtifact, archive_path: Path) -> None:
        if not archive_path.is_file():
            raise RuntimeInstallError("archive_missing", f"运行时包不存在: {archive_path}")
        actual_size = archive_path.stat().st_size
        if actual_size != artifact.archive_size:
            raise RuntimeInstallError(
                "archive_size",
                f"运行时包大小不匹配: {actual_size} / {artifact.archive_size}",
            )
        actual_sha256 = self._sha256(archive_path)
        if actual_sha256 != artifact.sha256:
            raise RuntimeInstallError("archive_hash", "运行时包 SHA256 校验失败")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(DEFAULT_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _install_archive(
        self,
        artifact: RuntimeArtifact,
        archive_path: Path,
        progress: ProgressCallback | None,
    ) -> RuntimeCandidate:
        self._check_disk_space(artifact)
        self.paths.staging_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = self.paths.staging_dir / f"{artifact.runtime_id}-{uuid4().hex}"
        staging_dir.mkdir()
        try:
            if progress is not None:
                progress(DownloadProgress("extracting", 0, int(artifact.installed_size or 0)))
            self._safe_extract(artifact, archive_path, staging_dir, progress)
            self._validate_extracted_runtime(artifact, staging_dir)
            probe = dict(self.probe_runner(staging_dir / WORKER_EXE_NAME, artifact))
            self._validate_probe(artifact, probe)
            return self._activate(artifact, staging_dir)
        except RuntimeInstallError:
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            raise RuntimeInstallError("install", f"运行时安装失败: {exc}") from exc
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            try:
                self.paths.staging_dir.rmdir()
            except OSError:
                pass

    def _check_disk_space(self, artifact: RuntimeArtifact) -> None:
        free = self._free_disk_bytes()
        installed_size = int(artifact.installed_size or 0)
        required = installed_size + max(MIN_INSTALL_MARGIN, installed_size // 10)
        if free < required:
            raise RuntimeInstallError(
                "disk_space",
                f"运行时安装空间不足，需要 {required} 字节，可用 {free} 字节",
            )

    def _check_download_disk_space(
        self,
        artifact: RuntimeArtifact,
        downloaded: int,
    ) -> None:
        free = self._free_disk_bytes()
        installed_size = int(artifact.installed_size or 0)
        remaining = max(0, int(artifact.archive_size or 0) - int(downloaded))
        required = (
            remaining
            + installed_size
            + max(MIN_INSTALL_MARGIN, installed_size // 10)
        )
        if free < required:
            raise RuntimeInstallError(
                "disk_space",
                f"运行时下载和安装空间不足，需要 {required} 字节，可用 {free} 字节",
            )

    def _free_disk_bytes(self) -> int:
        target = self.paths.root_dir
        while not target.exists() and target != target.parent:
            target = target.parent
        try:
            return int(getattr(self.disk_usage(target), "free"))
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeInstallError("disk_check", f"无法检查磁盘空间: {exc}") from exc

    def _safe_extract(
        self,
        artifact: RuntimeArtifact,
        archive_path: Path,
        staging_dir: Path,
        progress: ProgressCallback | None,
    ) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names: set[str] = set()
            total_size = 0
            for info in infos:
                normalised = info.filename.replace("\\", "/")
                pure = PurePosixPath(normalised)
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    not normalised
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or any(":" in part for part in pure.parts)
                    or stat.S_ISLNK(mode)
                ):
                    raise RuntimeInstallError(
                        "zip_path",
                        f"运行时包包含非法路径: {info.filename}",
                    )
                key = pure.as_posix().lower()
                if key in names:
                    raise RuntimeInstallError(
                        "zip_duplicate",
                        f"运行时包包含重复路径: {info.filename}",
                    )
                names.add(key)
                if not info.is_dir():
                    total_size += int(info.file_size)
            if total_size != artifact.installed_size:
                raise RuntimeInstallError(
                    "installed_size",
                    f"运行时解压大小不匹配: {total_size} / {artifact.installed_size}",
                )

            completed = 0
            for info in infos:
                pure = PurePosixPath(info.filename.replace("\\", "/"))
                destination = staging_dir.joinpath(*pure.parts)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, destination.open("wb") as target:
                    while True:
                        chunk = source.read(self.chunk_size)
                        if not chunk:
                            break
                        target.write(chunk)
                        completed += len(chunk)
                        if progress is not None:
                            progress(
                                DownloadProgress(
                                    "extracting",
                                    completed,
                                    int(artifact.installed_size or 0),
                                )
                            )

    @staticmethod
    def _validate_extracted_runtime(
        artifact: RuntimeArtifact,
        staging_dir: Path,
    ) -> None:
        manifest_path = staging_dir / "runtime.json"
        worker_path = staging_dir / WORKER_EXE_NAME
        if not manifest_path.is_file() or not worker_path.is_file():
            raise RuntimeInstallError(
                "runtime_structure",
                "运行时包缺少 runtime.json 或 Worker",
            )
        try:
            manifest = RuntimeArtifact.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeInstallError("runtime_manifest", f"运行时清单无效: {exc}") from exc
        if not artifact.is_compatible_with(manifest):
            raise RuntimeInstallError("runtime_mismatch", "运行时内部清单与可信清单不一致")

    @staticmethod
    def _validate_probe(
        artifact: RuntimeArtifact,
        probe: Mapping[str, object],
    ) -> None:
        try:
            ok = bool(probe.get("ok"))
            runtime_id = str(probe.get("runtime_id", ""))
            protocol = int(probe.get("worker_protocol", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeInstallError("probe_invalid", f"Worker 探针结果无效: {exc}") from exc
        if not ok:
            errors = probe.get("errors") or []
            detail = "; ".join(str(item) for item in errors) or "未知错误"
            raise RuntimeInstallError("probe_failed", f"Worker 探针失败: {detail}")
        if runtime_id != artifact.runtime_id or protocol != artifact.worker_protocol:
            raise RuntimeInstallError("probe_mismatch", "Worker 探针身份或协议不匹配")

    def _activate(
        self,
        artifact: RuntimeArtifact,
        staging_dir: Path,
    ) -> RuntimeCandidate:
        self.paths.runtimes_dir.mkdir(parents=True, exist_ok=True)
        final_dir = self.paths.runtimes_dir / artifact.runtime_id
        backup_dir = self.paths.runtimes_dir / f".{artifact.runtime_id}.rollback-{uuid4().hex}"
        had_previous = final_dir.exists()
        try:
            if had_previous:
                os.replace(final_dir, backup_dir)
            os.replace(staging_dir, final_dir)
            RuntimeStateStore(self.paths.state_file).save(
                RuntimeSelection(
                    runtime_id=artifact.runtime_id,
                    backend_kind="managed",
                )
            )
        except BaseException as exc:
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            if backup_dir.exists():
                os.replace(backup_dir, final_dir)
            if isinstance(exc, RuntimeInstallError):
                raise
            raise RuntimeInstallError("activate", f"运行时激活失败: {exc}") from exc
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        return RuntimeCandidate(
            artifact=artifact,
            runtime_dir=final_dir,
            worker_executable=final_dir / WORKER_EXE_NAME,
            selected=True,
        )

    def _default_probe_runner(
        self,
        worker_path: Path,
        artifact: RuntimeArtifact,
    ) -> Mapping[str, object]:
        try:
            result = subprocess.run(
                [
                    str(worker_path),
                    "probe",
                    "--runtime-id",
                    artifact.runtime_id,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeInstallError("probe_start", f"无法启动 Worker 探针: {exc}") from exc
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeInstallError("probe_empty", "Worker 探针没有返回结果")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeInstallError("probe_json", "Worker 探针返回了无效 JSON") from exc
        if result.returncode != 0 and payload.get("ok"):
            payload["ok"] = False
            payload["errors"] = [f"Worker 退出码: {result.returncode}"]
        return payload


__all__ = [
    "DownloadProgress",
    "RuntimeInstallError",
    "RuntimeInstaller",
]

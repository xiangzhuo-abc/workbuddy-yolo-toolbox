from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from tools.core.ml_runtime import RuntimeArtifact, RuntimeCatalog, RuntimeStateStore
from tools.core.runtime_installer import RuntimeInstallError, RuntimeInstaller
from tools.core.runtime_paths import MLRuntimePaths


def static_runtime_data() -> dict[str, object]:
    return {
        "runtime_id": "ml-cpu-win-x64-r1",
        "profile": "cpu",
        "platform": "windows",
        "architecture": "x86_64",
        "worker_protocol": 1,
        "torch_version": "2.7.1+cpu",
        "torchvision_version": "0.22.1+cpu",
        "ultralytics_version": "8.4.90",
        "cuda_version": None,
    }


def make_runtime_archive(
    path: Path,
    *,
    extra_entries: dict[str, bytes] | None = None,
) -> RuntimeArtifact:
    manifest = (json.dumps(static_runtime_data(), ensure_ascii=False) + "\n").encode("utf-8")
    entries = {
        "runtime.json": manifest,
        "YOLO工具箱Worker.exe": b"test-worker",
        "_internal/marker.txt": b"runtime",
    }
    entries.update(extra_entries or {})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    content = path.read_bytes()
    data = static_runtime_data()
    data.update(
        {
            "archive_size": len(content),
            "installed_size": sum(len(value) for value in entries.values()),
            "sha256": hashlib.sha256(content).hexdigest(),
            "url": "https://example.invalid/ml-cpu-win-x64-r1.zip",
        }
    )
    return RuntimeArtifact.from_dict(data)


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, *, status: int, headers: dict[str, str]):
        super().__init__(content)
        self.status = status
        self.headers = headers

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class RecordingOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout=0):
        self.requests.append((request, timeout))
        return self.responses.pop(0)


class RuntimeInstallerTests(TestCase):
    def make_installer(self, root: Path, **kwargs) -> RuntimeInstaller:
        paths = MLRuntimePaths.from_environment(
            {"YOLO_TOOLBOX_RUNTIME_DIR": str(root / "runtime-root")}
        )
        return RuntimeInstaller(
            paths,
            probe_runner=lambda worker, artifact: {
                "ok": True,
                "runtime_id": artifact.runtime_id,
                "worker_protocol": artifact.worker_protocol,
            },
            **kwargs,
        )

    def test_cached_archive_installs_and_selects_runtime(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "runtime.zip"
            artifact = make_runtime_archive(archive)
            installer = self.make_installer(root)

            candidate = installer.install_cached(artifact, archive)

            self.assertTrue(candidate.worker_executable.is_file())
            self.assertTrue(candidate.selected)
            selection = RuntimeStateStore(installer.paths.state_file).load()
            self.assertEqual(selection.runtime_id, artifact.runtime_id)
            self.assertFalse(installer.paths.staging_dir.exists())

    def test_hash_mismatch_is_rejected_without_creating_runtime(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "runtime.zip"
            artifact = make_runtime_archive(archive)
            archive.write_bytes(archive.read_bytes() + b"corrupt")
            installer = self.make_installer(root)

            with self.assertRaisesRegex(RuntimeInstallError, "大小|SHA256"):
                installer.install_cached(artifact, archive)

            self.assertFalse(
                (installer.paths.runtimes_dir / artifact.runtime_id).exists()
            )

    def test_zip_path_traversal_is_rejected(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "runtime.zip"
            artifact = make_runtime_archive(
                archive,
                extra_entries={"../outside.txt": b"escape"},
            )
            installer = self.make_installer(root)

            with self.assertRaisesRegex(RuntimeInstallError, "非法路径"):
                installer.install_cached(artifact, archive)

            self.assertFalse((root / "outside.txt").exists())

    def test_complete_cache_avoids_network_request(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            opener = RecordingOpener([])
            installer = self.make_installer(root, opener=opener)
            installer.paths.cache_dir.mkdir(parents=True)
            cache = installer.paths.cache_dir / f"{artifact.runtime_id}.zip"
            cache.write_bytes(source.read_bytes())

            installer.download_and_install(artifact)

            self.assertEqual(opener.requests, [])

    def test_interrupted_download_resumes_with_range_and_if_range(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            content = source.read_bytes()
            split = len(content) // 2
            response = FakeResponse(
                content[split:],
                status=206,
                headers={"ETag": '"runtime-v1"'},
            )
            opener = RecordingOpener([response])
            installer = self.make_installer(root, opener=opener)
            installer.paths.cache_dir.mkdir(parents=True)
            part = installer.paths.cache_dir / f"{artifact.runtime_id}.zip.part"
            part.write_bytes(content[:split])
            part.with_suffix(part.suffix + ".json").write_text(
                json.dumps(
                    {
                        "url": artifact.url,
                        "etag": '"runtime-v1"',
                        "last_modified": None,
                        "total": artifact.archive_size,
                        "downloaded": split,
                    }
                ),
                encoding="utf-8",
            )

            installer.download_and_install(artifact)

            request = opener.requests[0][0]
            self.assertEqual(request.get_header("Range"), f"bytes={split}-")
            self.assertEqual(request.get_header("If-range"), '"runtime-v1"')
            self.assertFalse(part.exists())

    def test_server_ignoring_range_replaces_partial_content(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            content = source.read_bytes()
            opener = RecordingOpener(
                [FakeResponse(content, status=200, headers={"ETag": '"v2"'})]
            )
            installer = self.make_installer(root, opener=opener)
            installer.paths.cache_dir.mkdir(parents=True)
            part = installer.paths.cache_dir / f"{artifact.runtime_id}.zip.part"
            part.write_bytes(content[:20])
            part.with_suffix(part.suffix + ".json").write_text(
                json.dumps(
                    {
                        "url": artifact.url,
                        "etag": '"v1"',
                        "last_modified": None,
                        "total": artifact.archive_size,
                        "downloaded": 20,
                    }
                ),
                encoding="utf-8",
            )

            installer.download_and_install(artifact)

            self.assertEqual(len(opener.requests), 1)
            self.assertFalse(part.exists())

    def test_changed_etag_on_partial_response_restarts_from_zero(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            content = source.read_bytes()
            split = len(content) // 2
            opener = RecordingOpener(
                [
                    FakeResponse(
                        content[split:],
                        status=206,
                        headers={"ETag": '"v2"'},
                    ),
                    FakeResponse(content, status=200, headers={"ETag": '"v2"'}),
                ]
            )
            installer = self.make_installer(root, opener=opener)
            installer.paths.cache_dir.mkdir(parents=True)
            part = installer.paths.cache_dir / f"{artifact.runtime_id}.zip.part"
            part.write_bytes(content[:split])
            part.with_suffix(part.suffix + ".json").write_text(
                json.dumps(
                    {
                        "url": artifact.url,
                        "etag": '"v1"',
                        "last_modified": None,
                        "total": artifact.archive_size,
                        "downloaded": split,
                    }
                ),
                encoding="utf-8",
            )

            installer.download_and_install(artifact)

            self.assertEqual(len(opener.requests), 2)
            self.assertEqual(opener.requests[0][0].get_header("Range"), f"bytes={split}-")
            self.assertIsNone(opener.requests[1][0].get_header("Range"))

    def test_complete_part_file_is_promoted_without_network(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            content = source.read_bytes()
            opener = RecordingOpener([])
            installer = self.make_installer(root, opener=opener)
            installer.paths.cache_dir.mkdir(parents=True)
            part = installer.paths.cache_dir / f"{artifact.runtime_id}.zip.part"
            part.write_bytes(content)
            part.with_suffix(part.suffix + ".json").write_text(
                json.dumps(
                    {
                        "url": artifact.url,
                        "etag": '"v1"',
                        "last_modified": None,
                        "total": artifact.archive_size,
                        "downloaded": artifact.archive_size,
                    }
                ),
                encoding="utf-8",
            )

            installer.download_and_install(artifact)

            self.assertEqual(opener.requests, [])

    def test_download_checks_archive_and_install_space_before_network(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            opener = RecordingOpener([])
            installer = self.make_installer(
                root,
                opener=opener,
                disk_usage=lambda path: SimpleNamespace(free=100),
            )

            with self.assertRaisesRegex(RuntimeInstallError, "空间不足"):
                installer.download_and_install(artifact)

            self.assertEqual(opener.requests, [])

    def test_cancel_keeps_valid_partial_download(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.zip"
            artifact = make_runtime_archive(source)
            content = source.read_bytes()
            opener = RecordingOpener(
                [FakeResponse(content, status=200, headers={"ETag": '"v1"'})]
            )
            installer = self.make_installer(root, opener=opener, chunk_size=16)
            calls = 0

            def cancel():
                nonlocal calls
                calls += 1
                return calls > 2

            with self.assertRaisesRegex(RuntimeInstallError, "取消"):
                installer.download_and_install(artifact, cancel=cancel)

            part = installer.paths.cache_dir / f"{artifact.runtime_id}.zip.part"
            self.assertTrue(part.is_file())
            self.assertGreater(part.stat().st_size, 0)
            self.assertLess(part.stat().st_size, len(content))
            self.assertTrue(part.with_suffix(part.suffix + ".json").is_file())

    def test_offline_import_selects_matching_catalog_artifact(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "offline.zip"
            artifact = make_runtime_archive(archive)
            catalog = RuntimeCatalog(1, (artifact,))
            installer = self.make_installer(root)

            candidate = installer.import_offline(archive, catalog)

            self.assertEqual(candidate.artifact.runtime_id, artifact.runtime_id)

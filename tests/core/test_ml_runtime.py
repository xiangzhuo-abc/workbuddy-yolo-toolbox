from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.core.ml_runtime import (
    RuntimeArtifact,
    RuntimeCatalog,
    RuntimeDiscovery,
    RuntimeProfile,
    RuntimeSelection,
    RuntimeStateStore,
)
from tools.core.runtime_paths import MLRuntimePaths


def artifact_data(runtime_id: str, profile: str) -> dict[str, object]:
    is_gpu = profile == "cuda118"
    suffix = "+cu118" if is_gpu else "+cpu"
    return {
        "runtime_id": runtime_id,
        "profile": profile,
        "platform": "windows",
        "architecture": "x86_64",
        "worker_protocol": 1,
        "torch_version": f"2.7.1{suffix}",
        "torchvision_version": f"0.22.1{suffix}",
        "ultralytics_version": "8.4.90",
        "cuda_version": "11.8" if is_gpu else None,
    }


class RuntimeArtifactTests(TestCase):
    def test_static_profile_is_valid_but_not_downloadable(self):
        artifact = RuntimeArtifact.from_dict(
            artifact_data("ml-cpu-win-x64-r1", "cpu")
        )

        self.assertEqual(artifact.profile, RuntimeProfile.CPU)
        self.assertFalse(artifact.is_downloadable)

    def test_complete_download_metadata_is_validated(self):
        data = artifact_data("ml-cu118-win-x64-r1", "cuda118")
        data.update(
            {
                "archive_size": 123,
                "installed_size": 456,
                "sha256": "a" * 64,
                "url": "https://example.invalid/ml-cu118-win-x64-r1.zip",
            }
        )

        artifact = RuntimeArtifact.from_dict(data)

        self.assertTrue(artifact.is_downloadable)
        self.assertEqual(artifact.archive_size, 123)

    def test_partial_or_untrusted_download_metadata_is_rejected(self):
        partial = artifact_data("ml-cpu-win-x64-r1", "cpu")
        partial["archive_size"] = 123
        with self.assertRaisesRegex(ValueError, "下载元数据"):
            RuntimeArtifact.from_dict(partial)

        insecure = artifact_data("ml-cpu-win-x64-r1", "cpu")
        insecure.update(
            {
                "archive_size": 123,
                "installed_size": 456,
                "sha256": "b" * 64,
                "url": "http://example.invalid/runtime.zip",
            }
        )
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            RuntimeArtifact.from_dict(insecure)

    def test_cpu_and_gpu_cuda_versions_cannot_be_mixed(self):
        cpu = artifact_data("ml-cpu-win-x64-r1", "cpu")
        cpu["cuda_version"] = "11.8"
        with self.assertRaisesRegex(ValueError, "CPU"):
            RuntimeArtifact.from_dict(cpu)

        gpu = artifact_data("ml-cu118-win-x64-r1", "cuda118")
        gpu["cuda_version"] = None
        with self.assertRaisesRegex(ValueError, "CUDA"):
            RuntimeArtifact.from_dict(gpu)


class RuntimeCatalogTests(TestCase):
    def test_catalog_loads_profiles_and_rejects_duplicate_ids(self):
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "catalog_version": 1,
                        "artifacts": [
                            artifact_data("ml-cpu-win-x64-r1", "cpu"),
                            artifact_data("ml-cu118-win-x64-r1", "cuda118"),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            catalog = RuntimeCatalog.from_file(path)

            self.assertEqual(catalog.catalog_version, 1)
            self.assertEqual(
                catalog.for_profile(RuntimeProfile.CPU)[0].runtime_id,
                "ml-cpu-win-x64-r1",
            )

            duplicate = json.loads(path.read_text(encoding="utf-8"))
            duplicate["artifacts"].append(duplicate["artifacts"][0])
            path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复"):
                RuntimeCatalog.from_file(path)


class RuntimeStateTests(TestCase):
    def test_state_round_trip_uses_managed_runtime(self):
        with TemporaryDirectory() as temp_name:
            state_file = Path(temp_name) / "config" / "runtime_state.json"
            store = RuntimeStateStore(state_file)
            selection = RuntimeSelection(
                runtime_id="ml-cu118-win-x64-r1",
                backend_kind="managed",
            )

            store.save(selection)

            self.assertEqual(store.load(), selection)
            self.assertIsNone(store.last_error)
            self.assertFalse(state_file.with_suffix(".json.tmp").exists())

    def test_corrupt_state_returns_empty_selection(self):
        with TemporaryDirectory() as temp_name:
            state_file = Path(temp_name) / "runtime_state.json"
            state_file.write_text("{broken", encoding="utf-8")
            store = RuntimeStateStore(state_file)

            self.assertEqual(store.load(), RuntimeSelection())
            self.assertIn("运行时状态", store.last_error or "")


class RuntimeDiscoveryTests(TestCase):
    def test_discovery_only_checks_catalog_ids_and_prioritises_selection(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            paths = MLRuntimePaths.from_environment(
                {"LOCALAPPDATA": str(root / "local")}
            )
            catalog = RuntimeCatalog.from_dict(
                {
                    "catalog_version": 1,
                    "artifacts": [
                        artifact_data("ml-cpu-win-x64-r1", "cpu"),
                        artifact_data("ml-cu118-win-x64-r1", "cuda118"),
                    ],
                }
            )
            state = RuntimeStateStore(paths.state_file)
            state.save(
                RuntimeSelection(
                    runtime_id="ml-cu118-win-x64-r1",
                    backend_kind="managed",
                )
            )
            for artifact in catalog.artifacts:
                runtime_dir = paths.runtimes_dir / artifact.runtime_id
                runtime_dir.mkdir(parents=True)
                (runtime_dir / "YOLO工具箱Worker.exe").write_bytes(b"worker")
                (runtime_dir / "runtime.json").write_text(
                    json.dumps(artifact.to_dict()),
                    encoding="utf-8",
                )
            unrelated = paths.runtimes_dir / "foreign-runtime"
            unrelated.mkdir()
            (unrelated / "YOLO工具箱Worker.exe").write_bytes(b"foreign")

            candidates = RuntimeDiscovery(catalog, paths, state).find_compatible()

            self.assertEqual(
                [candidate.artifact.runtime_id for candidate in candidates],
                ["ml-cu118-win-x64-r1", "ml-cpu-win-x64-r1"],
            )
            self.assertNotIn(
                unrelated,
                [candidate.runtime_dir for candidate in candidates],
            )

    def test_discovery_filters_profile_and_protocol(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            paths = MLRuntimePaths.from_environment(
                {"LOCALAPPDATA": str(root / "local")}
            )
            cpu = artifact_data("ml-cpu-win-x64-r1", "cpu")
            incompatible = artifact_data("ml-cu118-win-x64-r1", "cuda118")
            incompatible["worker_protocol"] = 2
            catalog = RuntimeCatalog.from_dict(
                {"catalog_version": 1, "artifacts": [cpu, incompatible]}
            )
            for artifact in catalog.artifacts:
                runtime_dir = paths.runtimes_dir / artifact.runtime_id
                runtime_dir.mkdir(parents=True)
                (runtime_dir / "YOLO工具箱Worker.exe").write_bytes(b"worker")
                (runtime_dir / "runtime.json").write_text(
                    json.dumps(artifact.to_dict()),
                    encoding="utf-8",
                )

            discovery = RuntimeDiscovery(
                catalog,
                paths,
                RuntimeStateStore(paths.state_file),
                supported_protocols={1},
            )

            self.assertEqual(
                [item.artifact.runtime_id for item in discovery.find_compatible()],
                ["ml-cpu-win-x64-r1"],
            )
            self.assertEqual(
                discovery.find_compatible(RuntimeProfile.CUDA118),
                [],
            )


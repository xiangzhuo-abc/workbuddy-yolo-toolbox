from __future__ import annotations

import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.build_runtime import (
    check_runtime_archive,
    check_runtime_manifest,
    package_runtime_directory,
    prune_duplicate_torch_dlls,
    write_runtime_catalog,
)
from tools.core.ml_runtime import RuntimeArtifact, RuntimeCatalog


def artifact_data(profile: str) -> dict[str, object]:
    gpu = profile == "cuda118"
    return {
        "runtime_id": "ml-cu118-win-x64-r1" if gpu else "ml-cpu-win-x64-r1",
        "profile": profile,
        "platform": "windows",
        "architecture": "x86_64",
        "worker_protocol": 1,
        "torch_version": "2.7.1+cu118" if gpu else "2.7.1+cpu",
        "torchvision_version": "0.22.1+cu118" if gpu else "0.22.1+cpu",
        "ultralytics_version": "8.4.90",
        "cuda_version": "11.8" if gpu else None,
    }


class RuntimeBuildTests(TestCase):
    @staticmethod
    def _write(path: Path, content: bytes = b"MZ") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _make_runtime(self, root: Path, profile: str) -> RuntimeArtifact:
        artifact = RuntimeArtifact.from_dict(artifact_data(profile))
        self._write(root / "YOLO工具箱Worker.exe")
        self._write(root / "_internal" / "torch" / "lib" / "torch_cpu.dll")
        if profile == "cuda118":
            self._write(root / "_internal" / "torch" / "lib" / "torch_cuda.dll")
            self._write(root / "_internal" / "torch" / "lib" / "cudnn64_9.dll")
            self._write(root / "_internal" / "torch" / "lib" / "cublas64_11.dll")
        (root / "runtime.json").write_text(
            json.dumps(artifact.to_dict()),
            encoding="utf-8",
        )
        (root / "LICENSE").write_text("license", encoding="utf-8")
        (root / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
        return artifact

    def test_cpu_and_gpu_runtime_manifests_are_distinct(self):
        with TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            cpu = base / "cpu"
            gpu = base / "gpu"
            self._make_runtime(cpu, "cpu")
            self._make_runtime(gpu, "cuda118")

            self.assertEqual(check_runtime_manifest(cpu, "cpu"), [])
            self.assertEqual(check_runtime_manifest(gpu, "cuda118"), [])
            self._write(cpu / "_internal" / "torch" / "lib" / "torch_cuda.dll")
            self.assertTrue(any("CPU 运行时包含 CUDA" in item for item in check_runtime_manifest(cpu, "cpu")))

    def test_gpu_runtime_requires_cuda_libraries_and_rejects_gui(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_runtime(root, "cuda118")
            (root / "_internal" / "torch" / "lib" / "torch_cuda.dll").unlink()
            self._write(root / "YOLO工具箱.exe")

            errors = check_runtime_manifest(root, "cuda118")

        self.assertTrue(any("缺少 GPU 动态库" in item for item in errors))
        self.assertTrue(any("禁止运行时包含主界面" in item for item in errors))

    def test_duplicate_torch_root_dlls_are_pruned(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._write(root / "_internal" / "torch_cpu.dll", b"same")
            self._write(root / "_internal" / "torch" / "lib" / "torch_cpu.dll", b"same")

            removed = prune_duplicate_torch_dlls(root)

            self.assertEqual(removed, ["torch_cpu.dll"])

    def test_different_torch_root_dll_is_not_pruned(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._write(root / "_internal" / "torch_cpu.dll", b"root")
            self._write(root / "_internal" / "torch" / "lib" / "torch_cpu.dll", b"canonical")

            removed = prune_duplicate_torch_dlls(root)

            self.assertEqual(removed, [])
            self.assertTrue((root / "_internal" / "torch_cpu.dll").is_file())

    def test_runtime_rejects_user_data_models_and_pyqt(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            self._make_runtime(root, "cpu")
            self._write(root / "dataset" / "images" / "sample.jpg")
            self._write(root / "weights.pt")
            self._write(root / "_internal" / "PyQt5" / "Qt5Core.dll")

            errors = check_runtime_manifest(root, "cpu")

        self.assertTrue(any("用户运行目录" in item for item in errors))
        self.assertTrue(any("模型文件" in item for item in errors))
        self.assertTrue(any("PyQt 主界面依赖" in item for item in errors))

    def test_worker_spec_excludes_gui_and_selects_worker_entry(self):
        root = Path(__file__).resolve().parents[2]
        text = (root / "packaging" / "workbuddy_yolo_runtime.spec").read_text(
            encoding="utf-8"
        )

        self.assertIn('select_entry_scripts(analysis.scripts, "yolo_worker_entry")', text)
        self.assertIn('"PyQt5"', text)
        self.assertNotIn('[str(TOOLS / "yolo_tool_launcher.py")]', text)

    def test_dependency_locks_keep_base_cpu_and_gpu_isolated(self):
        root = Path(__file__).resolve().parents[2]
        base = (root / "requirements-base-windows.txt").read_text(encoding="utf-8")
        cpu = (root / "requirements-runtime-windows-cpu.txt").read_text(encoding="utf-8")
        gpu = (root / "requirements-release-windows-cu118.txt").read_text(encoding="utf-8")

        for package in ("torch", "torchvision", "ultralytics", "tensorboard"):
            self.assertNotIn(package, base.lower())
        self.assertIn("torch==2.7.1+cpu", cpu)
        self.assertIn("torchvision==0.22.1+cpu", cpu)
        self.assertNotIn("cu118", cpu)
        self.assertIn("torch==2.7.1+cu118", gpu)
        self.assertIn("torchvision==0.22.1+cu118", gpu)
        self.assertNotIn("PyQt5", gpu)

    def test_packaging_writes_downloadable_catalog_entry(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            runtime_dir = root / "runtime"
            output_dir = root / "output"
            static = self._make_runtime(runtime_dir, "cpu")

            archive, complete = package_runtime_directory(
                runtime_dir,
                static,
                output_dir,
                base_url="https://example.invalid/releases/download/runtime-r1",
            )
            catalog_path = write_runtime_catalog((complete,), output_dir / "runtime_catalog.json")
            loaded = RuntimeCatalog.from_file(catalog_path)

            self.assertTrue(archive.is_file())
            self.assertTrue(loaded.artifacts[0].is_downloadable)
            self.assertEqual(loaded.artifacts[0].archive_size, archive.stat().st_size)
            self.assertEqual(loaded.artifacts[0].sha256, complete.sha256)
            self.assertEqual(check_runtime_archive(archive, complete), [])
            with zipfile.ZipFile(archive) as package:
                self.assertTrue(package.infolist())
                self.assertTrue(
                    all(
                        item.compress_type == zipfile.ZIP_LZMA
                        for item in package.infolist()
                        if not item.is_dir()
                    )
                )

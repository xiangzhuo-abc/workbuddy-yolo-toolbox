from __future__ import annotations

import io
import threading
import zipfile
from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.core.model_registry import (
    DownloadableModel,
    MODEL_CATALOG,
    ModelDownloadCancelled,
    ModelDownloadError,
    download_model,
    get_model,
    installed_models,
    validate_model_file,
)


class _FakeResponse:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0
        self.headers = {"Content-Length": str(len(data))}
        self.closed = False

    def read(self, size=-1):
        if self.offset >= len(self.data):
            return b""
        end = len(self.data) if size < 0 else min(len(self.data), self.offset + size)
        chunk = self.data[self.offset:end]
        self.offset = end
        return chunk

    def close(self):
        self.closed = True


def _zip_bytes(size: int) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("weights.pt", b"x" * max(1, size - 140))
    data = stream.getvalue()
    if len(data) > size:
        raise ValueError("测试权重目标大小过小")
    return data + (b"\0" * (size - len(data)))


class ModelRegistryTests(TestCase):
    def test_catalog_contains_only_detection_models(self):
        self.assertEqual(len(MODEL_CATALOG), 10)
        self.assertEqual({model.series for model in MODEL_CATALOG}, {"YOLOv8", "YOLO11"})
        self.assertEqual({model.size for model in MODEL_CATALOG}, {"n", "s", "m", "l", "x"})
        for model in MODEL_CATALOG:
            self.assertTrue(model.url.startswith("https://github.com/ultralytics/assets/"))
            self.assertTrue(model.filename.endswith(".pt"))
            self.assertNotIn("-seg", model.filename)
            self.assertNotIn("-pose", model.filename)
            self.assertNotIn("-cls", model.filename)

    def test_get_model_rejects_unknown_path_and_url(self):
        self.assertEqual(get_model("yolov8n.pt").filename, "yolov8n.pt")
        for value in ("../yolov8n.pt", "https://example.com/yolov8n.pt", "unknown.pt"):
            with self.assertRaises(ModelDownloadError):
                get_model(value)

        forged = DownloadableModel(
            series="YOLOv8",
            size="n",
            filename="yolov8n.pt",
            expected_size=get_model("yolov8n.pt").expected_size,
            url="https://example.com/yolov8n.pt",
        )
        with TemporaryDirectory() as temp_name, self.assertRaises(ModelDownloadError):
            download_model(forged, Path(temp_name))

    def test_validate_model_file_checks_expected_size_and_zip(self):
        model = get_model("yolov8n.pt")
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / model.filename
            path.write_bytes(b"not a model")
            self.assertFalse(validate_model_file(path, model))
            valid = Path(temp_name) / "valid.pt"
            valid.write_bytes(_zip_bytes(200))
            self.assertTrue(validate_model_file(valid))

    def test_download_is_atomic_and_existing_valid_file_is_reused(self):
        model = get_model("yolov8n.pt")
        data = _zip_bytes(model.expected_size)
        with TemporaryDirectory() as temp_name:
            output = Path(temp_name)
            result = download_model(
                model,
                output,
                opener=lambda request, timeout: _FakeResponse(data),
            )
            self.assertTrue(result.downloaded)
            self.assertTrue(validate_model_file(result.path, model))
            self.assertEqual(result.path.stat().st_size, model.expected_size)
            self.assertEqual(list(output.glob("*.part")), [])
            reused = download_model(
                model,
                output,
                opener=lambda request, timeout: self.fail("不应重复下载"),
            )
            self.assertFalse(reused.downloaded)
            self.assertEqual(installed_models(output)[model.filename], True)

    def test_cancel_cleans_partial_file(self):
        model = get_model("yolov8n.pt")
        cancel = threading.Event()
        cancel.set()
        with TemporaryDirectory() as temp_name:
            with self.assertRaises(ModelDownloadCancelled):
                download_model(
                    model,
                    Path(temp_name),
                    cancel_event=cancel,
                    opener=lambda request, timeout: _FakeResponse(_zip_bytes(model.expected_size)),
                )
            self.assertEqual(list(Path(temp_name).glob("*.part")), [])

    def test_cancel_before_request_does_not_open_network(self):
        model = get_model("yolo11n.pt")
        cancel = threading.Event()
        cancel.set()
        with TemporaryDirectory() as temp_name:
            with self.assertRaises(ModelDownloadCancelled):
                download_model(
                    model,
                    Path(temp_name),
                    cancel_event=cancel,
                    opener=lambda request, timeout: self.fail("取消后不应建立网络连接"),
                )

    def test_size_mismatch_is_rejected_without_writing_target(self):
        model = get_model("yolov8n.pt")
        with TemporaryDirectory() as temp_name:
            with self.assertRaises(ModelDownloadError):
                download_model(
                    model,
                    Path(temp_name),
                    opener=lambda request, timeout: _FakeResponse(b"short"),
                )
            self.assertFalse((Path(temp_name) / model.filename).exists())

    def test_insufficient_disk_space_fails_before_network_request(self):
        model = get_model("yolov8n.pt")
        usage = namedtuple("usage", "total used free")(100, 99, 1)
        with TemporaryDirectory() as temp_name:
            from unittest.mock import patch

            with patch("tools.core.model_registry.shutil.disk_usage", return_value=usage):
                with self.assertRaises(ModelDownloadError):
                    download_model(
                        model,
                        Path(temp_name),
                        opener=lambda request, timeout: self.fail("空间不足时不应联网"),
                    )

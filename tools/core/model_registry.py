"""官方目标检测模型清单、校验和原子下载服务。"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


OFFICIAL_ASSET_BASE = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
)
MODEL_SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class DownloadableModel:
    """可下载的官方目标检测权重。"""

    series: str
    size: str
    filename: str
    expected_size: int
    url: str

    @property
    def label(self) -> str:
        return f"{self.series} {self.size}"

    @property
    def task(self) -> str:
        return "目标检测"


def _build_catalog() -> tuple[DownloadableModel, ...]:
    sizes = {
        "yolov8": {"n": 6549796, "s": 22588772, "m": 52136884, "l": 87792836, "x": 136890692},
        "yolo11": {"n": 5613764, "s": 19313732, "m": 40684120, "l": 51387343, "x": 114636239},
    }
    catalog = []
    for series in ("yolov8", "yolo11"):
        display_series = "YOLOv8" if series == "yolov8" else "YOLO11"
        for size in MODEL_SIZES:
            filename = f"{series}{size}.pt"
            catalog.append(
                DownloadableModel(
                    series=display_series,
                    size=size,
                    filename=filename,
                    expected_size=sizes[series][size],
                    url=OFFICIAL_ASSET_BASE + filename,
                )
            )
    return tuple(catalog)


MODEL_CATALOG = _build_catalog()
MODEL_BY_FILENAME = {model.filename: model for model in MODEL_CATALOG}


class ModelDownloadError(RuntimeError):
    """模型下载或校验失败。"""


class ModelDownloadCancelled(ModelDownloadError):
    """下载被用户取消。"""


@dataclass(frozen=True)
class ModelDownloadResult:
    model: DownloadableModel
    path: Path
    downloaded: bool
    size: int


def get_model(filename: str) -> DownloadableModel:
    """只从内置清单获取模型，拒绝任意 URL 或路径。"""
    key = Path(str(filename)).name
    model = MODEL_BY_FILENAME.get(key)
    if model is None or key != str(filename):
        raise ModelDownloadError(f"模型不在官方下载清单中: {filename}")
    return model


def _validate_model_payload(
    path: Path,
    model: DownloadableModel | None = None,
) -> bool:
    """校验权重内容，允许用于尚未改名的临时下载文件。"""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        if model is not None and path.stat().st_size != model.expected_size:
            return False
        with path.open("rb") as stream:
            if stream.read(2) != b"PK":
                return False
        return zipfile.is_zipfile(path)
    except (OSError, zipfile.BadZipFile):
        return False


def validate_model_file(path: Path, model: DownloadableModel | None = None) -> bool:
    """校验正式 .pt 文件，不在当前进程加载或执行权重。"""
    path = Path(path)
    return path.suffix.lower() == ".pt" and _validate_model_payload(path, model)


def _emit_progress(
    progress: Callable[[int, int], None] | None,
    downloaded: int,
    total: int,
) -> None:
    if progress is not None:
        progress(downloaded, total)


def download_model(
    model: DownloadableModel | str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> ModelDownloadResult:
    """从固定官方地址下载模型，并以原子替换方式写入 models 目录。"""
    if isinstance(model, str):
        model = get_model(model)
    if not isinstance(model, DownloadableModel):
        raise ModelDownloadError("无效的模型清单项")
    registered = get_model(model.filename)
    if model != registered:
        raise ModelDownloadError("模型清单项已被修改，拒绝下载")
    model = registered

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / model.filename
    if target.parent != output_dir:
        raise ModelDownloadError("模型目标路径不安全")
    if target.exists() and not overwrite:
        if validate_model_file(target, model):
            return ModelDownloadResult(model, target, False, target.stat().st_size)
        raise ModelDownloadError(
            f"模型文件已存在但校验失败，请选择覆盖下载: {target.name}"
        )
    required_space = model.expected_size + 64 * 1024 * 1024
    if shutil.disk_usage(output_dir).free < required_space:
        raise ModelDownloadError(
            f"模型目录剩余空间不足，至少需要 {required_space / 1024 / 1024:.0f} MB"
        )
    if cancel_event is not None and cancel_event.is_set():
        raise ModelDownloadCancelled("模型下载已取消")

    request = urllib.request.Request(
        model.url,
        headers={"User-Agent": "YOLO-tool-model-manager/1.0"},
    )
    temp_path: Path | None = None
    try:
        response = opener(request, timeout=60)
        total = int(response.headers.get("Content-Length") or model.expected_size)
        if total != model.expected_size:
            raise ModelDownloadError(
                f"官方文件大小发生变化: 预计 {model.expected_size}，实际 {total}"
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{model.filename}.",
            suffix=".part",
            dir=str(output_dir),
        )
        temp_path = Path(temp_name)
        downloaded = 0
        with os.fdopen(descriptor, "wb") as stream:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise ModelDownloadCancelled("模型下载已取消")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > model.expected_size:
                    raise ModelDownloadError("下载内容超过官方文件大小")
                stream.write(chunk)
                _emit_progress(progress, downloaded, total)
            stream.flush()
            os.fsync(stream.fileno())
        if downloaded != model.expected_size or not _validate_model_payload(temp_path, model):
            raise ModelDownloadError("下载完成但模型文件校验失败")
        temp_path.replace(target)
        temp_path = None
        return ModelDownloadResult(model, target, True, downloaded)
    except ModelDownloadError:
        raise
    except Exception as exc:
        raise ModelDownloadError(f"下载模型失败: {exc}") from exc
    finally:
        try:
            response.close()
        except (NameError, AttributeError, OSError):
            pass
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def installed_models(models_dir: Path) -> Mapping[str, bool]:
    """返回内置清单中每个模型是否已经存在且通过基础校验。"""
    models_dir = Path(models_dir)
    return {
        model.filename: validate_model_file(models_dir / model.filename, model)
        for model in MODEL_CATALOG
    }

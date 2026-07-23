"""
YOLO 检测子进程。

这个脚本刻意不导入 PyQt：主界面只启动它并读取 JSON 结果，避免
Qt 进程中执行 ultralytics/torch 推理时 native 崩溃带关整个工具。
"""

import argparse
import json
import sys
import traceback
from contextlib import nullcontext
from multiprocessing import freeze_support
from pathlib import Path
from uuid import uuid4

from core.task_protocol import TaskEventEmitter

import cv2
import numpy as np

try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

READY_PREFIX = "__YOLO_DETECT_READY__"
RESULT_PREFIX = "__YOLO_DETECT_RESULT__"


def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def _to_scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return value.item()
    return value


def _to_xyxy(value):
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = arr.reshape(-1, 4)[0]
    return [int(round(float(v))) for v in arr]


def _normalise_names(names):
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {str(i): str(v) for i, v in enumerate(names)}
    return {}


def load_model(model_path):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = _normalise_names(getattr(model, "names", {}))
    return model, names


def predict_with_model(model, names, image_path, conf):

    try:
        import torch
        inference_context = torch.inference_mode()
    except Exception:
        inference_context = nullcontext()

    image = cv2_imread(image_path)
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    with inference_context:
        results = model.predict(image, conf=conf, verbose=False)

    detections = []
    if results:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            for box in boxes:
                cls_id = int(_to_scalar(box.cls[0]))
                score = float(_to_scalar(box.conf[0]))
                xyxy = _to_xyxy(box.xyxy)
                detections.append({
                    "class_id": cls_id,
                    "name": names.get(str(cls_id), str(cls_id)),
                    "conf": score,
                    "xyxy": xyxy,
                })

    return {
        "ok": True,
        "names": names,
        "detections": detections,
    }


def run_detect(model_path, image_path, conf):
    model, names = load_model(model_path)
    return predict_with_model(model, names, image_path, conf)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_event(prefix, payload):
    sys.__stdout__.write(prefix + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def serve(
    model_path,
    task_id=None,
    load_model_func=None,
    predict_func=None,
    stream=None,
    stdin=None,
):
    task_id = task_id or str(uuid4())
    emitter = TaskEventEmitter(task_id, "detection", stream=stream)
    load_model_func = load_model_func or load_model
    predict_func = predict_func or predict_with_model
    stdin = stdin if stdin is not None else sys.stdin
    emitter.started("检测 worker 正在加载模型", {"model": str(model_path)})
    try:
        model, names = load_model_func(model_path)
        emitter.progress(0.2, "模型已加载")
        write_event(READY_PREFIX, {"ok": True, "names": names})
    except Exception as exc:
        emitter.failed(
            f"模型加载失败: {exc}",
            {"traceback": traceback.format_exc()},
        )
        write_event(READY_PREFIX, {
            "ok": False, "error": str(exc), "traceback": traceback.format_exc()
        })
        return 1

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        request = {}
        try:
            request = json.loads(line)
            if request.get("cmd") == "quit":
                emitter.cancelled("检测 worker 已退出")
                return 0
            if request.get("cmd") != "detect":
                raise ValueError(f"未知命令: {request.get('cmd')}")
            request_task_id = request.get("task_id") or str(uuid4())
            request_emitter = TaskEventEmitter(
                request_task_id,
                "detection_request",
                stream=stream,
            )
            request_emitter.started(
                "检测请求开始",
                {"request_id": request.get("id"), "image": request.get("image")},
            )
            payload = predict_func(
                model,
                names,
                Path(request["image"]),
                float(request.get("conf", 0.5)),
            )
            payload["id"] = request.get("id")
            request_emitter.result("检测完成", payload)
            write_event(RESULT_PREFIX, payload)
        except Exception as exc:
            request_task_id = request.get("task_id") or str(uuid4())
            request_emitter = TaskEventEmitter(
                request_task_id,
                "detection_request",
                stream=stream,
            )
            request_emitter.failed(
                f"检测失败: {exc}",
                {"traceback": traceback.format_exc()},
            )
            write_event(RESULT_PREFIX, {
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
    emitter.cancelled("检测 worker 输入已关闭")
    return 0


def main(
    argv=None,
    load_model_func=None,
    predict_func=None,
    stream=None,
    stdin=None,
):
    parser = argparse.ArgumentParser(description="YOLO 检测子进程")
    parser.add_argument("--task-id", default=None, help="统一任务 ID")
    parser.add_argument("--model", required=True, help="模型 .pt 文件")
    parser.add_argument("--image", help="待检测图片")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--output", help="JSON 输出路径")
    parser.add_argument("--serve", action="store_true", help="常驻模式：加载一次模型后从 stdin 接收检测请求")
    args = parser.parse_args(argv)

    if args.serve:
        return serve(
            Path(args.model),
            task_id=args.task_id,
            load_model_func=load_model_func,
            predict_func=predict_func,
            stream=stream,
            stdin=stdin,
        )
    if not args.image or not args.output:
        parser.error("非常驻模式需要 --image 和 --output")

    try:
        model_loader = load_model_func or load_model
        predictor = predict_func or predict_with_model
        model, names = model_loader(Path(args.model))
        payload = predictor(model, names, Path(args.image), args.conf)
        write_json(args.output, payload)
        emitter = TaskEventEmitter(args.task_id or str(uuid4()), "detection", stream=stream)
        emitter.started("单图检测开始")
        emitter.result("检测完成", payload)
        return 0
    except Exception as exc:
        write_json(args.output, {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        emitter = TaskEventEmitter(args.task_id or str(uuid4()), "detection", stream=stream)
        emitter.started("单图检测开始")
        emitter.failed(f"检测失败: {exc}", {"traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

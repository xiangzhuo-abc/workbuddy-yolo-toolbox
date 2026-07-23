"""在独立进程中运行 YOLO 训练，并输出统一任务事件。"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from multiprocessing import freeze_support
from pathlib import Path
from uuid import uuid4

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import yolo_dataset_tools as tools  # noqa: E402
from core.runtime_paths import RuntimePaths  # noqa: E402
from core.task_protocol import TaskEventEmitter  # noqa: E402


EPOCH_PATTERN = re.compile(r"(?:Epoch\s+)?(\d+)\s*/\s*(\d+)", re.IGNORECASE)
RUNTIME_PATHS = RuntimePaths.from_environment()
PROJECT_DIR = RUNTIME_PATHS.resource_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 训练子进程")
    parser.add_argument("--task-id", default=None, help="统一任务 ID")
    parser.add_argument("--model", required=True, help="预训练模型 .pt 路径")
    parser.add_argument("--data", required=True, help="数据集 data.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--device", default="", help="训练设备，留空时由 Ultralytics 自动选择")
    parser.add_argument(
        "--project",
        default=str(RUNTIME_PATHS.runs_dir),
        help="训练结果目录",
    )
    parser.add_argument("--name", default="coc_detect", help="训练名称")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--resume", action="store_true", help="继续上次训练")
    return parser


def main(argv=None, train_func=None, stream=None) -> int:
    args = _build_parser().parse_args(argv)
    task_id = args.task_id or str(uuid4())
    emitter = TaskEventEmitter(task_id, "training", stream=stream)
    uses_default_backend = train_func is None
    train_func = train_func or tools.train_model

    project = Path(args.project)
    if not project.is_absolute():
        project = RUNTIME_PATHS.workspace_dir / project

    emitter.started(
        "训练任务已启动",
        {
            "model": str(args.model),
            "data": str(args.data),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "project": str(project),
            "name": args.name,
        },
    )

    def emit(level, message):
        text = str(message or "")
        emitter.log(text, level=str(level or "info"))
        match = EPOCH_PATTERN.search(text)
        if match:
            current, total = (int(value) for value in match.groups())
            if total == args.epochs and total > 0:
                emitter.progress(
                    min(1.0, current / total),
                    f"Epoch {current}/{total}",
                    {"epoch": current, "total_epochs": total},
                )

    try:
        if uses_default_backend:
            tools.setup_logging(
                log_file=str(RUNTIME_PATHS.logs_dir / "yolo_tool.log"),
                console=False,
            )
        success = train_func(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=str(project),
            name=args.name,
            patience=args.patience,
            resume=args.resume,
            emit=emit,
        )
    except KeyboardInterrupt:
        emitter.cancelled("训练任务已取消")
        return 130
    except BaseException as exc:
        emitter.failed(
            f"训练任务异常: {exc}",
            {"traceback": traceback.format_exc()},
        )
        return 1

    if success:
        emitter.result(
            "训练完成",
            {"project": str(project), "name": args.name},
        )
        return 0
    emitter.failed("训练失败，请查看任务日志")
    return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

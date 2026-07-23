"""兼容命令行入口：将截图导入 YOLO 数据集。

实际逻辑统一委托给 yolo_dataset_tools.prepare_dataset，避免绕过 GUI 的目录规范化、
data.yaml 同步和清空前自动备份。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yolo_dataset_tools as tools


def _print_emit(level, msg):
    prefix = {
        "debug": "[调试]",
        "info": "[信息]",
        "warning": "[警告]",
        "error": "[错误]",
    }.get(level, "[信息]")
    print(f"{prefix} {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="将截图复制到 YOLO 数据集目录")
    parser.add_argument("--source", required=True, help="截图文件或目录")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="目标子集")
    parser.add_argument("--clean", action="store_true", help="先自动备份，再清空目标目录并复制")
    parser.add_argument("--convert-format", choices=tools.TARGET_FORMATS, default=None, help="导入时转换图片格式")
    parser.add_argument("--dataset-dir", default=None, help="数据集目录，默认使用项目 dataset/")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else tools.get_dataset_dir()
    tools.setup_logging(console=False)
    ok = tools.prepare_dataset(
        args.source,
        split=args.split,
        clean=args.clean,
        convert_format=args.convert_format,
        dataset_dir=dataset_dir,
        emit=_print_emit,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

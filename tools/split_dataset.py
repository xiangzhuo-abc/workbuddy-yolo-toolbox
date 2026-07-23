"""类别感知 YOLO 数据集划分命令行入口。"""

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="预览或执行类别感知数据集划分")
    parser.add_argument("--train", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.05, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--mode",
        choices=("repair", "full"),
        default="repair",
        help="repair 最小修复，full 智能重划分",
    )
    parser.add_argument(
        "--min-train-images",
        type=int,
        default=5,
        help="每个类别优先保留的训练图片数",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行方案；省略时只读预览",
    )
    parser.add_argument("--dataset-dir", default=None, help="数据集目录，默认使用项目 dataset/")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else tools.get_dataset_dir()
    tools.setup_logging(console=False)
    try:
        plan = tools.build_split_plan(
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
            mode=args.mode,
            min_train_images=args.min_train_images,
            dataset_dir=dataset_dir,
            emit=_print_emit,
        )
        if not plan.is_executable:
            _print_emit("error", "计划包含阻断问题，未执行。")
            return 1
        if not args.apply:
            _print_emit("info", "当前为只读预览；添加 --apply 才会执行。")
            return 0
        result = tools.apply_split_plan(plan, emit=_print_emit)
        return 0 if result.success else 1
    except Exception as exc:
        _print_emit("error", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

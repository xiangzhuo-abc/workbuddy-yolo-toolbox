"""
YOLO 训练脚本

用法：
    python tools/train_yolo.py --data dataset/data.yaml --epochs 100 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv8 目标检测模型")
    parser.add_argument("--data", default="dataset/data.yaml", help="数据集配置文件路径")
    parser.add_argument("--model", default="models/yolov8n.pt", help="预训练模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="runs", help="训练结果保存目录")
    parser.add_argument("--name", default="coc_detect", help="本次训练名称")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        save=True,
        verbose=True,
    )

    print(f"训练完成。最佳模型：{args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

"""
YOLO 推理脚本

用法：
    python tools/detect_yolo.py --weights runs/coc_detect/weights/best.pt --source coc_screenshot.png

输出：
    在图片上绘制检测框，并打印每个目标的类别和坐标。
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 推理")
    parser.add_argument("--weights", required=True, help="训练好的模型路径")
    parser.add_argument("--source", required=True, help="图片路径或目录")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--save", action="store_true", help="是否保存结果图")
    parser.add_argument("--output", default="debug/detect", help="结果保存目录")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(args.source, conf=args.conf, verbose=False)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        names = result.names
        print(f"\n图片: {result.path}")
        print("检测结果:")

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = names.get(cls_id, str(cls_id))
            print(f"  {name:20s} 置信度={conf:.3f} 位置=({x1}, {y1}, {x2}, {y2})")

        if args.save:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / Path(result.path).name
            result.save(filename=str(output_path))
            print(f"结果图已保存: {output_path}")


if __name__ == "__main__":
    main()

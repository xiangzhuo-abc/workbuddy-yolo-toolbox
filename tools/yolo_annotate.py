"""
YOLO 数据集标注工具

用法：
    python tools/yolo_annotate.py --images dataset/images/train --labels dataset/labels/train

操作：
- 鼠标左键拖动：框选目标
- 输入标签名：按回车确认（首次使用的新标签会自动加入类别）
- 右键点击：删除最近的一个框
- s 键：保存当前图片标注
- n / 空格键：下一张
- p 键：上一张
- q / Esc 键：退出

每个图片需要同名 .txt 标签文件，格式为 YOLO：
    class_id x_center y_center width height
（所有数值均已归一化到 [0, 1]）
"""

import argparse
import cv2
from pathlib import Path

from core.annotation_service import (
    load_classes_file,
    load_pixel_boxes,
    save_classes_file_atomic,
    save_pixel_boxes_atomic,
)


class YoloAnnotator:
    def __init__(self, image_dir: str, label_dir: str, classes_file: str = "dataset/classes.txt"):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.classes_file = Path(classes_file)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.classes_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载或创建类别列表
        self.class_names = self._load_classes()
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        # 加载图片列表
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ])
        if not self.image_paths:
            raise RuntimeError(f"未在 {image_dir} 找到图片")

        self.current_idx = 0
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.current_box = None
        self.boxes = []  # [(x1, y1, x2, y2, class_id)]
        self.window_name = "YOLO Annotator - 左键框选 | 右键删除 | s保存 | n下一张"

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _load_classes(self) -> list:
        names, issues = load_classes_file(self.classes_file)
        for issue in issues:
            print(f"类别文件读取失败: {issue.message}")
        return list(names)

    def _save_classes(self):
        saved_names = save_classes_file_atomic(self.classes_file, self.class_names)
        self.class_names = list(saved_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

    def _get_or_create_class_id(self, name: str) -> int:
        if name not in self.class_to_id:
            new_names = self.class_names + [name]
            saved_names = save_classes_file_atomic(self.classes_file, new_names)
            self.class_names = list(saved_names)
            self.class_to_id = {item: i for i, item in enumerate(self.class_names)}
        return self.class_to_id[name]

    def _load_labels(self, image_path: Path):
        self.boxes = []
        label_path = self.label_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            return

        img = cv2.imread(str(image_path))
        if img is None:
            return
        h, w = img.shape[:2]

        boxes, issues = load_pixel_boxes(
            label_path,
            image_size=(w, h),
            class_count=len(self.class_names) if self.class_names else None,
        )
        self.boxes = list(boxes)
        if issues:
            print(f"标签存在 {len(issues)} 处问题，已加载有效框: {label_path}")

    def _save_labels(self):
        image_path = self.image_paths[self.current_idx]
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"无法读取图片: {image_path}")
            return False
        h, w = img.shape[:2]

        label_path = self.label_dir / (image_path.stem + ".txt")
        try:
            saved_boxes = save_pixel_boxes_atomic(
                label_path,
                self.boxes,
                image_size=(w, h),
            )
        except Exception as exc:
            print(f"保存失败，原标签未修改: {exc}")
            return False

        self.boxes = list(saved_boxes)
        print(f"已保存: {label_path}")
        return True

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_box = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_box = (self.start_x, self.start_y, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1, x2, y2 = self.current_box
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                label = input(f"\n输入标签名（当前类别: {self.class_names}）：").strip()
                if label:
                    class_id = self._get_or_create_class_id(label)
                    self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), class_id))
                    print(f"添加框: {label} ({self.start_x}, {self.start_y}) -> ({x2}, {y2})")
            self.current_box = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.boxes:
                removed = self.boxes.pop()
                print(f"删除框: {self.class_names[removed[4]]}")

    def _draw(self):
        image_path = self.image_paths[self.current_idx]
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"无法读取图片: {image_path}")
            return

        # 绘制已保存的框
        for x1, y1, x2, y2, class_id in self.boxes:
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制当前拖动框
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 顶部信息
        info = f"{self.current_idx + 1}/{len(self.image_paths)} | {image_path.name} | 类别: {self.class_names}"
        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.window_name, img)

    def run(self):
        self._load_labels(self.image_paths[self.current_idx])
        print(f"共 {len(self.image_paths)} 张图片")
        print("操作：左键框选 | 右键删除 | s保存 | n下一张 | p上一张 | q退出")

        while True:
            self._draw()
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # q 或 Esc
                break
            elif key == ord('s'):
                self._save_labels()
            elif key == ord('n') or key == ord(' '):
                if not self._save_labels():
                    continue
                self.current_idx = min(self.current_idx + 1, len(self.image_paths) - 1)
                self.boxes = []
                self._load_labels(self.image_paths[self.current_idx])
                print(f"\n切换到第 {self.current_idx + 1} 张")
            elif key == ord('p'):
                if not self._save_labels():
                    continue
                self.current_idx = max(self.current_idx - 1, 0)
                self.boxes = []
                self._load_labels(self.image_paths[self.current_idx])
                print(f"\n切换到第 {self.current_idx + 1} 张")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 数据集标注工具")
    parser.add_argument("--images", default="dataset/images/train", help="图片目录")
    parser.add_argument("--labels", default="dataset/labels/train", help="标签输出目录")
    parser.add_argument("--classes", default="dataset/classes.txt", help="类别文件")
    args = parser.parse_args()

    annotator = YoloAnnotator(args.images, args.labels, args.classes)
    annotator.run()

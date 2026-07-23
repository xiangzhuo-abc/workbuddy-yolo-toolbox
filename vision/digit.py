"""
数字识别器

基于模板匹配逐位识别游戏 UI 中的数字。
比通用 OCR 更适合固定字体、带样式的游戏数字。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
from vision.template import TemplateMatcher


@dataclass
class DigitResult:
    """单个数字识别结果"""
    digit: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


class DigitRecognizer:
    """基于模板匹配的数字识别器"""

    def __init__(self, template_dir: str, threshold: float = 0.75):
        """
        Args:
            template_dir: 数字模板目录，包含 0.png ~ 9.png
            threshold: 匹配阈值
        """
        self.template_dir = Path(template_dir)
        self.threshold = threshold
        self.digits = self._load_digit_templates()
        self.matcher = TemplateMatcher(threshold=threshold)

    def _load_digit_templates(self) -> Dict[str, np.ndarray]:
        """加载 0-9 数字模板"""
        digits = {}
        for i in range(10):
            path = self.template_dir / f"{i}.png"
            if path.exists():
                # 使用 Pillow 读取，绕过 OpenCV Windows 读取问题
                pil_img = Image.open(str(path)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                if img is not None:
                    digits[str(i)] = img
        return digits

    def recognize(self, image: np.ndarray,
                  search_direction: str = "left_to_right") -> Tuple[str, List[DigitResult]]:
        """
        在图像中识别所有数字

        Args:
            image: 包含数字的图像区域
            search_direction: 搜索方向，left_to_right 或 right_to_left

        Returns:
            (识别出的数字字符串, 每个数字的详细结果)
        """
        results = []

        for digit, template in self.digits.items():
            matches = self.matcher.find_all(image, template, threshold=self.threshold, max_results=20)
            for match in matches:
                results.append(DigitResult(
                    digit=digit,
                    confidence=match.confidence,
                    x=match.x,
                    y=match.y,
                    width=match.width,
                    height=match.height
                ))

        # 按 x 坐标排序，从左到右拼接数字
        reverse = search_direction == "right_to_left"
        results.sort(key=lambda r: r.x, reverse=reverse)

        # 去重：如果两个数字重叠，保留置信度高的
        filtered = []
        for r in results:
            overlap = False
            for existing in filtered:
                if abs(r.x - existing.x) < r.width * 0.5:
                    overlap = True
                    break
            if not overlap:
                filtered.append(r)

        text = "".join([r.digit for r in filtered])
        return text, filtered

    def recognize_number(self, image: np.ndarray) -> Optional[int]:
        """识别数字并返回整数"""
        text, _ = self.recognize(image)
        digits_only = "".join(c for c in text if c.isdigit())
        if not digits_only:
            return None
        try:
            return int(digits_only)
        except ValueError:
            return None

    @staticmethod
    def extract_digit_templates(source_image: np.ndarray,
                                digit_regions: Dict[str, Tuple[int, int, int, int]],
                                output_dir: str):
        """
        从源图像中提取数字模板并保存

        Args:
            source_image: 源图像
            digit_regions: {数字: (x, y, w, h)}
            output_dir: 输出目录
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for digit, (x, y, w, h) in digit_regions.items():
            roi = source_image[y:y+h, x:x+w]
            save_path = out / f"{digit}.png"
            cv2.imwrite(str(save_path), roi)
            print(f"已保存数字 {digit} 模板: {save_path}")

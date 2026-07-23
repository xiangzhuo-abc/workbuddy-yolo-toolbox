"""
颜色检测模块

用于检测血条、进度条、特定颜色 UI 元素等。
"""

import numpy as np
import cv2
from typing import Tuple


class ColorDetector:
    """颜色/进度检测器"""

    @staticmethod
    def detect_color_region(image: np.ndarray,
                            lower_hsv: Tuple[int, int, int],
                            upper_hsv: Tuple[int, int, int]) -> np.ndarray:
        """
        检测指定 HSV 颜色范围的区域

        Args:
            image: BGR 图像
            lower_hsv: HSV 下限
            upper_hsv: HSV 上限

        Returns:
            二值掩码图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        return mask

    @staticmethod
    def count_pixels(image: np.ndarray,
                     lower_hsv: Tuple[int, int, int],
                     upper_hsv: Tuple[int, int, int]) -> int:
        """
        统计指定颜色范围内的像素数量

        Args:
            image: BGR 图像
            lower_hsv: HSV 下限
            upper_hsv: HSV 上限

        Returns:
            像素数量
        """
        mask = ColorDetector.detect_color_region(image, lower_hsv, upper_hsv)
        return cv2.countNonZero(mask)

    @staticmethod
    def fill_ratio(image: np.ndarray,
                   lower_hsv: Tuple[int, int, int],
                   upper_hsv: Tuple[int, int, int]) -> float:
        """
        计算指定颜色在图像中的填充比例

        Args:
            image: BGR 图像
            lower_hsv: HSV 下限
            upper_hsv: HSV 上限

        Returns:
            填充比例 [0, 1]
        """
        mask = ColorDetector.detect_color_region(image, lower_hsv, upper_hsv)
        total = image.shape[0] * image.shape[1]
        filled = cv2.countNonZero(mask)
        return filled / total if total > 0 else 0.0

    @staticmethod
    def find_color_centroid(image: np.ndarray,
                            lower_hsv: Tuple[int, int, int],
                            upper_hsv: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        找到指定颜色区域的重心坐标

        Args:
            image: BGR 图像
            lower_hsv: HSV 下限
            upper_hsv: HSV 上限

        Returns:
            (cx, cy) 重心坐标，未找到返回 (-1, -1)
        """
        mask = ColorDetector.detect_color_region(image, lower_hsv, upper_hsv)
        moments = cv2.moments(mask)

        if moments["m00"] == 0:
            return (-1, -1)

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

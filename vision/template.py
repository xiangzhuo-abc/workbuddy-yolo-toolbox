"""
模板匹配模块

用于在屏幕截图中查找指定的 UI 元素（按钮、图标等）。
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MatchResult:
    """模板匹配结果"""
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        """返回模板中心点坐标"""
        return (self.x + self.width // 2, self.y + self.height // 2)


class TemplateMatcher:
    """基于 OpenCV 的模板匹配器"""

    def __init__(self, threshold: float = 0.85):
        """
        Args:
            threshold: 默认匹配阈值，范围 [0, 1]
        """
        self.threshold = threshold

    def find(self, screen: np.ndarray, template: np.ndarray,
             threshold: float = None) -> Optional[MatchResult]:
        """
        在屏幕中查找单个模板

        Args:
            screen: 屏幕截图，BGR 格式
            template: 模板图像，BGR 格式
            threshold: 匹配阈值，None 时使用默认值

        Returns:
            MatchResult or None: 最佳匹配结果，未找到返回 None
        """
        if threshold is None:
            threshold = self.threshold

        if template.shape[0] > screen.shape[0] or template.shape[1] > screen.shape[1]:
            raise ValueError("模板尺寸不能大于屏幕尺寸")

        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold:
            return None

        h, w = template.shape[:2]
        return MatchResult(
            x=max_loc[0],
            y=max_loc[1],
            width=w,
            height=h,
            confidence=max_val
        )

    def find_all(self, screen: np.ndarray, template: np.ndarray,
                 threshold: float = None, max_results: int = 10) -> List[MatchResult]:
        """
        在屏幕中查找所有匹配位置（支持多个相同图标）

        Args:
            screen: 屏幕截图
            template: 模板图像
            threshold: 匹配阈值
            max_results: 最大返回结果数

        Returns:
            List[MatchResult]: 匹配结果列表
        """
        if threshold is None:
            threshold = self.threshold

        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        h, w = template.shape[:2]

        # 获取所有大于阈值的点
        locations = np.where(result >= threshold)
        points = list(zip(*locations[::-1]))

        # NMS（非极大值抑制）去除重叠框
        matches = []
        for pt in points:
            matches.append((pt[0], pt[1], pt[0] + w, pt[1] + h, result[pt[1], pt[0]]))

        matches = self._nms(matches, overlap_thresh=0.3)

        results = []
        for x1, y1, x2, y2, conf in matches[:max_results]:
            results.append(MatchResult(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
                confidence=conf
            ))

        return results

    def _nms(self, boxes: List[tuple], overlap_thresh: float = 0.3) -> List[tuple]:
        """简单 NMS 去重"""
        if not boxes:
            return []

        # 按置信度排序
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        picked = []

        while boxes:
            current = boxes[0]
            picked.append(current)
            boxes = boxes[1:]

            remaining = []
            for box in boxes:
                iou = self._iou(current[:4], box[:4])
                if iou < overlap_thresh:
                    remaining.append(box)
            boxes = remaining

        return picked

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """计算两个框的 IoU"""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter_area / float(box_a_area + box_b_area - inter_area + 1e-5)

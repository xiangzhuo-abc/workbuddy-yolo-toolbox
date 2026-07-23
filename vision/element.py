"""
元素检测器

识别页面中的具体 UI 元素：按钮、图标、资源条等。
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image
from vision.template import TemplateMatcher, MatchResult


@dataclass
class ElementResult:
    """元素检测结果"""
    name: str
    found: bool
    matches: List[MatchResult]


class ElementDetector:
    """UI 元素检测器"""

    def __init__(self, config_path: str):
        self.elements = self._load_config(config_path)
        self.matcher = TemplateMatcher()

    def _load_config(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def detect(self, screen: np.ndarray, element_name: str,
               threshold: float = None, region: List[int] = None) -> ElementResult:
        """检测单个元素"""
        config = self.elements.get(element_name)
        if config is None:
            raise ValueError(f"未定义元素: {element_name}")

        template_path = config["template"]
        if threshold is None:
            threshold = config.get("threshold", 0.85)
        if region is None:
            region = config.get("region")

        # 使用 Pillow 读取模板
        pil_img = Image.open(template_path).convert("RGB")
        template = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if template is None:
            raise RuntimeError(f"无法加载模板: {template_path}")

        search_screen = screen
        offset_x, offset_y = 0, 0
        if region is not None:
            x, y, w, h = region
            search_screen = screen[y:y+h, x:x+w]
            offset_x, offset_y = x, y

        matches = self.matcher.find_all(search_screen, template, threshold=threshold)

        for match in matches:
            match.x += offset_x
            match.y += offset_y

        return ElementResult(
            name=element_name,
            found=len(matches) > 0,
            matches=matches
        )

    def detect_all(self, screen: np.ndarray, element_names: List[str] = None) -> Dict[str, ElementResult]:
        """检测多个元素，默认检测全部"""
        if element_names is None:
            element_names = list(self.elements.keys())

        results = {}
        for name in element_names:
            results[name] = self.detect(screen, name)
        return results

    def get_center(self, element_result: ElementResult) -> Optional[tuple]:
        """获取第一个匹配位置的中心点"""
        if element_result.found and element_result.matches:
            return element_result.matches[0].center
        return None

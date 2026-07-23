"""
特征检测器

提供多种低层图像特征检测能力，供页面识别和元素检测使用。
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from PIL import Image
from vision.template import TemplateMatcher, MatchResult
from vision.color import ColorDetector
from vision.ocr import OcrEngine


@dataclass
class FeatureResult:
    """特征检测结果"""
    matched: bool          # 是否匹配
    confidence: float      # 置信度 [0, 1]
    detail: Any            # 附加信息（位置、文字等）


class Feature(ABC):
    """特征检测基类"""

    @abstractmethod
    def detect(self, screen: np.ndarray, config: Dict[str, Any]) -> FeatureResult:
        """在屏幕中检测该特征"""
        pass


class TemplateFeature(Feature):
    """模板匹配特征"""

    def detect(self, screen: np.ndarray, config: Dict[str, Any]) -> FeatureResult:
        template_path = config["template"]
        threshold = config.get("threshold", 0.85)
        region = config.get("region")  # 可选 [x, y, w, h]

        # 使用 Pillow 读取模板，绕过 OpenCV Windows 读取问题
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

        matcher = TemplateMatcher(threshold=threshold)
        match = matcher.find(search_screen, template)

        if match is None:
            return FeatureResult(matched=False, confidence=0.0, detail=None)

        match.x += offset_x
        match.y += offset_y
        return FeatureResult(matched=True, confidence=match.confidence, detail=match)


class ColorFeature(Feature):
    """颜色区域特征"""

    def detect(self, screen: np.ndarray, config: Dict[str, Any]) -> FeatureResult:
        region = config.get("region")
        lower_hsv = config["lower_hsv"]
        upper_hsv = config["upper_hsv"]
        min_ratio = config.get("min_ratio", 0.1)
        max_ratio = config.get("max_ratio", 1.0)

        image = screen
        if region is not None:
            x, y, w, h = region
            image = screen[y:y+h, x:x+w]

        ratio = ColorDetector.fill_ratio(image, tuple(lower_hsv), tuple(upper_hsv))
        matched = min_ratio <= ratio <= max_ratio
        confidence = 1.0 - abs(ratio - (min_ratio + max_ratio) / 2) / max_ratio
        confidence = max(0.0, min(1.0, confidence))

        return FeatureResult(matched=matched, confidence=confidence, detail={"ratio": ratio})


class OcrFeature(Feature):
    """文字识别特征"""

    def __init__(self, ocr_engine: OcrEngine = None):
        self.ocr = ocr_engine or OcrEngine()

    def detect(self, screen: np.ndarray, config: Dict[str, Any]) -> FeatureResult:
        region = config.get("region")
        expected = config.get("expected")
        whitelist = config.get("whitelist", "")

        if region is None:
            raise ValueError("OCR 特征必须指定 region")

        x, y, w, h = region
        roi = screen[y:y+h, x:x+w]
        text = self.ocr.recognize_region(roi, whitelist=whitelist)

        confidence = 0.0
        if expected:
            matched = expected.lower() in text.lower()
            if matched:
                confidence = 1.0
        else:
            matched = len(text.strip()) > 0
            confidence = min(1.0, len(text.strip()) / 5.0)

        return FeatureResult(matched=matched, confidence=confidence, detail={"text": text})


class FeatureFactory:
    """特征工厂"""

    _features = {
        "template": TemplateFeature,
        "color": ColorFeature,
        "ocr": OcrFeature,
    }

    @classmethod
    def create(cls, feature_type: str, ocr_engine: OcrEngine = None) -> Feature:
        if feature_type == "ocr":
            return OcrFeature(ocr_engine)
        feature_class = cls._features.get(feature_type)
        if feature_class is None:
            raise ValueError(f"未知特征类型: {feature_type}")
        return feature_class()

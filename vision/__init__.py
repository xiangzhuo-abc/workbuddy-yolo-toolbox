"""
视觉识别模块

基于 OpenCV + Tesseract 的图像识别封装，用于游戏自动化中的：
- 屏幕截图
- 页面识别
- UI 元素/按钮检测
- 模板匹配
- OCR 文字识别
- 颜色/进度检测
- 数字识别（模板匹配）
"""

from .capture import Capture
from .template import TemplateMatcher, MatchResult
from .ocr import OcrEngine
from .color import ColorDetector
from .page import PageClassifier, PageResult
from .element import ElementDetector, ElementResult
from .digit import DigitRecognizer, DigitResult
from .features import FeatureFactory, FeatureResult
from .utils import load_image, save_image, draw_match, save_debug

__all__ = [
    "Capture",
    "TemplateMatcher",
    "MatchResult",
    "OcrEngine",
    "ColorDetector",
    "PageClassifier",
    "PageResult",
    "ElementDetector",
    "ElementResult",
    "DigitRecognizer",
    "DigitResult",
    "FeatureFactory",
    "FeatureResult",
    "load_image",
    "save_image",
    "draw_match",
    "save_debug",
]

"""
OCR 文字识别模块

基于 Tesseract 的文字识别封装，支持数字、资源数量等识别。
"""

import numpy as np
import cv2
import pytesseract
import shutil
from pathlib import Path
from typing import Optional


# 常见 Tesseract 安装路径（Windows）
_CANDIDATE_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _find_tesseract() -> Optional[str]:
    """自动查找 Tesseract 可执行文件"""
    # 1. 检查 PATH 中是否有 tesseract
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    # 2. 检查候选路径
    for candidate in _CANDIDATE_PATHS:
        if Path(candidate).exists():
            return candidate

    return None


class OcrEngine:
    """基于 Tesseract 的 OCR 引擎"""

    def __init__(self, tesseract_cmd: str = None, lang: str = "eng"):
        """
        Args:
            tesseract_cmd: Tesseract 可执行文件路径，None 时自动查找
            lang: 识别语言，默认英文
        """
        if tesseract_cmd is None:
            tesseract_cmd = _find_tesseract()

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.lang = lang
        self.tesseract_cmd = tesseract_cmd

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        OCR 预处理：灰度化 + 放大 + OTSU 二值化

        Args:
            image: BGR 图像

        Returns:
            二值化后的图像
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 放大 2 倍，提高小字体识别率
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # OTSU 自适应二值化
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def recognize(self, image: np.ndarray,
                  whitelist: str = None,
                  psm: int = 7) -> str:
        """
        识别图像中的文字

        Args:
            image: 待识别图像（BGR 或灰度）
            whitelist: 允许的字符集，例如 "0123456789,"
            psm: Tesseract 页面分割模式，7 表示单行文字

        Returns:
            识别到的文字字符串
        """
        processed = self._preprocess(image)

        config = f"--psm {psm}"
        if whitelist:
            config += f' -c tessedit_char_whitelist={whitelist}'

        try:
            text = pytesseract.image_to_string(processed, lang=self.lang, config=config)
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "未找到 Tesseract 引擎。请安装 Tesseract 并配置路径，"
                "或初始化 OcrEngine 时传入 tesseract_cmd 参数。"
            ) from e

        return text.strip()

    def recognize_region(self, screen: np.ndarray, x: int, y: int, w: int, h: int,
                         whitelist: str = None, psm: int = 7) -> str:
        """
        识别屏幕指定区域的文字

        Args:
            screen: 完整屏幕截图
            x, y: 区域左上角坐标
            w, h: 区域宽高
            whitelist: 允许的字符集
            psm: 页面分割模式

        Returns:
            识别到的文字
        """
        roi = screen[y:y + h, x:x + w]
        return self.recognize(roi, whitelist=whitelist, psm=psm)

    def recognize_number(self, image: np.ndarray) -> Optional[int]:
        """
        识别图像中的数字（自动去除逗号）

        Args:
            image: 待识别图像

        Returns:
            整数，识别失败返回 None
        """
        text = self.recognize(image, whitelist="0123456789,", psm=7)
        # 去掉逗号和空格
        cleaned = text.replace(",", "").replace(" ", "").strip()

        if not cleaned:
            return None

        try:
            return int(cleaned)
        except ValueError:
            return None

"""
视觉工具函数

提供可视化、图像保存等辅助功能。
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """
    安全加载图像文件（支持中文路径）

    OpenCV 的 cv2.imread 在 Windows 上可能无法处理非 ASCII 路径，
    此函数使用 Python 读取文件后再用 cv2.imdecode 解码。
    """
    with open(path, "rb") as f:
        data = f.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法加载图像: {path}")
    return image


def save_image(path: str, image: np.ndarray) -> str:
    """
    安全保存图像文件（绕过 OpenCV 在 Windows 上的写入问题）

    使用 Pillow 保存，支持中文路径和各种格式。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 3 and image.shape[2] == 3:
        # BGR -> RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif image.ndim == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(image)

    pil_image.save(path)
    return path


def draw_match(image: np.ndarray, x: int, y: int, w: int, h: int,
               label: str = None, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    在图像上绘制匹配框

    Args:
        image: 原始图像
        x, y: 左上角坐标
        w, h: 宽高
        label: 标签文字
        color: 框颜色 (B, G, R)

    Returns:
        绘制后的图像
    """
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

    if label:
        cv2.putText(result, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


def save_debug(image: np.ndarray, name: str, output_dir: str = "debug") -> str:
    """保存调试图像"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/{name}"
    save_image(path, image)
    return path

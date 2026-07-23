"""
屏幕截图模块

通过 adb 获取 Android 设备/模拟器的屏幕截图。
"""

import subprocess
import numpy as np
import cv2
from pathlib import Path


class Capture:
    """基于 adb 的屏幕截图工具"""

    def __init__(self, device_serial: str = None, adb_path: str = "adb"):
        """
        Args:
            device_serial: 设备序列号，如 127.0.0.1:16448
            adb_path: adb 可执行文件路径
        """
        self.device_serial = device_serial
        self.adb_path = adb_path

    def _build_cmd(self, action: str) -> list:
        """构建 adb 命令"""
        cmd = [self.adb_path]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(action.split())
        return cmd

    def screenshot(self) -> np.ndarray:
        """
        截取屏幕并返回 OpenCV 图像 (BGR 格式)

        Returns:
            np.ndarray: 形状 (height, width, 3) 的 BGR 图像
        """
        cmd = self._build_cmd("shell screencap -p")
        result = subprocess.run(cmd, capture_output=True, check=True)

        # adb 在 Windows 上输出的 PNG 可能有 \r\n 问题，需要替换
        data = result.stdout.replace(b"\r\n", b"\n")

        # 解码为 OpenCV 图像
        img_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError("无法解码截图，adb 输出可能不是有效的 PNG")

        return image

    def screenshot_to_file(self, path: str) -> str:
        """截图并保存到文件"""
        image = self.screenshot()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return path

    def get_screen_size(self) -> tuple:
        """获取屏幕分辨率 (width, height)"""
        cmd = self._build_cmd("shell wm size")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # 输出格式: Physical size: 1080x2400
        line = result.stdout.strip()
        size_part = line.split(":")[-1].strip()
        width, height = map(int, size_part.split("x"))
        return width, height

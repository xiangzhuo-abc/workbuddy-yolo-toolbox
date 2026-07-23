# 第三方软件声明

本项目以 AGPL-3.0-only 开源发布。下列第三方软件保留各自的版权和许可证。本文件记录当前开发环境中准备进入 Windows EXE 的直接运行依赖；正式构建必须按实际锁定版本复核并随二进制一起分发对应许可证文本。

| 组件 | 当前版本 | 许可证 | 上游项目 |
| --- | --- | --- | --- |
| PyQt5 | 5.15.11 | GPL v3 | https://www.riverbankcomputing.com/software/pyqt/ |
| Qt Runtime (PyQt5-Qt5) | 5.15.2 | LGPL v3 / GPL v3 | https://www.qt.io/ |
| PyQt5-sip | 12.18.0 | BSD-2-Clause | https://github.com/Python-SIP/sip |
| Ultralytics | 8.4.90 | AGPL-3.0 | https://github.com/ultralytics/ultralytics |
| PyTorch | 2.7.1+cu118 | BSD-3-Clause | https://pytorch.org/ |
| TorchVision | 0.22.1+cu118 | BSD | https://github.com/pytorch/vision |
| OpenCV Python | 4.11.0.86 | Apache-2.0 | https://github.com/opencv/opencv-python |
| TensorBoard | 2.21.0 | Apache-2.0 | https://github.com/tensorflow/tensorboard |
| NumPy | 2.5.1 | BSD-3-Clause | https://numpy.org/ |
| Pillow | 12.3.0 | HPND | https://python-pillow.org/ |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/ |
| pytesseract | 0.3.13 | Apache-2.0 | https://github.com/madmaze/pytesseract |
| Inno Setup | 6.7.3 | Inno Setup License | https://jrsoftware.org/isinfo.php |
| Inno Setup 简体中文翻译 | 6.5.0+ | MIT | https://github.com/kira-96/Inno-Setup-Chinese-Simplified-Translation |

Inno Setup 简体中文翻译固定使用提交 `6da09d23e14443d4cf8f07b1c5fd821bfe459788`，版权归 2019-2020 kirakira 所有，并按 MIT 许可证使用。

PyInstaller 只用于生成发布物，不改变上述运行库的许可证。CUDA 运行库、Microsoft Visual C++ Runtime 以及由依赖递归引入的组件，必须在最终 PyInstaller 清单生成后补充到正式版本的许可证目录和本声明中。

本项目不是 Ultralytics、Qt、PyTorch、OpenCV 或 TensorFlow 的官方产品。各项目名称和商标归其权利人所有。

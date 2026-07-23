YOLO 数据标注工具箱 0.9.0-beta.1 - 源码绿色包

首次使用：
1. 双击 安装依赖.bat；脚本会从 Python 3.9-3.14 中选择可用版本，创建包内 .venv 并安装依赖。
2. 如果依赖安装失败，按窗口中的中文提示处理网络、Python 或 PyTorch/CUDA 问题。
3. 双击 启动YOLO工具箱.bat 打开主界面。
4. 在主界面先运行「环境体检」和「发布前自检」，再按数据准备、标注、划分、训练、测试流程操作。

说明：
- .venv 是发布包自己的运行环境，不会把依赖安装到系统 Python。
- 推荐 Python 3.11-3.13；Python 3.9-3.10 为兼容模式，Python 3.14 为实验性兼容。
- 发布包默认不包含你的 dataset、runs、debug、logs 和本机配置。
- dataset/ 只是空骨架，用户数据会在本机生成。
- 如需随包附带模型，请用 build_release.py --include-models 重新打包。
- 本项目使用 AGPL-3.0-only；许可证见 LICENSE。
- 对应源码: 未配置（开发构建）

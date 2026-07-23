# YOLO 数据标注工具箱

这是一个面向游戏 UI 目标检测的 YOLO 工具箱，用来完成截图导入、框选标注、数据集划分、模型训练、模型测试和 TensorBoard 查看。当前推荐使用图形界面，不再需要手动编辑 `data.yaml`。

## 开源许可证

本项目采用 AGPL-3.0-only 开源发布。发布 Windows 二进制时会同时提供对应版本源码、`LICENSE`、`THIRD_PARTY_NOTICES.md` 和 SHA256。项目使用 PyQt5、Ultralytics、PyTorch、OpenCV 与 TensorBoard 等第三方软件，各组件保留自己的版权和许可证；本项目不是这些上游项目的官方产品。

## 快速开始

支持 Python 3.9-3.14，推荐 Python 3.11-3.13。Python 3.9-3.10 属于兼容模式，Python 3.14 属于实验性兼容模式。

源码目录中运行：

```bash
python tools/yolo_tool_launcher.py
```

如果首次运行缺依赖，先执行：

```bash
python tools/install_dependencies.py
```

只检查依赖、不安装：

```bash
python tools/install_dependencies.py --check-only
```

发布包首次安装会从 Python 3.9-3.14 中选择可用版本创建包内 `.venv`，安装和启动均固定使用该环境，不依赖系统默认 Python。批处理入口会自动切换到自身所在目录，因此从桌面快捷方式、资源管理器或其他工作目录启动都可以正常找到 `tools` 和配置目录。
项目内的 `dataset`、`runs` 路径会以相对路径保存，移动整个工具目录后会自动使用新位置；外部数据集和外部训练结果目录仍保持原路径。

源码绿色发布包中可以直接双击：

```text
安装依赖.bat
启动YOLO工具箱.bat
```

首次使用建议先点击主界面的「环境体检」，确认 Python、PyQt5、opencv、ultralytics、torch、TensorBoard、路径权限和数据集概要都正常。准备发布或复制给别人前，再运行「发布前自检」确认依赖清单和发布包隔离规则。源码目录若未包含 `tests/core`，自检会明确提示跳过核心测试；这适用于默认发布包。

## 主流程

```text
日常流程：环境体检 → 数据集准备 → 启动标注工具 → 数据集划分 → 数据质量检查 → 模型管理 → 模型训练 → 模型评估 → 模型测试 → TensorBoard
发布前：发布前自检 → 生成发布包
```

主界面采用左侧工作流导航和右侧运行控制台：

- 数据流程：`01 数据集准备`、`02 启动标注工具`、`03 数据集划分`、`04 格式校验`、`05 数据质量检查`
- 训练评估：`06 模型训练`、`07 模型测试`、`08 模型评估`、`09 TensorBoard`
- 维护工具：`10 数据集统计`、`11 环境体检`、`13 备份恢复`、`12 发布前自检`、`14 查看日志`、`15 模型管理`

顶部项目栏集中显示依赖状态、数据集路径和训练结果路径。运行日志和任务进度位于右侧控制台，窗口缩放时工作流导航保持稳定宽度。

## 架构说明

### 核心服务层（第一阶段）

`tools/core` 提供统一路径、原子配置、结构化问题和只读数据集扫描。扫描可识别未处理、已标注、确认无目标、损坏标签、孤立标签和子集重叠，不会修改数据集文件。

### 标注服务（第二阶段）

`tools/core/annotation_service.py` 统一负责 YOLO 标签解析、像素坐标转换、类别映射和预标注去重。标签与 `classes.txt` 使用同目录临时文件原子替换；类别重排或删除中途失败时会回滚已更新文件。损坏标签不会关闭窗口，工具会保留并显示可正常解析的框，同时在状态栏提示问题数量。

标注窗口只有在当前标签保存成功后才会切换图片。若保存失败，原标签保持不变，窗口会停留在当前图片等待处理。

### 任务协议与 Worker（第三阶段）

训练、检测、后台工具和 TensorBoard 使用统一任务事件及生命周期。成功、失败和取消最多产生一个最终事件；窗口关闭时会按取消、等待、终止的顺序清理受管子进程。

### 按需机器学习运行时

Windows EXE 版把基础程序与机器学习运行时分开发布。基础安装包保留手动标注、数据集管理、质量检查、智能划分和备份恢复；训练、测试、评估、自动预标注和 TensorBoard 由独立 Worker 运行时提供。

- `ml-cpu-win-x64-r1`：适用于无 NVIDIA GPU 的电脑和 CPU 推理。
- `ml-cu118-win-x64-r1`：适用于驱动兼容 CUDA 11.8 的 NVIDIA GPU。
- 两种运行时可以共存，首次使用模型功能时按硬件推荐，用户也可以手动切换。
- 已安装运行时、完整缓存、有效 `.part` 断点和用户选择的官方离线包会优先复用；只有这些来源都不可用时才联网下载。
- 托管运行时、缓存和安装暂存位于 `%LOCALAPPDATA%\YOLOToolbox`，不会扫描其他 WorkBuddy 目录或整块磁盘。
- 外部 Python 只在用户主动打开“高级检测”后查询 `py -0p` 和 `PATH`，不会递归搜索本机环境。

### 发布诊断

主界面的“导出诊断”只收集产品版本、依赖版本、运行路径状态和经过脱敏的日志尾部，不会加入图片、标签、模型、数据集或训练结果。未处理异常会写入 `%LOCALAPPDATA%\WorkBuddyYoloTool\crash_reports`；崩溃报告不采集局部变量和用户文件。“关于”提供版本、许可证、第三方声明和公开源码入口。

### 共享界面与主工具箱（第四阶段）

`tools/yolo_ui_theme.py` 统一颜色、间距、控件尺寸和状态角色，`tools/yolo_ui_widgets.py` 提供无业务逻辑的标题、状态、工具按钮和路径栏组件。主工具箱已迁移为浅色工作流控制台。

### 标注工具界面（第五阶段 A）

标注工具已迁移为图片导航、画布工作区和标注检查器三栏布局。左侧图片列表只显示文件名与标注状态，不集中生成缩略图；支持文件名搜索和“全部 / 未标注 / 已标注”筛选。通过列表跳转时仍会先保存当前图片，保存失败会恢复当前选择并阻止切换。右侧使用“标注框 / 类别 / 预标注”标签页，画布与框列表保持选择同步。

缩略图懒加载、精确的未保存修改状态和坐标属性编辑不在第五阶段 A 范围内，后续单独设计和验收。

### 模型测试界面（第五阶段 B）

模型测试窗口已统一为顶部检测配置、中央图片画布和右侧结果检查器。选择图片目录后可通过文件名下拉直接定位；检测运行期间会同时锁定上一张、下一张、文件名下拉和图片选择，避免过期结果覆盖当前图片。结果区显示目标数量、类别摘要和 GUI 侧端到端耗时。

首次检测的端到端耗时包含 worker 启动、模型加载和预热，后续图片会复用常驻 worker，不能直接用首次耗时判断纯推理速度。

### 模型训练界面（第五阶段 C）

模型训练对话框已统一为顶部训练来源、中部参数标签页和底部输出位置。常用的轮数、批次、设备和训练名称位于“基础参数”，图像尺寸与继续训练位于“高级参数”；“数据检查”展示训练集、验证集、测试集和类别数量。

顶部状态会明确显示“数据可用”“建议划分验证集”“训练集不可用”“类别不可用”或“读取失败”。训练集、类别或配置不可用时会禁用“开始训练”；验证集为空只提示警告，不阻断试运行。数据检查页同时显示预计每轮的训练、验证和总 batch 数。

检测到 CUDA 时默认选择 GPU 0，推荐 `batch=16`；CPU 模式推荐 `batch=4`。“自动选择”会真正交给 Ultralytics 决定设备，不再隐式退回 CPU。用户手动修改 batch 后，切换设备不会覆盖自定义值。

### 数据质量检查（第六阶段 A）

主工具箱的“数据质量检查”会在后台执行只读扫描，集中展示标签完整性、精确重复图片、train/val/test 内容泄漏和类别分布。结果分为“问题 / 类别分布 / 重复图片”三个视图；可定位的问题支持直接打开对应分组并跳到目标图片。

本阶段使用 SHA256 检查内容完全相同的图片，不包含近似图片感知哈希，也不会在扫描时自动删除、移动或重划分数据。错误表示会影响训练或验证可信度，警告表示建议人工复查，提示用于补充数据规划。

低样本和类别覆盖按“不同图片数”判断，不把一张图片中的多个框当成多个独立样本。默认先为每个类别保留 5 张训练图片；类别至少达到 6 张后才要求进入验证集，避免对样本不足的类别给出错误的重划分建议。

### 类别感知数据集划分（第六阶段 B）

“数据集划分”采用设置与预览两阶段流程。生成预览只读取数据，集中展示当前数量、计划数量、类别覆盖、移动清单和未解决风险；只有用户确认后才执行。

- `最小修复`：保留现有分组，优先用最少移动补齐可满足的类别覆盖。
- `智能重划分`：重新规划全部有效图片，先保护稀有类别训练样本，再接近目标比例。

划分计划包含数据指纹。图片、标签或类别文件变化后，旧计划会被拒绝并要求重新预览。执行时图片和标签成对进入事务暂存目录；中途失败会自动回滚。新版备份记录图片原分组、标签、配置和 SHA256，可完整恢复已知快照；旧备份继续标记为“仅标签和配置”。

质量检查中的“优化划分”会打开同一对话框并默认选择最小修复，不会在质量检查窗口直接移动文件。

## 1. 环境体检

点击「环境体检」会检查：

- 当前 Python 路径和版本
- PyQt5、opencv-python、numpy、ultralytics、torch、tensorboard、PyYAML、Pillow
- 项目目录、数据集目录、训练结果目录、配置目录是否存在和可写
- `data.yaml` 指向的数据集概要
- 可用 `.pt` 模型数量
- CUDA / GPU 状态

如果出现错误，优先处理错误项；警告项通常不阻塞使用，但会影响训练速度或部分功能。

## 2. 数据集准备

点击「数据集准备」，选择截图目录或单张图片，选择导入到 `train`、`val` 或 `test`。

可选项：

- 清空目标目录后再导入：执行前会自动备份 `labels/`、`classes.txt`、`data.yaml`
- 格式转换：可把图片统一转换为 `jpg` 或 `png`

工具会自动维护标准目录：

```text
dataset/
  images/train
  images/val
  images/test
  images/unlabeled
  labels/train
  labels/val
  labels/test
  classes.txt
  data.yaml
```

## 3. 标注

点击「启动标注工具」打开标注窗口。

常用操作：

- 左键拖动：框选目标
- 输入标签名后添加标签
- 选择标签后继续框选同类目标
- `S`：保存当前图片
- `N`：下一张
- `P`：上一张
- `U`：下一张未标注
- `F`：适应窗口
- `Del`：删除选中框

左侧图片导航支持直接点击定位、按文件名过滤和按标注状态筛选。大数据集只创建轻量文本项，不会在启动时批量解码图片。

标签变化会自动同步：

- `dataset/classes.txt`
- `dataset/data.yaml`
- Ultralytics 的标签缓存会被清理，避免训练读到旧标签

删除标签、重命名标签、排序标签、拖动调整标签顺序前，会自动备份标签和配置。

## 4. 数据集划分

点击「数据集划分」设置 `train / val / test` 比例。

说明：

- `train / val / test` 按设置比例互斥划分，同一图片不会重复出现在多个子集
- 划分会先在临时目录构建并校验，成功后才替换正式目录
- 未标注图片会移动到 `images/unlabeled`
- 执行前会自动备份 `labels/`、`classes.txt`、`data.yaml`
- 划分完成后会自动更新 `data.yaml`

建议先用默认或接近默认比例，例如：

```text
train=0.70, val=0.20, test=0.10
```

如果数据量很少，可以先不启用 test：

```text
train=0.80, val=0.20, test=0.00
```

## 5. 检查分析

训练前建议依次运行：

- 「格式校验」：检查 YOLO 标签格式、类别 ID、坐标范围
- 「数据质量检查」：检查精确重复、跨分组内容泄漏、低样本类别和验证集覆盖
- 「数据集统计」：查看类别分布、标注数量、图片数量
- 「环境体检」：确认依赖和路径状态
- 「发布前自检」：确认 `requirements.txt` 可解析、关键源码齐全、发布包不会混入真实数据和运行产物
- 「备份恢复」：查看自动备份，并恢复选中的标签和配置

如果格式校验失败，先回到标注工具修复问题，再训练。

## 6. 模型训练

点击「模型训练」选择预训练模型和数据集配置，再通过“基础参数 / 高级参数 / 数据检查”三个标签页完成设置与核对。开始训练前应查看顶部数据状态；验证集为空时仍可继续，但建议先完成数据集划分。

### 模型管理

点击「模型管理」可以下载官方预训练目标检测模型。当前提供 YOLOv8 和 YOLO11 两个系列，每个系列包含 `n`、`s`、`m`、`l`、`x` 五种尺寸。模型会保存到项目的 `models/` 目录，下载完成后可在模型训练、模型测试和预标注中选择。

- 下载地址固定为 Ultralytics 官方发布地址，不接受任意 URL 或本地路径作为下载源。
- 下载完成会校验文件大小和 PyTorch 权重容器格式，失败不会替换已有模型。
- 已存在的合法模型会显示为已安装；重新下载前会要求确认覆盖。
- 下载不会自动加载模型、开始训练或执行推理；需要在对应功能窗口中手动选择。
- 下载过程中可以取消，未完成的临时文件会自动清理。

常用参数：

- 预训练模型：通过「模型管理」下载的 `yolov8n.pt`、`yolo11n.pt` 或项目中的其他 `.pt`
- 训练轮数：小数据集可先用 50-100 轮
- 批次大小：显存不足时调小
- 图像尺寸：默认 640
- 训练设备：自动、CPU 或 GPU
- 训练名称：结果会保存到 `runs/{名称}/`

每轮耗时主要由图片数量和 batch 共同决定。训练图片增加或 batch 减小时，每轮 batch 数会增加；应优先对比“预计每轮”中的 batch 数，而不是只比较不同数据规模下的单轮秒数。为保持 Windows 训练稳定，当前仍使用 `workers=0`，未自动启用多进程加载或内存缓存。

训练产物：

```text
runs/{名称}/weights/best.pt
runs/{名称}/weights/last.pt
```

训练过程中主界面会显示实时日志。

## 7. 模型测试

点击「模型测试」打开测试窗口。

流程：

1. 选择或加载 `.pt` 模型
2. 选择单张图片或图片目录
3. 调整置信度阈值
4. 从文件名下拉中定位需要测试的图片
5. 点击「开始检测」
6. 查看右侧检测结果、端到端耗时和图片上的检测框

如果没有输出结果：

- 降低置信度阈值
- 确认模型已经加载成功
- 确认图片场景和训练数据一致
- 确认类别名和标注文件同步
- 运行主界面的「格式校验」和「数据集统计」

## 8. 模型评估

点击「模型评估」打开评估窗口。该功能用于验证训练结果，不会自动替换模型，也不会修改数据集。

1. 选择候选模型，通常使用训练结果目录下的 `weights/best.pt`。
2. 可选选择基准模型，用于比较当前候选模型是否值得采用。
3. 选择同一份 `data.yaml`、评估分组、图像尺寸和设备后开始评估。
4. 查看 Precision、Recall、mAP50、mAP50-95 以及逐类别指标。

评估结果写入 `runs/evaluations/`，包含标准化 `evaluation.json`、混淆矩阵、曲线和验证图。候选与基准只有在数据指纹、类别定义和评估参数一致时才会比较；测试图片少于 20 张时，界面会标记“仅供参考”。

模型比较以 mAP50-95 为主要依据，同时检查 Recall 和 mAP50 是否明显下降。建议先查看逐类别退化项和预测图，再决定是否继续训练或更换模型。

## 9. TensorBoard

训练完成后点击「TensorBoard」，工具会启动 TensorBoard 并自动打开浏览器。

默认地址：

```text
http://localhost:6006
```

如果 `6006` 已被占用，工具会自动改用后续可用端口，并在执行日志中显示实际访问地址。

如果启动失败，先在「环境体检」中确认 `tensorboard` 已安装。

## 自动备份

以下操作会自动备份：

- 数据集准备时勾选“清空目标目录后再导入”
- 数据集划分
- 删除标签
- 重命名标签
- 标签排序
- 拖动调整标签顺序

备份位置：

```text
dataset/backups/{时间}-{原因}/
```

备份范围：

- `labels/`
- `classes.txt`
- `data.yaml`

不会备份图片文件，避免占用过多空间。

点击主界面的「备份恢复」可以查看备份列表，打开备份目录，或恢复选中的备份。恢复范围同样只包含：

- `labels/`
- `classes.txt`
- `data.yaml`

恢复前会再次自动备份当前状态，恢复后建议重新运行「格式校验」和「数据集统计」。图片文件不会被恢复或修改。

## 命令行兼容入口

图形界面是推荐入口；以下脚本仍保留，内部已改为调用安全后端。

安装或检查运行依赖：

```bash
python tools/install_dependencies.py
python tools/install_dependencies.py --check-only
python tools/install_dependencies.py --dry-run
```

导入截图：

```bash
python tools/prepare_dataset.py --source "你的截图目录" --split train
```

清空后导入，执行前会自动备份：

```bash
python tools/prepare_dataset.py --source "你的截图目录" --split train --clean
```

只读预览类别感知划分：

```bash
python tools/split_dataset.py --mode repair --train 0.8 --val 0.15 --test 0.05
```

确认预览后显式执行，执行前会创建完整分组备份：

```bash
python tools/split_dataset.py --mode repair --train 0.8 --val 0.15 --test 0.05 --apply
```

对新数据集进行完整智能重划分：

```bash
python tools/split_dataset.py --mode full --train 0.8 --val 0.15 --test 0.05 --apply
```

指定外部数据集目录：

```bash
python tools/prepare_dataset.py --source "你的截图目录" --dataset-dir "D:\你的数据集"
python tools/split_dataset.py --dataset-dir "D:\你的数据集"
```

CLI 默认只生成预览。没有 `--apply` 时不会移动图片、标签或清理缓存。

## 发布包

发布前建议先运行自检：

```bash
python tools/preflight_check.py
```

如果只想检查发布清单和空数据集骨架，不检查当前运行环境：

```bash
python tools/preflight_check.py --release-only
```

生成源码绿色发布包：

```bash
python tools/build_release.py
```

### Windows 轻量安装包与按需运行时

Windows 二进制发布拆成三个独立产物：轻量基础安装包、CPU 运行时 ZIP 和 CUDA 11.8 运行时 ZIP。目标电脑不需要安装 Python；只有训练、测试、评估、预标注或 TensorBoard 需要机器学习运行时。

先在独立的 Python 3.12/3.13 x64 环境中构建 CPU 运行时：

```powershell
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-runtime-windows-cpu.txt
python tools/build_runtime.py --profile cpu --clean --out-dir tmp\runtime-candidate --base-url "https://github.com/xiangzhuo-abc/workbuddy-yolo-toolbox/releases/download/v0.9.0-beta.1"
```

再在独立的 CUDA 11.8 构建环境中生成 GPU 运行时：

```powershell
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-release-windows-cu118.txt
python tools/build_runtime.py --profile cuda118 --clean --out-dir tmp\runtime-candidate --base-url "https://github.com/xiangzhuo-abc/workbuddy-yolo-toolbox/releases/download/v0.9.0-beta.1"
```

两个 ZIP 都生成后，`tmp\runtime-candidate\runtime_catalog.json` 会包含实际字节数、安装体积、SHA256 和固定 HTTPS 地址。基础程序只接受同时包含 CPU 与 CUDA 11.8 且所有下载字段完整的可信清单：

```powershell
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-base-windows.txt
python tools/build_exe.py --clean --out-dir tmp\exe-candidate --runtime-catalog tmp\runtime-candidate\runtime_catalog.json
```

构建环境使用 Python 3.12 或 3.13 x64，并至少保留 12 GiB 可用磁盘空间。CPU 与 GPU 依赖必须在各自隔离的构建环境中安装，不向基础构建注入外部 `site-packages`。候选产物位于：

```text
tmp\runtime-candidate\
├─ ml-cpu-win-x64-r1.zip
├─ ml-cu118-win-x64-r1.zip
└─ runtime_catalog.json

tmp\exe-candidate\YOLO数据标注工具箱\
├─ YOLO工具箱.exe
└─ _internal\
```

基础目录明确禁止包含 Worker、Torch、Ultralytics、TensorBoard、Polars 和 CUDA DLL，目标是不超过 300 MiB。运行时 ZIP 明确禁止包含主界面、PyQt、模型、数据集、训练结果和用户配置。CUDA 运行时需要 Windows 10/11 x64 和兼容的 NVIDIA 驱动；没有可用 GPU 时选择 CPU 运行时。

安装 Inno Setup 6/7 后，可以从已验证的便携目录生成 Windows 安装器：

```powershell
python tools/build_installer.py --exe-dir tmp\exe-candidate --out-dir tmp\installer-candidate
```

正式公开发布必须同时提供对应源码地址：

```powershell
python tools/build_installer.py --exe-dir tmp\exe-candidate --out-dir tmp\installer-candidate --official-release --source-url "https://github.com/xiangzhuo-abc/workbuddy-yolo-toolbox/tree/v0.9.0-beta.1"
```

安装器使用固定 AppId，安装到 `Program Files\WorkBuddy\YOLO数据标注工具箱`，创建开始菜单入口并提供可选桌面快捷方式。升级只覆盖程序目录；兼容的运行时会继续复用，不会重复下载。卸载基础程序不会删除 `%LOCALAPPDATA%\YOLOToolbox`、`%LOCALAPPDATA%\WorkBuddyYoloTool` 或用户工作区。公开分发前应对基础安装器和两个 Worker EXE 使用受信任证书签名，并重新记录所有发布文件的 SHA256。

验证完整候选：

```powershell
python tools/preflight_check.py --release-only --no-pip --exe-dir tmp\exe-candidate --runtime-dir tmp\runtime-candidate
powershell -ExecutionPolicy Bypass -File packaging\smoke_test.ps1 -ExeDir tmp\exe-candidate -RuntimeDir tmp\runtime-pyinstaller\dist\cpu\ml-cpu-win-x64-r1 -Workspace tmp\smoke-workspace
```

未传 `-RuntimeDir` 时，冒烟脚本只验证轻量基础程序，并把 Worker、Torch 和 TensorBoard 检查明确标为 `SKIP`。

默认发布包会排除：

- 真实 `dataset` 数据
- `runs`
- `debug`
- `logs`
- `.git`
- `.workbuddy`
- `__pycache__`
- `.pt` 模型文件

默认发布包不包含模型。安装后可以打开「模型管理」按需下载；这会避免把较大的权重文件和具体机器环境一起打包。

源码绿色包安装流程：

1. 安装 Python 3.9-3.14 中任意受支持版本；推荐 Python 3.11-3.13。
2. 双击 `安装依赖.bat`，脚本会自动选择可用版本，创建包内 `.venv` 并安装依赖。
3. 双击 `启动YOLO工具箱.bat` 启动主界面。
4. 若更换 Python 版本或安装环境损坏，删除发布包内 `.venv` 后重新执行安装脚本。

如需附带模型：

```bash
python tools/build_release.py --include-models
```

## 标注建议

1. 每个类别尽量至少 50-100 个框。
2. 框要贴边，不要留太多空白，也不要截掉目标。
3. 覆盖不同场景、不同分辨率、不同 UI 状态。
4. 类别命名保持稳定，不要频繁改名。
5. 训练前先做格式校验和数据集统计。

常见类别示例：

- `attack_button`
- `shop_button`
- `next_button`
- `end_battle_button`
- `surrender_button`
- `gold_bar`
- `elixir_bar`
- `builder_icon`

## 注意事项

- 训练可能耗时较长，CPU 训练会明显慢于 GPU。
- 发布包默认不包含真实数据和模型，用户需要自行导入截图和放置模型。
- EXE 安装目录只保存程序文件；配置和日志位于 `%LOCALAPPDATA%\WorkBuddyYoloTool`，数据集、模型和训练结果位于用户选择的工作区。
- 未签名安装器可能触发 Windows SmartScreen 提示；公开发布时应同时提供 SHA256 和可验证的对应源码链接。
- 反馈故障时优先使用主界面的“导出诊断”；发送前仍建议检查 ZIP 内容是否符合自己的隐私要求。
- 工作区内的 `dataset`、`runs`、`debug`、`logs` 是运行数据，不应直接混入公开发布包。

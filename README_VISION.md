# CoC 视觉识别模块

基于 OpenCV 的游戏画面识别模块，支持页面识别、按钮/元素检测、颜色检测、OCR、数字识别。

## 模块结构

```
vision/
├── __init__.py      # 模块导出
├── capture.py       # 屏幕截图（adb）
├── template.py      # 模板匹配
├── element.py       # UI 元素/按钮检测
├── page.py          # 页面识别器
├── features.py      # 特征检测（模板/颜色/OCR）
├── ocr.py           # Tesseract OCR
├── digit.py         # 模板匹配数字识别
├── color.py         # 颜色/进度检测
└── utils.py         # 工具函数

config/
├── pages.json       # 页面定义
└── elements.json    # 元素/按钮定义

templates/
├── elements/        # 按钮/图标模板
└── digits/          # 数字模板 0-9
```

## 快速开始

### 1. 检测页面

```python
from vision import load_image, PageClassifier

screen = load_image("coc_screenshot.png")
classifier = PageClassifier("config/pages.json")
result = classifier.detect(screen)

print(result.name)        # 最可能页面，如 "home_village"
print(result.confidence)  # 置信度
```

### 2. 检测按钮/元素

```python
from vision import load_image, ElementDetector

screen = load_image("coc_screenshot.png")
detector = ElementDetector("config/elements.json")

# 检测单个元素
attack = detector.detect(screen, "attack_button")
if attack.found:
    x, y = attack.matches[0].center
    print(f"进攻按钮中心: ({x}, {y})")

# 批量检测
results = detector.detect_all(screen, ["attack_button", "shop_button"])
```

### 3. 截图

```python
from vision import Capture

cap = Capture(device_serial="127.0.0.1:16448")
screen = cap.capture()
```

## 添加新页面

编辑 `config/pages.json`：

```json
{
  "home_village": {
    "features": {
      "attack_button_present": {
        "type": "template",
        "template": "templates/elements/attack_button.png",
        "threshold": 0.8,
        "region": [0, 780, 200, 200],
        "weight": 2.0
      }
    }
  }
}
```

特征类型：
- `template`：模板匹配
- `color`：颜色区域占比
- `ocr`：文字识别

## 添加新按钮

1. 从截图中截取按钮小图，保存到 `templates/elements/`
2. 编辑 `config/elements.json`：

```json
{
  "my_button": {
    "description": "我的按钮",
    "template": "templates/elements/my_button.png",
    "threshold": 0.8,
    "region": [100, 100, 200, 200]
  }
}
```

## 数字识别

游戏字体不适合通用 OCR，使用模板匹配逐位识别：

1. 准备清晰的 `0.png` ~ `9.png` 放到 `templates/digits/`
2. 使用 `DigitRecognizer`：

```python
from vision import DigitRecognizer

recognizer = DigitRecognizer("templates/digits", threshold=0.8)
text, details = recognizer.recognize(roi)
number = recognizer.recognize_number(roi)
```

## 测试

```bash
python test_recognition.py
```

输出：
- 当前页面识别结果
- 所有按钮/元素位置
- 数字识别结果
- 可视化调试图保存到 `debug/`

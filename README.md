# Game Theory AI

## ADB工具包 (Android Debug Bridge Tools)
这个包提供了一系列用于Android设备自动化操作的工具。

### 主要功能
1. 设备截图获取
   - 支持自定义保存路径
   - 自动生成时间戳文件名
   - 错误重试机制

2. ADB工具管理
   - 自动验证ADB可用性
   - 支持自定义ADB路径
   - 异常处理和日志记录

### 配置说明
配置文件位置: config/app_config.yaml
主要配置项:
- adb.path: ADB工具路径
- adb.screenshot.remote_path: 设备端截图路径
- adb.screenshot.local_dir: 本地保存目录
- adb.screenshot.filename_format: 文件名格式
- logging: 日志配置

### 使用示例
```python
from src.core.mahjong import MahjongAI
ai = MahjongAI()
screenshot_path = ai.take_screenshot()
print(f"截图保存在: {screenshot_path}")
```

### 环境硬性要求
1. Python 3.8 (必须使用 Conda 环境)
2. Tesseract OCR
3. ADB 工具
4. 必须使用 Conda 3.8 环境
5. 不要使用 venv
6. 确保 ADB 工具可用
7. 运行前检查配置文件
8. 保持依赖包版本一致

### 环境配置步骤
```bash
# 1. 创建并激活conda环境
conda create -n gameai python=3.8
conda activate gameai

# 2. 安装项目依赖
pip install -e .

# 3. 安装Tesseract OCR
# Mac
brew install tesseract
# Ubuntu
sudo apt-get install tesseract-ocr
# Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装
```

### 使用说明
1. 配置说明
```bash
# 编辑配置文件
vim config/app_config.yaml  # 设置ADB路径等参数
```

2. 运行说明
```bash
# 初始化环境
python setup.py

# 启动麻将AI
python -m src.core.mahjong
```

## 项目简介
麻将AI分析工具，提供自动截图、OCR识别和决策功能。

## 项目结构
```
GameTheoryAI/
├── src/                    # 源代码
│   └── core/              # 核心功能
│       └── mahjong.py     # 麻将AI主程序
├── data/                  # 数据目录
│   ├── screenshots/      # 截图
│   └── templates/        # 模板图片
├── config/               # 配置文件
│   └── app_config.yaml   # 应用配置
├── README.md            # 项目文档
├── requirements.txt     # 依赖管理
└── setup.py            # 安装配置
```

## 开发目标
| 指标 | 目标值 |
|------|--------|
| 截屏延迟 | <16ms |
| OCR准确率 | >95% |
| 决策延迟 | <200ms |

## 依赖要求
- Python >= 3.8
- opencv-python
- numpy
- pytesseract
- pyyaml
- adb工具

## 作者信息
- 作者: GameTheoryAI Team
- 版本: 1.0.0
- 许可: MIT
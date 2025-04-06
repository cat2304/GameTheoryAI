# GameTheoryAI

麻将游戏分析工具，支持OCR识别和游戏监控。

## 功能特点

- OCR麻将牌识别
- 实时游戏监控
- 自动截图和分析
- Web界面展示
- ADB设备控制

## 项目结构

```
GameTheoryAI/
├── src/                    # 源代码目录
│   ├── core/              # 核心功能
│   │   ├── ocr/          # OCR相关功能
│   │   └── game/         # 游戏相关功能
│   ├── utils/            # 工具函数
│   └── web/             # Web界面
├── tests/                # 测试目录
├── data/                 # 数据目录
│   ├── screenshots/     # 截图存储
│   └── logs/           # 日志存储
├── config/              # 配置文件
└── docs/               # 文档
```

## 环境要求

- Python: Conda 环境 3.8
  ```bash
  # 创建conda环境
  conda create -n mahjong python=3.8
  
  # 激活环境
  conda activate mahjong
  
  # 安装依赖
  conda install pytest pyyaml opencv pillow numpy
  pip install pytesseract
  ```

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/GameTheoryAI.git
cd GameTheoryAI
```

2. 安装依赖：
```bash
# 确保在conda环境中
conda activate mahjong

# 安装项目依赖
pip install -e .
```

3. 安装Tesseract OCR：
```bash
# Mac
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Windows
# 下载安装器：https://github.com/UB-Mannheim/tesseract/wiki
```

## 使用说明

1. 配置：
   - 编辑 `config/app_config.yaml` 设置ADB路径和其他参数
   - 编辑 `config/ocr_config.json` 设置OCR识别参数

2. 运行游戏监控：
```python
from src.core.game.monitor import MahjongGameMonitor

monitor = MahjongGameMonitor()
monitor.start()
```

3. 运行Web界面：
```bash
# 确保在conda环境中
conda activate mahjong
python src/web/app.py
```

## 开发指南

1. 环境设置
   - 确保使用 Conda 3.8 环境
   - 安装所有必要依赖
   - 配置 ADB 工具（用于Android设备交互）

2. 运行测试
   ```bash
   # 确保在conda环境中
   conda activate mahjong
   
   # 运行测试
   python -m pytest tests/
   ```

3. 代码规范
   - 使用 Python 类型注解
   - 遵循 PEP 8 编码规范
   - 所有函数和类都需要添加文档字符串

## 配置说明

主配置文件：`app_config.yml`

```yaml
# 示例配置
adb:
  path: "adb"
  screenshot:
    remote_path: "/sdcard/screenshot.png"
    local_dir: "data/screenshots"
    
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 注意事项

- 必须使用 Conda 3.8 环境
- 运行前检查配置文件是否正确
- 确保 ADB 工具可用且设备已连接
- 不要使用 venv，本项目只支持 conda 环境

## 许可证

MIT License

## 作者

GameTheoryAI Team
# AI 棋牌辅助系统

基于博弈论与强化学习的智能辅助系统。

## 核心模块

| 模块 | 技术方案 | 目标指标 |
|------|----------|----------|
| 实时截屏 | DXGI桌面复制API | 延迟 <16ms |
| 图像识别 | PaddleOCR | 准确率 >95% |
| 策略引擎 | 规则树 + Q-learning | 决策延迟 <200ms |
| 操作模拟 | 标准输入事件 | 响应时间 <50ms |

## 系统要求

### 硬件配置
d 

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | GTX 1060 6GB | RTX 4090 |
| CPU | Intel i5 (4核) | Intel i9 (16核) |
| 内存 | 8GB | 64GB |

### 软件依赖

```bash
python >= 3.8
cuda >= 11.0
paddlepaddle-gpu
opencv-python
numpy
```

## 开发规范

1. 核心模块使用 C++ 开发，提供 Python 接口
2. 遵循 PEP 8 编码规范

## 免责声明

本项目仅用于教育研究目的，禁止用于任何违法用途。使用者需遵守相关法律法规。

## 更新记录

### 2024-04-05
- 添加ADB工具模块
  - 支持自动截图功能
  - 按日期组织文件存储（格式：YYYYMMDD）
  - 自动递增的文件命名（1.png, 2.png, ...）
  - 完整的错误处理和日志记录
- 添加配置文件支持
  - 使用YAML格式配置文件
  - 支持自定义ADB路径
  - 支持自定义保存目录
  - 支持自定义文件命名规则
- 添加测试用例
  - 截图功能测试
  - 错误处理测试
- 添加项目依赖管理
  - 创建requirements.txt
  - 添加pyyaml依赖

## 项目结构
```
GameTheoryAI/
├── config/
│   └── config.yaml      # 配置文件
├── fetch/
│   ├── __init__.py      # ADB工具包
│   └── adb.py           # ADB工具实现
├── test/
│   └── test_screenshot.py # 测试脚本
└── requirements.txt      # 项目依赖
```

## 使用说明

### 环境要求
- Python >= 3.6
- ADB工具
- 依赖包：`pip install -r requirements.txt`

### 基本用法
```python
from fetch.adb import ADBHelper

# 创建ADB助手实例
adb = ADBHelper()

# 执行截图
screenshot_path = adb.take_screenshot()
print(f"截图保存在: {screenshot_path}")
```

### 配置文件说明
配置文件位置：`config/config.yaml`
```yaml
adb:
  path: "/path/to/adb"  # ADB工具路径
  screenshot:
    remote_path: "/sdcard/screenshot.png"  # 设备端保存路径
    local_dir: "/path/to/save"  # 本地保存目录
    date_format: "%Y%m%d"  # 日期格式
    filename_format: "{number}.png"  # 文件名格式
    counter_file: "counter.txt"  # 计数器文件

logging:
  level: "INFO"  # 日志级别
  format: "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式
```
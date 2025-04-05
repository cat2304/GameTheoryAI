# GameTheoryAI

## 核心功能

### ADB工具模块
- 自动截图功能
- 文件存储规则：
  - 按日期组织（YYYYMMDD）
  - 自动递增命名（1.png, 2.png, ...）
- 配置文件支持：
  - ADB路径配置
  - 保存目录配置
  - 文件命名规则配置

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
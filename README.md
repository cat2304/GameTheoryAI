# Mumu模拟器控制服务

## 快速开始

### 1. 环境准备
- Python 3.8+
- ADB工具
- Mumu模拟器

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动服务
```bash
python main.py
```

## API接口

### 1. 点击操作
```bash
curl -X POST http://localhost:8000/api/mumu/click \
  -H "Content-Type: application/json" \
  -d '{"x": 100, "y": 200}'
```

响应示例：
```json
{
    "success": true,
    "message": "点击成功",
    "data": {
        "x": 100,
        "y": 200,
        "timestamp": 1234567890.123
    }
}
```

### 2. 截屏操作
```bash
curl -X POST http://localhost:8000/api/mumu/screenshot
```

响应示例：
```json
{
    "success": true,
    "message": "截图成功",
    "data": {
        "path": "data/screenshots/screenshot_1234567890.png",
        "timestamp": 1234567890
    }
}
```

### 3. OCR识别
```bash
curl -X POST http://localhost:8000/api/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/screenshots/screenshot_1234567890.png"}'
```

响应示例：
```json
{
    "success": true,
    "message": "识别成功",
    "data": {
        "text": "识别结果",
        "confidence": 0.95
    }
}
```

## 项目结构

```
GameTheoryAI/
├── data/
│   ├── debug/          # 调试输出目录
│   ├── screenshots/    # 截图保存目录
│   └── templates/      # 模板图片目录
├── logs/               # 日志文件目录
├── src/
│   ├── control/        # 控制模块
│   ├── core/           # 核心逻辑
│   └── vision/         # 视觉识别模块
├── main.py            # 主程序入口
└── README.md          # 项目说明文档
```

## 核心业务逻辑

### 1. 程序入口 (main.py)
- 配置日志系统
- 创建必要的目录结构
- 启动异步截图线程
- 初始化游戏控制器
- 进入主循环

### 2. 异步截图模块 (src/vision/screen.py)
- 每5秒执行一次截图
- 保存截图到 `data/screenshots/latest.png`
- 处理截图异常情况
- 提供截图状态反馈

### 3. 游戏控制器 (src/core/game_controller.py)

## 环境要求
- Python 3.8+
- OpenCV
- ADB工具
- 其他依赖见 requirements.txt

## 使用说明
1. 安装依赖：`pip install -r requirements.txt`
2. 确保ADB已连接设备
3. 运行程序：`python main.py`

### 注释风格
1. 步骤清晰：
   - 每个方法都标注是第几步
   - 注释直接说明方法的主要功能
   - 去掉了冗长的参数和返回值说明

2. 逻辑流程：
   - 第一步：获取屏幕截图
   - 第二步：识别截图中的牌面
   - 第三步：处理决策
   - 第四步：执行决策

3. 注释风格：
   - 简洁明了
   - 直接说明功能
   - 保持一致性

## API文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

# GameTheoryAI

基于深度学习的游戏AI系统，用于自动化游戏操作和决策。

## 功能特点

- 基于 PaddleOCR 的文本识别
- 基于 ADB 的自动化操作
- 基于 FastAPI 的 RESTful API 服务
- 支持设备管理和监控
- 支持屏幕捕获和图像处理
- 支持自动化点击和操作

## 环境要求

- Python 3.8+
- ADB 工具
- Android 设备或模拟器
- PaddleOCR 模型

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/GameTheoryAI.git
cd GameTheoryAI
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装 ADB 工具（如果尚未安装）：
```bash
# macOS
brew install android-platform-tools

# Ubuntu
sudo apt-get install android-tools-adb
```

## 使用方法

1. 启动服务：
```bash
python main.py
```

2. 服务将在 http://localhost:8000 上运行

## API 接口

### 设备管理

#### 获取设备列表
```bash
GET /api/device/list
```
返回所有已连接的设备列表。

#### 获取当前设备信息
```bash
GET /api/device/current
```
返回当前连接的设备信息，包括设备ID和屏幕尺寸。

#### 连接设备
```bash
POST /api/device/connect
Content-Type: application/json

{
    "device_id": "设备ID"
}
```
连接指定的设备。

#### 断开设备连接
```bash
POST /api/device/disconnect
```
断开当前设备的连接。

### 游戏操作

#### 点击操作
```bash
POST /api/mumu/click
Content-Type: application/json

{
    "x": 100,
    "y": 100
}
```
在指定坐标执行点击操作。

#### 截图
```bash
POST /api/mumu/screenshot
```
获取当前屏幕截图。

#### OCR识别
```bash
POST /api/ocr/recognize
Content-Type: application/json

{
    "image_path": "data/screenshots/latest.png"
}
```
对指定图片进行OCR识别。

## 目录结构

```
GameTheoryAI/
├── main.py              # 主程序入口
├── requirements.txt     # 依赖包列表
├── README.md           # 项目说明文档
├── src/                # 源代码目录
│   ├── api/            # API接口
│   │   └── routes.py   # 路由定义
│   ├── core/           # 核心功能
│   │   ├── adb.py      # ADB控制器
│   │   ├── ocr.py      # OCR处理器
│   │   └── screen.py   # 屏幕捕获
│   └── utils/          # 工具函数
├── data/               # 数据目录
│   ├── screenshots/    # 截图保存目录
│   └── debug/         # 调试输出目录
└── logs/              # 日志目录
```

## 开发说明

1. 代码风格遵循 PEP 8 规范
2. 使用类型注解提高代码可读性
3. 使用 logging 模块进行日志记录
4. 使用 FastAPI 框架提供 RESTful API
5. 使用 PaddleOCR 进行文本识别
6. 使用 ADB 进行设备控制和操作

## 注意事项

1. 确保 ADB 工具已正确安装并添加到系统环境变量
2. 确保 Android 设备已启用 USB 调试模式
3. 确保设备已通过 USB 连接或网络连接
4. 首次运行时需要下载 PaddleOCR 模型

## 贡献指南

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

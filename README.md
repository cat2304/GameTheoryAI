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

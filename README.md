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

#### 3.1 全图文字识别
```bash
curl -X POST http://localhost:8000/api/ocr/recognize_all \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/screenshots/screenshot_1234567890.png"}'
```

响应示例：
```json
{
    "success": true,
    "message": "识别成功",
    "data": {
        "texts": [
            {
                "text": "识别结果",
                "confidence": 0.95,
                "position": {
                    "x": 100,
                    "y": 200,
                    "width": 50,
                    "height": 30
                }
            }
        ]
    }
}
```

#### 3.2 区域文字识别
```bash
curl -X POST http://localhost:8000/api/ocr/recognize_region \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/screenshots/screenshot_1234567890.png",
    "region": [100, 200, 300, 400],
    "type": 1
  }'
```

参数说明：
- `image_path`: 图片路径
- `region`: 区域坐标 [x, y, width, height]
- `type`: 区域类型
  - `1`: 公牌区域 (OCR_PUBLIC)
  - `2`: 手牌区域 (OCR_HAND)
  - `3`: 操作区域 (OCR_OP)

响应示例：
```json
{
    "success": true,
    "message": "识别成功",
    "data": {
        "texts": [
            {
                "text": "识别结果",
                "confidence": 0.95,
                "position": {
                    "x": 100,
                    "y": 200,
                    "width": 50,
                    "height": 30
                }
            }
        ],
        "type": 1
    }
}
```

#### 3.3 卡牌识别
```bash
curl -X POST http://localhost:8000/api/ocr/recognize_cards \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/screenshots/screenshot_1234567890.png"}'
```

响应示例：
```json
{
    "success": true,
    "message": "识别成功",
    "data": {
        "hand_cards": ["10s", "7s", "6h", "Am"],
        "public_cards": []
    }
}
```

#### 3.4 AI卡牌识别
```bash
curl -X POST http://localhost:8000/api/ai/recognize_cards \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/screenshots/screenshot_1234567890.png"}'
```

响应示例：
```json
{
    "success": true,
    "message": "识别成功",
    "data": {
        "predictions": [
            {
                "x": 345,
                "y": 824.5,
                "width": 32,
                "height": 41,
                "confidence": 0.759,
                "class": "2f",
                "class_id": 4,
                "detection_id": "cf360137-7578-48e7-9d95-14c488c04e78"
            }
        ],
        "hand_cards": ["10s", "7s", "6h", "Am"],
        "public_cards": []
    }
}
```

#### 3.5 颜色识别
```bash
curl -X POST http://localhost:8000/api/color/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/screenshots/screenshot_1234567890.png",
    "region": {"x": 151, "y": 97, "width": 49, "height": 40}
  }'
```

参数说明：
- `image_path`: 图片路径
- `region`: 区域坐标，包含 x, y, width, height

响应示例：
```json
{
    "success": true,
    "message": "识别成功",
    "data": {
        "region": {"x": 151, "y": 97, "width": 49, "height": 40},
        "color": {"b": 255, "g": 0, "r": 0, "hex": "#ff0000"}
    }
}
```

## 项目结构

```
.
├── data/
│   └── screenshots/      # 截图保存目录
├── src/
│   └── core/
│       ├── screen_all.py     # 全屏截图功能
│       └── screen_region.py  # 区域截图功能
├── docs/
│   └── api.md           # API文档
└── test_screen_region.py  # 测试脚本
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

## 项目结构

```
.
├── data/
│   └── screenshots/      # 截图保存目录
├── src/
│   ├── api/
│   │   └── routes.py     # API接口定义
│   └── core/
│       ├── adb.py        # ADB控制
│       ├── screen_all.py # 全屏截图
│       ├── ocr_all.py    # 全图OCR
│       ├── ocr_region.py # 区域OCR
│       └── copy_region.py # 区域截图
├── docs/
│   └── api.md           # API文档
└── main.py             # 主程序入口
```

## API 接口

### 设备管理

1. 获取设备列表
   - 路径: `/api/device/list`
   - 方法: POST
   - 请求: `{}`
   - 响应: `{"success": true, "data": {"devices": ["127.0.0.1:16384"]}}`

2. 获取当前设备信息
   - 路径: `/api/device/current`
   - 方法: POST
   - 请求: `{"device_id": "127.0.0.1:16384"}`
   - 响应: `{"success": true, "data": {"device_id": "127.0.0.1:16384", "status": "connected", "screen_size": {"width": 1920, "height": 1080}}}`

3. 连接设备
   - 路径: `/api/device/connect`
   - 方法: POST
   - 请求: `{"device_id": "127.0.0.1:16384"}`
   - 响应: `{"success": true, "data": {"device_id": "127.0.0.1:16384"}}`

4. 断开设备连接
   - 路径: `/api/device/disconnect`
   - 方法: POST
   - 请求: `{"device_id": "127.0.0.1:16384"}`
   - 响应: `{"success": true, "message": "设备断开连接成功"}`

### 设备操作

1. 点击操作
   - 路径: `/api/device/click`
   - 方法: POST
   - 请求: `{"device_id": "127.0.0.1:16384", "x": 100, "y": 200}`
   - 响应: `{"success": true, "data": {"device_id": "127.0.0.1:16384", "x": 100, "y": 200, "timestamp": 1234567890.123}}`

2. 截屏操作
   - 路径: `/api/device/screenshot`
   - 方法: POST
   - 请求: `{"device_id": "127.0.0.1:16384"}`
   - 响应: `{"success": true, "data": {"device_id": "127.0.0.1:16384", "path": "data/screenshots/screenshot_1234567890.png", "timestamp": 1234567890}}`

### OCR识别

1. 全图文字识别
   - 路径: `/api/ocr/recognize_all`
   - 方法: POST
   - 请求: `{"image_path": "data/screenshots/screenshot_1234567890.png"}`
   - 响应: `{"success": true, "data": {"texts": [{"text": "识别结果", "confidence": 0.95, "position": {"x": 100, "y": 200, "width": 50, "height": 30}}]}}`

2. 区域文字识别
   - 路径: `/api/ocr/recognize_region`
   - 方法: POST
   - 请求: `{"image_path": "data/screenshots/screenshot_1234567890.png", "region": [100, 200, 300, 400], "type": 1}`
   - 响应: `{"success": true, "data": {"texts": [{"text": "识别结果", "confidence": 0.95, "position": {"x": 100, "y": 200, "width": 50, "height": 30}}], "type": 1}}`

### 区域操作

1. 区域截图
   - 路径: `/api/region/copy`
   - 方法: POST
   - 请求: `{"image_path": "data/screenshots/screenshot_1234567890.png", "region": [100, 200, 300, 400], "type": "public"}`
   - 响应: `{"success": true, "data": {"path": "data/screenshots/public/public.png"}}`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python main.py
```

## 注意事项

1. 使用前请确保：
   - Android 设备已连接
   - ADB 已正确配置
   - 设备已授权调试

2. 区域坐标：
   - 坐标原点在屏幕左上角
   - 坐标单位为像素
   - 建议区域大小不超过屏幕分辨率

3. 文件保存：
   - 自动创建目录
   - 自动处理文件名
   - 支持自定义文件名

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

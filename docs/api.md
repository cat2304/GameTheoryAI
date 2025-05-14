# 屏幕区域截图 API 文档

## 概述

本文档描述了屏幕区域截图模块的 API 接口。该模块提供了对 Android 设备屏幕进行区域截图的功能。

## 区域类型

```python
class RegionType(Enum):
    PUBLIC = "public"  # 公牌区域
    HAND = "hand"      # 手牌区域
    OP = "op"          # 操作区域
    CUSTOM = "custom"  # 自定义区域
```

## API 接口

### 1. 初始化

```python
processor = ScreenRegionProcessor()
```

### 2. 区域截图

```python
def capture_region(
    device_id: str, 
    region_type: RegionType, 
    custom_region: Optional[Tuple[int, int, int, int]] = None,
    custom_name: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]
```

#### 参数说明

- `device_id`: 设备ID，例如 "127.0.0.1:16384"
- `region_type`: 区域类型，使用 RegionType 枚举
- `custom_region`: 自定义区域坐标，格式为 (x, y, width, height)
- `custom_name`: 自定义文件名（不包含扩展名）

#### 返回值

返回一个元组 (success, result)：
- `success`: 布尔值，表示操作是否成功
- `result`: 字典，包含以下字段：
  - 成功时：
    ```python
    {
        "success": True,
        "path": "保存路径",
        "region": {
            "x": x坐标,
            "y": y坐标,
            "width": 宽度,
            "height": 高度
        },
        "type": "区域类型"
    }
    ```
  - 失败时：
    ```python
    {
        "error": "错误信息"
    }
    ```

## 使用示例

### 1. 使用预定义区域

```python
# 公牌区域截图
success, result = processor.capture_region(device_id, RegionType.PUBLIC)

# 手牌区域截图
success, result = processor.capture_region(device_id, RegionType.HAND)

# 操作区域截图
success, result = processor.capture_region(device_id, RegionType.OP)
```

### 2. 使用自定义区域

```python
# 自定义区域截图
custom_region = (200, 300, 400, 500)  # (x, y, width, height)
success, result = processor.capture_region(
    device_id, 
    RegionType.CUSTOM,
    custom_region=custom_region,
    custom_name="my_custom_region"
)
```

## 保存路径

截图会按照以下结构保存：
```
data/screenshots/
├── public/
│   └── public.png
├── hand/
│   └── hand.png
├── op/
│   └── op.png
└── custom/
    └── my_custom_region.png
```

## 错误处理

1. 区域坐标无效：
   - 坐标不能为负数
   - 区域不能超出屏幕范围
   - 宽度和高度必须大于0

2. 图片质量检查：
   - 图片不能为空
   - 图片不能全黑或全白

3. 设备连接：
   - 确保设备已连接
   - 确保设备ID正确

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
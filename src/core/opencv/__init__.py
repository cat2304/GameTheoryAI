"""
OpenCV图像处理模块

提供图像处理和计算机视觉功能。

子模块
-----
- opencv_processor: 图像处理器
  - 图像预处理
  - 特征提取
  - 目标检测
- opencv_algorithm: 图像算法
  - 图像增强
  - 边缘检测
  - 模板匹配

主要功能
-------
1. 图像处理
   - 图像预处理
   - 特征提取
   - 目标检测
   - 图像增强

2. 计算机视觉
   - 边缘检测
   - 模板匹配
   - 特征匹配
   - 目标跟踪

使用示例
-------
```python
from src.core.opencv.opencv_processor import OpenCVProcessor
from src.core.opencv.opencv_algorithm import OpenCVAlgorithm

# 初始化图像处理器
processor = OpenCVProcessor()

# 加载图像
image = processor.load_image('screenshot.png')

# 预处理图像
processed = processor.preprocess(image)

# 使用算法处理
algorithm = OpenCVAlgorithm()
result = algorithm.detect_edges(processed)
```

注意事项
-------
1. 确保OpenCV正确安装
2. 图像格式要正确
3. 处理参数要合理
"""

# 版本信息
__version__ = '0.1.0'

# 模块级导入
from .opencv_processor import OpenCVProcessor
from .opencv_algorithm import OpenCVAlgorithm

# 公共接口列表
__all__ = [
    'OpenCVProcessor',
    'OpenCVAlgorithm'
] 
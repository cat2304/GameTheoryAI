"""
任务管理模块

提供任务调度和执行功能。

子模块
-----
- ocr_task: OCR识别任务
  - 图像预处理
  - 文字识别
  - 结果处理

主要功能
-------
1. 任务调度
   - 任务创建
   - 任务优先级
   - 任务队列
   - 任务取消

2. 任务执行
   - 并行执行
   - 错误处理
   - 结果收集
   - 状态监控

使用示例
-------
```python
from src.core.task.ocr_task import OCRTask

# 创建OCR任务
task = OCRTask(image_path='screenshot.png')

# 执行任务
result = task.execute()

# 获取结果
text = result.get_text()
```

注意事项
-------
1. 任务执行需要合理配置资源
2. 注意任务优先级设置
3. 及时处理任务异常
"""

# 版本信息
__version__ = '0.1.0'

# 模块级导入
from .ocr_task import OCRTask

# 公共接口列表
__all__ = [
    'OCRTask'
] 
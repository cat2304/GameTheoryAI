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

注意事项
-------
1. 确保OpenCV正确安装
2. 图像格式要正确
3. 处理参数要合理
"""

# 版本信息
__version__ = '0.1.0'

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .opencv_algorithm import OpenCVAlgorithm

# 公共接口列表
__all__ = [
    'OpenCVProcessor',
    'OpenCVAlgorithm'
]

class OpenCVProcessor:
    """OpenCV处理器，用于处理图像和识别游戏元素"""
    
    def __init__(self):
        """初始化OpenCV处理器"""
        self.logger = logging.getLogger(__name__)
        self.opencv_engine = OpenCVAlgorithm()
    
    def process(self, image_path):
        """处理图片并识别游戏元素"""
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 图像预处理
            processed_img = self.opencv_engine.preprocess_image(image)
            
            # 查找游戏元素区域
            elements, detection_result = self.opencv_engine.find_game_elements(processed_img, image)
            
            # 识别每个游戏元素
            predictions = []
            for element in elements:
                result = self.opencv_engine.recognize_game_element(image, element['region'])
                element['result'] = result
                predictions.append(result)
            
            # 绘制可视化结果
            visualization = self.opencv_engine.draw_visualization(image, elements, predictions)
            
            self.logger.info(f"识别完成，共找到 {len(elements)} 个游戏元素")
            
            return {
                'success': True,
                'original_image': image,
                'processed_image': processed_img,
                'detection_result': detection_result,
                'visualization': visualization,
                'elements': elements,
                'elements_count': len(elements)
            }
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_results(self, results, output_dir="output"):
        """保存处理结果"""
        try:
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存处理结果
            cv2.imwrite(str(output_path / "original.png"), results['original_image'])
            cv2.imwrite(str(output_path / "processed.png"), results['processed_image'])
            cv2.imwrite(str(output_path / "detection.png"), results['detection_result'])
            cv2.imwrite(str(output_path / "visualization.png"), results['visualization'])
            
            # 保存识别结果文本
            with open(output_path / "results.txt", "w") as f:
                f.write(f"识别到 {results['elements_count']} 个游戏元素\n\n")
                for i, element in enumerate(results['elements']):
                    x, y, w, h = element['region']
                    f.write(f"元素 {i+1}:\n")
                    f.write(f"  位置: ({x}, {y}, {w}, {h})\n")
                    f.write(f"  识别结果: {element['result']}\n")
                    f.write(f"  面积: {element['area']}\n")
                    f.write(f"  宽高比: {element['aspect_ratio']:.2f}\n\n")
            
            self.logger.info(f"结果已保存到目录: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
            return False

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建处理器并处理图片
    processor = OpenCVProcessor()
    result = processor.process('game_screenshot.png')
    
    if result['success']:
        processor.save_results(result) 
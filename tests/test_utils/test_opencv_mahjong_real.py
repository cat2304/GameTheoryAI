"""
麻将牌识别实际测试模块
=================

使用真实麻将牌图片测试识别效果。
"""

import os
import unittest
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from src.utils.log_utils import get_logger
from src.core.opencv.opencv_processor import OpenCVProcessor

class TestOpenCVMahjongReal(unittest.TestCase):
    """麻将牌识别实际测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 设置图片目录
        self.image_dir = Path("/Users/mac/ai/temp/img/20250405")
        self.output_dir = Path("tests/temp/recognition_results")
        self.result_dir = Path("/Users/mac/ai/temp/result")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 创建OpenCV实例
        self.processor = OpenCVProcessor()
        
        # 获取所有图片文件
        self.image_files = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        
        # 获取日志记录器
        self.logger = get_logger("opencv_test")
        
        # 检查图片目录是否存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.image_dir}")
        
        # 检查是否有图片文件
        if not self.image_files:
            raise FileNotFoundError(f"在目录 {self.image_dir} 中没有找到图片文件")
        
        self.logger.info(f"找到 {len(self.image_files)} 张图片")
    
    def save_recognition_result(self, image_path: Path, result: dict):
        """保存识别结果到文件
        
        Args:
            image_path: 图片路径
            result: 识别结果
        """
        # 生成结果文件名
        result_file = self.result_dir / f"{image_path.stem}_.txt"
        
        # 准备结果内容
        content = [
            f"图片: {image_path.name}",
            f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"识别状态: {'成功' if result['success'] else '失败'}",
            f"检测到的麻将牌数量: {result.get('total_tiles', 0)}",
            "\n识别结果:"
        ]
        
        # 添加每个麻将牌的识别结果
        if 'tiles' in result and result['tiles']:
            for tile in result['tiles']:
                content.append(f"  位置: {tile['region']}")
                content.append(f"  面积: {tile['area']}")
                content.append(f"  宽高比: {tile['aspect_ratio']:.2f}")
                content.append(f"  识别结果: {tile.get('result', '未知')}")
                content.append("")
        else:
            content.append("  未识别到任何麻将牌")
            content.append("")
        
        # 添加调试信息
        content.append("调试信息:")
        content.append(f"  图片尺寸: {result.get('image_size', '未知')}")
        content.append(f"  预处理状态: {result.get('preprocess_status', '未知')}")
        content.append(f"  错误信息: {result.get('error', '无')}")
        
        # 写入文件
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        self.logger.info(f"保存识别结果到: {result_file}")
    
    def test_recognize_all_images(self):
        """测试识别所有图片"""
        total_images = len(self.image_files)
        self.logger.info(f"开始测试 {total_images} 张图片的识别效果")
        
        for idx, image_path in enumerate(self.image_files, 1):
            self.logger.info(f"处理第 {idx}/{total_images} 张图片: {image_path.name}")
            
            try:
                # 读取原始图片
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"无法读取图片: {image_path}")
                
                # 记录图片尺寸
                image_size = f"{image.shape[1]}x{image.shape[0]}"
                self.logger.info(f"图片尺寸: {image_size}")
                
                # 识别图片
                result = self.processor.process(str(image_path))
                
                # 添加调试信息
                result['image_size'] = image_size
                
                # 验证结果
                self.assertTrue(result['success'], f"图片 {image_path.name} 识别失败")
                self.assertIn('tiles', result, f"图片 {image_path.name} 结果中缺少 tiles 字段")
                
                # 验证检测到的麻将牌数量
                tile_count = len(result['tiles'])
                self.assertGreaterEqual(tile_count, 10, f"检测到的麻将牌数量过少: {tile_count}")
                self.assertLessEqual(tile_count, 15, f"检测到的麻将牌数量过多: {tile_count}")
                
                # 保存识别结果
                self.save_recognition_result(image_path, result)
                
                # 可视化结果
                vis_image = self.processor.opencv_engine.visualize_results(image, result['tiles'])
                
                # 保存可视化结果
                output_path = self.output_dir / f"result_{image_path.name}"
                cv2.imwrite(str(output_path), vis_image)
                self.logger.info(f"保存可视化结果到: {output_path}")
                
                # 打印识别结果
                self.logger.info(f"图片 {image_path.name} 识别结果:")
                if result['tiles']:
                    for tile in result['tiles']:
                        self.logger.info(f"  位置: {tile['region']}, 面积: {tile['area']}, "
                                       f"宽高比: {tile['aspect_ratio']:.2f}, "
                                       f"识别结果: {tile.get('result', '未知')}")
                else:
                    self.logger.warning(f"  未识别到任何麻将牌")
                
            except Exception as e:
                self.logger.error(f"处理图片 {image_path.name} 时出错: {str(e)}")
                # 保存错误信息
                error_result = {
                    'success': False,
                    'error': str(e),
                    'image_size': image_size if 'image_size' in locals() else '未知'
                }
                self.save_recognition_result(image_path, error_result)
                raise
    
    def test_analyze_detection_accuracy(self):
        """分析检测准确率"""
        total_images = len(self.image_files)
        successful_detections = 0
        total_tiles = 0
        
        for image_path in self.image_files:
            try:
                result = self.processor.process(str(image_path))
                if result['success']:
                    tile_count = len(result['tiles'])
                    if 10 <= tile_count <= 15:  # 正常麻将牌数量范围
                        successful_detections += 1
                    total_tiles += tile_count
            except Exception as e:
                self.logger.error(f"分析图片 {image_path.name} 时出错: {str(e)}")
        
        if total_images > 0:
            detection_rate = successful_detections / total_images * 100
            avg_tiles = total_tiles / total_images
            
            self.logger.info(f"检测准确率分析:")
            self.logger.info(f"  总图片数: {total_images}")
            self.logger.info(f"  成功检测数: {successful_detections}")
            self.logger.info(f"  检测率: {detection_rate:.2f}%")
            self.logger.info(f"  平均每图检测到的麻将牌数: {avg_tiles:.1f}")
        else:
            self.logger.warning("没有找到可分析的图片")

if __name__ == '__main__':
    unittest.main() 
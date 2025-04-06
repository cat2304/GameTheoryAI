"""
麻将牌识别测试模块
===============

测试麻将牌识别功能。
"""

import os
import unittest
import cv2
import numpy as np
from pathlib import Path

from src.utils.mahjong_ocr import MahjongOCR

class TestMahjongOCR(unittest.TestCase):
    """麻将牌识别测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试用的临时目录
        self.test_dir = Path("tests/temp")
        self.test_dir.mkdir(exist_ok=True)
        
        # 创建测试图片
        self.test_image = self.test_dir / "test.png"
        self._create_test_image()
        
        # 创建OCR实例
        self.ocr = MahjongOCR()
    
    def tearDown(self):
        """测试后清理"""
        # 删除测试文件
        if self.test_image.exists():
            self.test_image.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def _create_test_image(self):
        """创建测试图片"""
        # 创建一个空白图片
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # 绘制一个模拟的麻将牌
        # 1. 绘制牌面背景
        cv2.rectangle(image, (50, 150), (150, 200), (255, 255, 255), -1)
        
        # 2. 绘制数字
        cv2.putText(image, "一", (80, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # 3. 绘制花色
        cv2.putText(image, "万", (80, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存图片
        cv2.imwrite(str(self.test_image), image)
    
    def test_preprocess_image(self):
        """测试图片预处理"""
        # 读取测试图片
        image = cv2.imread(str(self.test_image))
        
        # 预处理
        processed = self.ocr.preprocess_image(image)
        
        # 验证结果
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape[:2], image.shape[:2])
    
    def test_find_mahjong_tiles(self):
        """测试查找麻将牌区域"""
        # 读取测试图片
        image = cv2.imread(str(self.test_image))
        
        # 预处理
        processed = self.ocr.preprocess_image(image)
        
        # 查找麻将牌区域
        tiles = self.ocr.find_mahjong_tiles(processed)
        
        # 验证结果
        self.assertGreater(len(tiles), 0)
        for tile in tiles:
            self.assertEqual(len(tile), 4)  # x, y, w, h
    
    def test_extract_tile_features(self):
        """测试提取麻将牌特征"""
        # 读取测试图片
        image = cv2.imread(str(self.test_image))
        
        # 定义测试区域（模拟的麻将牌区域）
        region = (50, 150, 100, 50)
        
        # 提取特征
        features = self.ocr.extract_tile_features(image, region)
        
        # 验证结果
        self.assertIn('color_features', features)
        self.assertIn('shape_features', features)
    
    def test_recognize_tile(self):
        """测试识别单个麻将牌"""
        # 读取测试图片
        image = cv2.imread(str(self.test_image))
        
        # 识别麻将牌
        result = self.ocr.recognize_tile(image, (50, 150, 100, 50))
        
        # 验证结果
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_recognize_image(self):
        """测试识别整张图片"""
        # 识别图片
        result = self.ocr.recognize_image(str(self.test_image))
        
        # 验证结果
        self.assertTrue(result['success'])
        self.assertIn('tiles', result)
        self.assertGreater(len(result['tiles']), 0)
    
    def test_visualize_results(self):
        """测试可视化结果"""
        # 读取测试图片
        image = cv2.imread(str(self.test_image))
        
        # 识别图片
        result = self.ocr.recognize_image(str(self.test_image))
        
        # 可视化结果
        vis_image = self.ocr.visualize_results(image, result['tiles'])
        
        # 验证结果
        self.assertIsNotNone(vis_image)
        self.assertEqual(vis_image.shape, image.shape)

if __name__ == '__main__':
    unittest.main() 
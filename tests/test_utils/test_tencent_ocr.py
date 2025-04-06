"""
麻将牌识别测试模块
=================

测试麻将牌识别功能。
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.mahjong_ocr import MahjongOCR

class TestMahjongOCR(unittest.TestCase):
    """麻将牌识别测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.ocr = MahjongOCR()
        self.test_dir = Path("tests/temp/ocr")
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()
    
    @patch('src.utils.mahjong_ocr.MahjongOCR.recognize_image')
    def test_recognize_single_image(self, mock_recognize):
        """测试单图片识别"""
        # 准备测试数据
        test_image = self.test_dir / "test.png"
        test_image.touch()
        
        # 设置模拟返回值
        mock_recognize.return_value = {
            'success': True,
            'tiles': [
                {'region': (100, 100, 50, 50), 'result': '一万'},
                {'region': (200, 100, 50, 50), 'result': '二万'}
            ]
        }
        
        # 执行识别
        result = self.ocr.recognize_image(str(test_image))
        
        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(len(result['tiles']), 2)
        self.assertEqual(result['tiles'][0]['result'], '一万')
        self.assertEqual(result['tiles'][1]['result'], '二万')
    
    @patch('src.utils.mahjong_ocr.MahjongOCR.recognize_image')
    def test_recognize_directory(self, mock_recognize):
        """测试目录批量识别"""
        # 准备测试数据
        test_images = [self.test_dir / f"test{i}.png" for i in range(3)]
        for img in test_images:
            img.touch()
        
        # 设置模拟返回值
        mock_recognize.return_value = {
            'success': True,
            'tiles': [{'region': (100, 100, 50, 50), 'result': '一万'}]
        }
        
        # 执行识别
        results = self.ocr.recognize_directory(str(self.test_dir))
        
        # 验证结果
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])
            self.assertEqual(len(result['tiles']), 1)
    
    def test_recognize_nonexistent_image(self):
        """测试识别不存在的图片"""
        with self.assertRaises(FileNotFoundError):
            self.ocr.recognize_image("nonexistent.png")
    
    def test_recognize_nonexistent_directory(self):
        """测试识别不存在的目录"""
        with self.assertRaises(NotADirectoryError):
            self.ocr.recognize_directory("nonexistent_dir")
    
    @patch('src.utils.mahjong_ocr.MahjongOCR.recognize_image')
    def test_error_handling(self, mock_recognize):
        """测试错误处理"""
        # 准备测试数据
        test_image = self.test_dir / "test.png"
        test_image.touch()
        
        # 设置模拟异常
        mock_recognize.side_effect = Exception("测试错误")
        
        # 执行识别
        result = self.ocr.recognize_image(str(test_image))
        
        # 验证结果
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error'], "测试错误")

if __name__ == '__main__':
    unittest.main() 
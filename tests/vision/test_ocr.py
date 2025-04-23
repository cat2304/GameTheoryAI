import unittest
import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.vision.ocr import PokerOCR

class TestPokerOCR(unittest.TestCase):
    def setUp(self):
        self.ocr = PokerOCR()
        self.test_image_path = "data/templates/5.png"
    
    def test_get_regions(self):
        """测试区域获取"""
        regions = self.ocr.get_regions(1920, 1080)
        self.assertIn("PUBLIC_REGION", regions)
        self.assertIn("HAND_REGION", regions)
        self.assertIn("CLICK_REGION", regions)
        
        # 检查区域坐标
        pub_region = regions["PUBLIC_REGION"]
        self.assertEqual(len(pub_region), 4)
        self.assertTrue(all(isinstance(x, int) for x in pub_region))
    
    def test_preprocess_card(self):
        """测试牌面预处理"""
        img = cv2.imread(self.test_image_path)
        if img is None:
            self.fail(f"无法读取测试图片: {self.test_image_path}")
        processed = self.ocr.preprocess_card(img)
        self.assertEqual(processed.shape[:2], (300, 300))
    
    def test_extract_cards(self):
        """测试牌面提取"""
        # 模拟 OCR 结果
        test_lines = [
            ([[0, 0], [100, 0], [100, 100], [0, 100]], ("A", 0.9)),
            ([[200, 0], [300, 0], [300, 100], [200, 100]], ("K", 0.8))
        ]
        cards = self.ocr.extract_cards(test_lines)
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0][0], "A")
        self.assertEqual(cards[1][0], "K")
    
    def test_is_action(self):
        """测试动作判断"""
        self.assertTrue(self.ocr.is_action("弃牌"))
        self.assertTrue(self.ocr.is_action("加注"))
        self.assertTrue(self.ocr.is_action("让牌"))
        self.assertTrue(self.ocr.is_action("跟注"))
        self.assertFalse(self.ocr.is_action("其他"))
    
    def test_recognize_cards(self):
        """测试牌面识别"""
        result = self.ocr.recognize_cards(self.test_image_path)
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("publicCards", result)
        self.assertIn("handCards", result)
        self.assertIn("actions", result)

if __name__ == '__main__':
    unittest.main() 
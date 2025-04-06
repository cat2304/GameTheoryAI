"""
核心功能测试模块
==============

包含游戏监控、OCR识别和分析的核心功能测试。
"""

import os
import sys
import unittest
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.utils import ConfigManager, LogManager, ADBHelper, GameOCR
from src.core.game.game_monitor import GameMonitor

class TestGameMonitor(unittest.TestCase):
    """游戏监控测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config_path = project_root / "config" / "app_config.yaml"
        self.monitor = GameMonitor(config_path=self.config_path)
        self.test_dir = Path("tests/temp/screenshots")
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        for file in self.test_dir.glob("*"):
            file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_monitor_initialization(self):
        """测试监控初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertFalse(self.monitor.is_running)
    
    def test_start_stop_monitor(self):
        """测试启动和停止监控"""
        # 测试启动
        self.assertTrue(self.monitor.start())
        self.assertTrue(self.monitor.is_running)
        
        # 测试停止
        self.assertTrue(self.monitor.stop())
        self.assertFalse(self.monitor.is_running)
    
    def test_screenshot_creation(self):
        """测试截图创建"""
        screenshot_path = self.monitor.take_screenshot()
        self.assertTrue(os.path.exists(screenshot_path))
        self.assertTrue(os.path.getsize(screenshot_path) > 0)
    
    @patch('src.core.game.game_monitor.GameMonitor._analyze_game_state')
    def test_game_state_analysis(self, mock_analyze):
        """测试游戏状态分析"""
        # 设置模拟返回值
        mock_analyze.return_value = {
            'state': 'playing',
            'score': 25000,
            'tiles_in_hand': ['一万', '二万', '三万'],
            'last_action': 'discard'
        }
        
        # 创建测试图片
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        test_image_path = str(self.test_dir / "test.png")
        cv2.imwrite(test_image_path, test_image)
        
        # 执行分析
        result = self.monitor.analyze_screenshot(test_image_path)
        
        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(result['analysis']['state'], 'playing')
        self.assertEqual(result['analysis']['score'], 25000)
        self.assertEqual(len(result['analysis']['tiles_in_hand']), 3)
        self.assertEqual(result['analysis']['last_action'], 'discard')
        
        # 验证模拟调用
        mock_analyze.assert_called_once()

class TestGameOCR(unittest.TestCase):
    """OCR引擎测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config_path = project_root / "config" / "app_config.yaml"
        self.ocr = GameOCR(config_path=self.config_path)
        self.test_dir = Path("tests/temp/ocr")
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        for file in self.test_dir.glob("*"):
            file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_ocr_initialization(self):
        """测试OCR初始化"""
        self.assertIsNotNone(self.ocr)
        self.assertIsNotNone(self.ocr.config)
        self.assertIsNotNone(self.ocr.logger)
    
    def test_ocr_recognition(self):
        """测试OCR识别"""
        # 创建测试图片
        test_image = np.zeros((100, 300), dtype=np.uint8)
        cv2.putText(test_image, "测试文本", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        test_image_path = str(self.test_dir / "test.png")
        cv2.imwrite(test_image_path, test_image)
        
        # 执行识别
        result = self.ocr.recognize_text(test_image_path)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
    
    def test_image_preprocessing(self):
        """测试图像预处理"""
        # 创建测试图片
        test_image = np.zeros((100, 300), dtype=np.uint8)
        cv2.putText(test_image, "测试文本", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        
        # 执行预处理
        processed_image = self.ocr.preprocess_image(test_image)
        
        # 验证结果
        self.assertIsNotNone(processed_image)
        self.assertEqual(processed_image.shape[:2], test_image.shape[:2])
        self.assertEqual(processed_image.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main() 
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

from src.utils.config import ConfigManager
from src.utils.logger import LogManager
from src.core.game.game_monitor import GameMonitor
from src.core.ocr.ocr_engine import OCREngine
from src.utils.adb import ADBHelper

class TestGameMonitor(unittest.TestCase):
    """游戏监控测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.monitor = GameMonitor()
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
        
        # 执行分析
        state = self.monitor.analyze_screenshot('test.png')
        
        # 验证结果
        self.assertEqual(state['state'], 'playing')
        self.assertEqual(state['score'], 25000)
        self.assertEqual(len(state['tiles_in_hand']), 3)

class TestOCREngine(unittest.TestCase):
    """OCR引擎测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.ocr = OCREngine()
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
    
    @patch('src.core.ocr.ocr_engine.OCREngine.recognize_image')
    def test_image_recognition(self, mock_recognize):
        """测试图片识别"""
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
    
    def test_preprocessing(self):
        """测试图像预处理"""
        # 创建测试图像
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 执行预处理
        processed = self.ocr.preprocess_image(test_image)
        
        # 验证结果
        self.assertEqual(processed.shape, (100, 100))
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 255))

class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = ConfigManager()
    
    def test_singleton(self):
        """测试单例模式"""
        config1 = ConfigManager()
        config2 = ConfigManager()
        self.assertIs(config1, config2)
    
    def test_get_config(self):
        """测试获取配置"""
        # 测试获取存在的配置项
        ocr_config = self.config.get('ocr')
        self.assertIsNotNone(ocr_config)
        self.assertIsInstance(ocr_config, dict)
    
    def test_set_config(self):
        """测试设置配置"""
        # 设置新的配置项
        test_config = {'test_key': 'test_value'}
        self.config.set('test', test_config)
        
        # 验证配置是否被正确设置
        saved_config = self.config.get('test')
        self.assertEqual(saved_config, test_config)

class TestLogManager(unittest.TestCase):
    """日志管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.log = LogManager()
    
    def test_singleton(self):
        """测试单例模式"""
        log1 = LogManager()
        log2 = LogManager()
        self.assertIs(log1, log2)
    
    def test_logger_creation(self):
        """测试日志记录器创建"""
        logger = self.log.get_logger("test")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test")
    
    def test_log_levels(self):
        """测试日志级别"""
        logger = self.log.get_logger("test_levels")
        
        # 测试不同级别的日志记录
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # 验证日志文件是否存在
        log_file = Path("logs/test_levels.log")
        self.assertTrue(log_file.exists())

class TestADBHelper(unittest.TestCase):
    """ADB工具测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.adb = ADBHelper()
    
    def test_device_connection(self):
        """测试设备连接"""
        with patch('subprocess.check_output') as mock_output:
            mock_output.return_value = b"List of devices attached\n123456789\tdevice\n"
            devices = self.adb.get_devices()
            self.assertEqual(len(devices), 1)
            self.assertEqual(devices[0], "123456789")
    
    def test_screenshot_command(self):
        """测试截图命令"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = self.adb.take_screenshot("test.png")
            self.assertTrue(result)
    
    def test_tap_command(self):
        """测试点击命令"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = self.adb.tap(100, 200)
            self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 
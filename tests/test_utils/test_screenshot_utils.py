import os
import unittest
import psutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.auto_screenshot import AutoScreenshot

class TestAutoScreenshot(unittest.TestCase):
    def setUp(self):
        self.screenshot = AutoScreenshot()
        self.test_dir = Path("tests/temp/screenshots")
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        # 清理测试目录
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()
    
    @patch('psutil.process_iter')
    def test_find_game_process(self, mock_process_iter):
        # 模拟进程列表
        mock_process = MagicMock()
        mock_process.name.return_value = "TencentMahjong.exe"
        mock_process.pid = 12345
        mock_process_iter.return_value = [mock_process]
        
        # 测试查找游戏进程
        pid = self.screenshot._check_device()
        self.assertTrue(pid)
    
    @patch('psutil.process_iter')
    def test_find_game_process_not_found(self, mock_process_iter):
        # 模拟没有找到游戏进程
        mock_process = MagicMock()
        mock_process.name.return_value = "other.exe"
        mock_process_iter.return_value = [mock_process]
        
        # 测试查找游戏进程
        pid = self.screenshot._check_device()
        self.assertFalse(pid)
    
    def test_start_and_stop(self):
        # 测试启动和停止任务
        self.assertTrue(self.screenshot.start())
        self.assertTrue(self.screenshot.is_running)
        
        self.assertTrue(self.screenshot.stop())
        self.assertFalse(self.screenshot.is_running)
    
    def test_start_already_running(self):
        # 测试重复启动
        self.screenshot.start()
        self.assertFalse(self.screenshot.start())
    
    def test_stop_not_running(self):
        # 测试停止未运行的任务
        self.assertFalse(self.screenshot.stop())

if __name__ == '__main__':
    unittest.main() 
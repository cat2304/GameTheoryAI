import unittest
import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.vision.ocr import PokerOCR

class TestScreenCapture(unittest.TestCase):
    def setUp(self):
        self.screen_capture = ScreenCapture()
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.screen_capture.device_id)
        self.assertIsNotNone(self.screen_capture.adb_path)
        self.assertTrue(os.path.exists(self.screen_capture.output_dir))
    
    def test_take_screenshot(self):
        """测试截图功能"""
        success, result = self.screen_capture.take_screenshot()
        self.assertIsInstance(success, bool)
        if success:
            self.assertIsInstance(result, str)
            self.assertTrue(os.path.exists(result))
            # 检查文件大小
            self.assertGreater(os.path.getsize(result), 0)
    
    def test_start_capture_loop(self):
        """测试循环截图"""
        # 由于这是一个长时间运行的测试，我们只运行很短的时间
        def stop_after_2_seconds():
            time.sleep(2)
            raise KeyboardInterrupt()
        
        # 使用线程来停止循环
        import threading
        stop_thread = threading.Thread(target=stop_after_2_seconds)
        stop_thread.start()
        
        # 运行循环
        self.screen_capture.start_capture_loop(interval=0.5)
        
        # 等待线程结束
        stop_thread.join()
        
        # 检查是否生成了截图
        files = os.listdir(self.screen_capture.output_dir)
        self.assertGreater(len(files), 0)
    
    def tearDown(self):
        # 清理测试文件
        if os.path.exists(self.screen_capture.output_dir):
            for file in os.listdir(self.screen_capture.output_dir):
                if file.startswith("screen_"):
                    os.remove(os.path.join(self.screen_capture.output_dir, file))

if __name__ == '__main__':
    unittest.main() 
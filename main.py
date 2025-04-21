#!/usr/bin/env python3
import time
import logging
from typing import Dict, Any
from adb import ScrcpyScreenshot
from poker_ocr import DualChannelPokerOCR
import cv2
import numpy as np

class PokerGameMonitor:
    def __init__(self):
        # 初始化截图工具
        self.screenshot = ScrcpyScreenshot()
        
        # 初始化OCR工具
        self.ocr = DualChannelPokerOCR()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 游戏状态
        self.last_hand_cards = []
        self.last_public_cards = []

    def process_screenshot(self, image_path: str) -> Dict[str, Any]:
        """处理截图并返回识别结果"""
        try:
            # 使用 OCR 识别扑克牌
            result = self.ocr.recognize_cards(image_path)
            
            # 检查是否有新的牌
            hand_changed = result["handCards"] != self.last_hand_cards
            public_changed = result["publicCards"] != self.last_public_cards
            
            if hand_changed or public_changed:
                self.logger.info(f"手牌变化: {self.last_hand_cards} -> {result['handCards']}")
                self.logger.info(f"公牌变化: {self.last_public_cards} -> {result['publicCards']}")
                
                # 更新状态
                self.last_hand_cards = result["handCards"]
                self.last_public_cards = result["publicCards"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理截图失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def run(self):
        """运行游戏监控"""
        try:
            while True:
                # 获取截图
                success, image_path = self.screenshot.take_screenshot()
                if not success:
                    self.logger.error(f"截图失败: {image_path}")
                    time.sleep(1)
                    continue
                
                # 识别扑克牌
                result = self.ocr.recognize(image_path)
                self.logger.info(f"识别结果: {result}")
                
                # 等待一段时间
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("程序已停止")
        except Exception as e:
            self.logger.error(f"发生错误: {str(e)}")
        finally:
            self.screenshot._stop_scrcpy()

def main():
    monitor = PokerGameMonitor()
    monitor.run()

if __name__ == "__main__":
    main()

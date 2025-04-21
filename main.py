#!/usr/bin/env python3
import time
import logging
from typing import Dict, Any
from adb import ScrcpyScreenshot
from poker_ocr import PokerOCR

class PokerGameMonitor:
    def __init__(self):
        # 初始化截图工具
        self.screenshot = ScrcpyScreenshot()
        
        # 初始化 OCR 工具
        self.ocr = PokerOCR()
        
        # 配置日志
        self.logger = logging.getLogger("PokerGameMonitor")
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)
        
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

    def start_monitoring(self, interval: int = 5):
        """开始监控游戏"""
        self.logger.info("开始监控扑克游戏...")
        try:
            while True:
                # 获取截图
                success, image_path = self.screenshot.take_screenshot()
                if not success:
                    self.logger.error(f"截图失败: {image_path}")
                    continue
                
                # 处理截图
                result = self.process_screenshot(image_path)
                if not result["success"]:
                    self.logger.error(f"识别失败: {result.get('error', '未知错误')}")
                    continue
                
                # 等待下一次截图
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("用户中断监控")
        except Exception as e:
            self.logger.error(f"监控过程发生错误: {str(e)}")
        finally:
            self.screenshot._stop_scrcpy()
            self.logger.info("监控结束")

def main():
    monitor = PokerGameMonitor()
    monitor.start_monitoring()

if __name__ == "__main__":
    main()

import time
import logging
from typing import Dict, Any
from src.vision.screen import ScreenCapture
from src.vision.ocr import recognize_cards
from src.core.game_state import GameState, GameRound
from src.core.game_maker import GameMaker
from src.core.game_executor import GameExecutor
from src.core.game_click import GameClicker

class GameController:
    def __init__(self, screen_capture: ScreenCapture = None):
        # 初始化各个模块
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏控制器...")
        
        # 使用传入的screen_capture或创建新的实例
        self.screen_capture = screen_capture if screen_capture is not None else ScreenCapture()
        self.logger.info("屏幕捕获模块初始化完成")
        
        self.game_state = GameState()
        self.logger.info("游戏状态模块初始化完成")
        
        self.game_maker = GameMaker()
        self.logger.info("游戏决策模块初始化完成")
        
        self.game_executor = GameExecutor()
        self.logger.info("游戏执行模块初始化完成")
        
        self.game_clicker = GameClicker()
        self.logger.info("游戏点击模块初始化完成")
        
        # 游戏状态
        self.last_hand_cards = []
        self.last_public_cards = []
        self.last_action = None
        self.logger.info("游戏控制器初始化完成")

    def process_screenshot(self, image_path: str) -> Dict[str, Any]:
        """处理截图并返回识别结果"""
        try:
            self.logger.info(f"开始处理截图: {image_path}")
            
            # 使用 OCR 识别扑克牌
            self.logger.info("开始OCR识别...")
            result = recognize_cards(image_path)
            
            if not result["success"]:
                self.logger.error(f"识别失败: {result['error']}")
                return result
            
            self.logger.info("OCR识别完成")
            
            # 检查是否有新的牌
            hand_changed = result["handCards"] != self.last_hand_cards
            public_changed = result["publicCards"] != self.last_public_cards
            
            if hand_changed or public_changed:
                self.logger.info(f"检测到牌面变化:")
                self.logger.info(f"手牌变化: {self.last_hand_cards} -> {result['handCards']}")
                self.logger.info(f"公牌变化: {self.last_public_cards} -> {result['publicCards']}")
                
                # 更新状态
                self.last_hand_cards = result["handCards"]
                self.last_public_cards = result["publicCards"]
                
                # 更新游戏状态
                self.logger.info("更新游戏状态...")
                self.game_state.update_cards(
                    [card["action"] for card in result["handCards"]],
                    [card["action"] for card in result["publicCards"]]
                )
                
                # 开始决策
                self.logger.info("开始决策过程...")
                decision = self.game_maker.make_decision(self.game_state)
                self.logger.info(f"决策结果: {decision}")
                
                # 执行决策
                if decision and decision != self.last_action:
                    self.logger.info(f"执行决策: {decision}")
                    if self.game_executor.execute_decision(decision):
                        self.last_action = decision
                else:
                    self.logger.info("无需执行新的决策")
            else:
                self.logger.info("牌面未发生变化")
            
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
            self.logger.info("开始游戏监控...")
            while True:
                # 获取截图
                self.logger.info("开始获取截图...")
                success, image_path = self.screen_capture.take_screenshot()
                if not success:
                    self.logger.error(f"截图失败: {image_path}")
                    time.sleep(5)  # 失败后等待5秒
                    continue
                
                self.logger.info(f"截图成功: {image_path}")
                
                # 处理截图并获取结果
                result = self.process_screenshot(image_path)
                if not result["success"]:
                    self.logger.error(f"识别失败: {result['error']}")
                    time.sleep(5)  # 识别失败后等待5秒
                    continue
                
                self.logger.info(f"识别结果: {result}")
                
                # 等待5秒后再次截图
                self.logger.info("等待5秒后继续...")
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("程序已停止")
        except Exception as e:
            self.logger.error(f"发生错误: {str(e)}")
        finally:
            self.logger.info("清理资源...") 
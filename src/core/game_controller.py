import time
import logging
from typing import Dict, Any
from src.vision.screen import ScreenCapture
from src.vision.ocr import recognize_cards
from src.control.click import PokerClicker
from src.core.game_state import GameState, GameRound

class GameController:
    def __init__(self, screen_capture: ScreenCapture = None):
        # 初始化各个模块
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏控制器...")
        
        # 使用传入的screen_capture或创建新的实例
        self.screen_capture = screen_capture if screen_capture is not None else ScreenCapture()
        self.logger.info("屏幕捕获模块初始化完成")
        
        self.clicker = PokerClicker()
        self.logger.info("点击控制模块初始化完成")
        
        self.game_state = GameState()
        self.logger.info("游戏状态模块初始化完成")
        
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
                decision = self.make_decision()
                self.logger.info(f"决策结果: {decision}")
                
                # 执行决策
                if decision and decision != self.last_action:
                    self.logger.info(f"执行决策: {decision}")
                    #self.execute_decision(decision)
                    # self.last_action = decision
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

    def make_decision(self) -> str:
        """根据当前游戏状态做出决策"""
        try:
            # 获取当前游戏状态
            current_round = self.game_state.current_round
            hand_cards = self.game_state.hand_cards
            public_cards = self.game_state.public_cards
            
            self.logger.info(f"当前游戏状态:")
            self.logger.info(f"回合: {current_round}")
            self.logger.info(f"手牌: {hand_cards}")
            self.logger.info(f"公牌: {public_cards}")
            
            # 根据游戏状态做出决策
            if not hand_cards:
                self.logger.warning("没有手牌，无法做出决策")
                return None
                
            if current_round == GameRound.PREFLOP:
                # 翻牌前策略
                if len(hand_cards) == 2:
                    # 检查是否是对子
                    if hand_cards[0] == hand_cards[1]:
                        self.logger.info("检测到对子，建议加注")
                        return "raise"
                    # 检查是否是大牌
                    if hand_cards[0] in ["A", "K", "Q"] and hand_cards[1] in ["A", "K", "Q"]:
                        self.logger.info("检测到大牌，建议加注")
                        return "raise"
                    self.logger.info("手牌一般，建议跟注")
                    return "call"
            
            elif current_round == GameRound.FLOP:
                # 翻牌策略
                if len(public_cards) >= 3:
                    # 检查是否形成对子
                    if any(card in hand_cards for card in public_cards):
                        self.logger.info("检测到对子，建议加注")
                        return "raise"
                    self.logger.info("没有形成对子，建议跟注")
                    return "call"
            
            elif current_round == GameRound.TURN:
                # 转牌策略
                if len(public_cards) >= 4:
                    # 检查是否形成顺子或同花
                    if self.check_straight(hand_cards + public_cards):
                        self.logger.info("检测到顺子，建议加注")
                        return "raise"
                    if self.check_flush(hand_cards + public_cards):
                        self.logger.info("检测到同花，建议加注")
                        return "raise"
                    self.logger.info("没有形成顺子或同花，建议跟注")
                    return "call"
            
            elif current_round == GameRound.RIVER:
                # 河牌策略
                if len(public_cards) >= 5:
                    # 检查是否形成同花顺
                    if self.check_straight_flush(hand_cards + public_cards):
                        self.logger.info("检测到同花顺，建议加注")
                        return "raise"
                    # 检查是否形成四条
                    if self.check_four_of_a_kind(hand_cards + public_cards):
                        self.logger.info("检测到四条，建议加注")
                        return "raise"
                    self.logger.info("没有形成大牌，建议跟注")
                    return "call"
            
            self.logger.info("无法做出决策，建议跟注")
            return "call"
            
        except Exception as e:
            self.logger.error(f"决策过程出错: {str(e)}")
            return None

    def execute_decision(self, decision: str):
        """执行决策"""
        try:
            self.logger.info(f"开始执行决策: {decision}")
            # 暂时注释掉执行决策的代码，先测试决策逻辑
            """
            if decision == "raise":
                self.logger.info("执行加注操作")
                self.clicker.click_action("加注")
            elif decision == "call":
                self.logger.info("执行跟注操作")
                self.clicker.click_action("跟注")
            elif decision == "fold":
                self.logger.info("执行弃牌操作")
                self.clicker.click_action("弃牌")
            else:
                self.logger.warning(f"未知的决策: {decision}")
            """
            self.logger.info("决策执行完成")
        except Exception as e:
            self.logger.error(f"执行决策失败: {str(e)}")

    def check_straight(self, cards: list) -> bool:
        """检查是否形成顺子"""
        # 实现顺子检查逻辑
        return False

    def check_flush(self, cards: list) -> bool:
        """检查是否形成同花"""
        # 实现同花检查逻辑
        return False

    def check_straight_flush(self, cards: list) -> bool:
        """检查是否形成同花顺"""
        # 实现同花顺检查逻辑
        return False

    def check_four_of_a_kind(self, cards: list) -> bool:
        """检查是否形成四条"""
        # 实现四条检查逻辑
        return False

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
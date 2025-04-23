import logging
from src.core.game_click import GameClicker

class GameExecutor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏执行模块...")
        self.clicker = GameClicker()
        self.logger.info("点击控制模块初始化完成")

    def execute_decision(self, decision: str) -> bool:
        """执行决策"""
        try:
            self.logger.info(f"开始执行决策: {decision}")
            
            if decision == "raise":
                self.logger.info("执行加注操作")
                self.clicker.get_action_position("加注")
            elif decision == "call":
                self.logger.info("执行跟注操作")
                self.clicker.get_action_position("跟注")
            elif decision == "fold":
                self.logger.info("执行弃牌操作")
                self.clicker.get_action_position("弃牌")
            else:
                self.logger.warning(f"未知的决策: {decision}")
                return False
                
            self.logger.info("决策执行完成")
            return True
            
        except Exception as e:
            self.logger.error(f"执行决策失败: {str(e)}")
            return False 
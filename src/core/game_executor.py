import logging
from src.core.game_click import GameClicker
from src.utils.adb import ADBController

class GameExecutor:
    def __init__(self):
        """初始化执行模块"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏执行模块...")
        self.adb = ADBController()
        self.clicker = GameClicker(self.adb.device_id)
        self.logger.info("点击控制模块初始化完成")

    def execute_decision(self, decision: str) -> bool:
        """执行决策"""
        try:
            self.logger.info(f"开始执行决策: {decision}")
            
            # 第一步：将决策映射到具体操作
            action = self._map_decision_to_action(decision)
            if not action:
                return False
                
            # 第二步：获取操作按钮位置
            position = self.clicker.get_action_position(action)
            if not position:
                self.logger.warning(f"未找到{action}按钮位置")
                return False
                
            # 第三步：获取屏幕尺寸
            screen_size = self.adb.get_screen_size()
            if not screen_size:
                self.logger.error("获取屏幕尺寸失败")
                return False
                
            # 第四步：执行点击操作
            return self._execute_click(position, screen_size)
                
        except Exception as e:
            self.logger.error(f"执行决策失败: {str(e)}")
            return False

    def _map_decision_to_action(self, decision: str) -> str:
        """将决策映射到具体操作"""
        action_map = {
            "加注": "加注",
            "跟注": "跟注",
            "弃牌": "弃牌",
            "让牌": "让牌"
        }
        
        if decision in action_map:
            action = action_map[decision]
            self.logger.info(f"执行{action}操作")
            return action
        else:
            self.logger.warning(f"未知的决策: {decision}")
            return ""

    def _execute_click(self, position: tuple, screen_size: tuple) -> bool:
        """执行点击操作"""
        width, height = screen_size
        x = int(width * position[0])
        y = int(height * position[1])
        
        self.logger.info(f"点击坐标: ({x}, {y})")
        return self.adb.execute_click(x, y) 
import logging
from src.core.game_click import GameClicker
from src.utils.adb import ADBController

class GameExecutor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏执行模块...")
        self.adb = ADBController()
        self.clicker = GameClicker(self.adb.device_id)
        self.logger.info("点击控制模块初始化完成")

    def execute_decision(self, decision: str) -> bool:
        """执行决策"""
        try:
            self.logger.info(f"开始执行决策: {decision}")
            
            # 将决策映射到具体操作
            action_map = {
                "raise": "加注",
                "call": "跟注",
                "fold": "弃牌",
                "check": "让牌"
            }
            
            if decision in action_map:
                action = action_map[decision]
                self.logger.info(f"执行{action}操作")
                
                # 获取操作按钮位置
                position = self.clicker.get_action_position(action)
                if position:
                    self.logger.info(f"获取到{action}按钮位置: {position}")
                    
                    # 获取屏幕尺寸
                    screen_size = self.adb.get_screen_size()
                    if screen_size:
                        width, height = screen_size
                        
                        # 计算实际点击坐标
                        x = int(width * position[0])
                        y = int(height * position[1])
                        
                        # 执行点击
                        return self.adb.execute_click(x, y)
                    else:
                        self.logger.error("获取屏幕尺寸失败")
                        return False
                else:
                    self.logger.warning(f"未找到{action}按钮位置")
                    return False
            else:
                self.logger.warning(f"未知的决策: {decision}")
                return False
                
        except Exception as e:
            self.logger.error(f"执行决策失败: {str(e)}")
            return False 
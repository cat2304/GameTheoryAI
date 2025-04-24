import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.adb import ADBController
from typing import Dict, Any, List, Tuple, Optional

class GameExecutor:
    def __init__(self):
        """第一步：初始化执行模块"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏执行模块...")
        self.adb = ADBController()
        self.logger.info("执行模块初始化完成")

    def execute_decision(self, decision: str, actions: List[Dict[str, Any]]) -> bool:
        """执行决策流程"""
        try:
            # 第一步：记录决策信息
            self.logger.info(f"执行决策: {decision}")
            
            # 第二步：查找对应按钮
            for btn in actions:
                if btn["action"] == decision:
                    # 第三步：计算点击坐标
                    box = btn["box"]
                    x = sum(point[0] for point in box) // 4
                    y = sum(point[1] for point in box) // 4
                    
                    # 第四步：执行点击
                    self.logger.info(f"点击按钮: ({x}, {y})")
                    return self.adb.execute_click(x, y)
                    
            # 未找到按钮
            self.logger.warning(f"未找到按钮: {decision}")
            return False
                
        except Exception as e:
            # 错误处理
            self.logger.error(f"执行失败: {str(e)}")
            return False

def main():
    """测试让牌按钮点击"""
    # 第一步：配置日志
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # 第二步：创建执行器
    executor = GameExecutor()
    
    # 第三步：准备测试数据
    test_actions = [
        {
            "action": "底池",
            "box": [[299.0, 1124.0], [348.0, 1124.0], [348.0, 1152.0], [299.0, 1152.0]]
        },
        {
            "action": "底池",
            "box": [[416.0, 1124.0], [467.0, 1124.0], [467.0, 1152.0], [416.0, 1152.0]]
        },
        {
            "action": "底池",
            "box": [[535.0, 1124.0], [583.0, 1124.0], [583.0, 1152.0], [535.0, 1152.0]]
        },
        {
            "action": "弃牌",
            "box": [[298.0, 1241.0], [349.0, 1241.0], [349.0, 1269.0], [298.0, 1269.0]]
        },
        {
            "action": "加注",
            "box": [[414.0, 1241.0], [468.0, 1241.0], [468.0, 1269.0], [414.0, 1269.0]]
        },
        {
            "action": "让牌",
            "box": [[534.0, 1241.0], [584.0, 1241.0], [584.0, 1269.0], [534.0, 1269.0]]
        }
    ]
    
    # 第四步：执行测试
    success = executor.execute_decision("让牌", test_actions)
    print(f"点击结果: {'成功' if success else '失败'}")

if __name__ == "__main__":
    main() 
import logging
import os
from typing import Optional, Tuple

class GameClicker:
    def __init__(self, device_id: str = "127.0.0.1:16384"):
        self.device_id = device_id
        
        # 确保日志目录存在
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志记录器
        self.logger = logging.getLogger("GameClicker")
        self.logger.setLevel(logging.INFO)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        log_file = os.path.join(log_dir, "game_click.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.info("游戏点击控制器初始化完成")

    def get_action_position(self, action: str) -> Optional[Tuple[int, int]]:
        """获取操作按钮的位置"""
        if not action:
            self.logger.info("无需获取操作按钮位置")
            return None
            
        self.logger.info(f"获取操作按钮位置: {action}")
        
        # 操作按钮的位置坐标（相对位置）
        positions = {
            "弃牌": (0.3, 0.7),
            "加注": (0.5, 0.7),
            "让牌": (0.7, 0.7),
            "跟注": (0.5, 0.7)
        }
        
        if action in positions:
            rel_x, rel_y = positions[action]
            return (rel_x, rel_y)
        
        self.logger.error(f"未知的操作按钮: {action}")
        return None

    def get_card_position(self, card: str, is_public: bool = False) -> Optional[Tuple[int, int]]:
        """获取牌的位置"""
        if not card:
            self.logger.info("无需获取牌的位置")
            return None
            
        self.logger.info(f"获取{'公共' if is_public else '手'}牌位置: {card}")
        
        # 牌的位置坐标（相对位置）
        positions = {
            'A': (0.3, 0.4 if is_public else 0.8),
            'K': (0.4, 0.4 if is_public else 0.8),
            'Q': (0.5, 0.4 if is_public else 0.8),
            'J': (0.6, 0.4 if is_public else 0.8),
            '10': (0.7, 0.4 if is_public else 0.8)
        }
        
        if card in positions:
            return positions[card]
        
        self.logger.error(f"未知的牌: {card}")
        return None 
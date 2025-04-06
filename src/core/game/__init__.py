"""
游戏状态模块
===========

定义游戏状态相关的类和函数。
"""

from enum import Enum
from typing import List, Dict, Optional

class GameState:
    """游戏状态类
    
    用于表示和管理游戏的当前状态。
    
    Attributes:
        tiles: 当前手牌列表
        discarded_tiles: 已打出的牌列表
        turn: 当前回合数
        is_active: 游戏是否进行中
    """
    
    def __init__(self):
        """初始化游戏状态"""
        self.tiles = []  # 当前手牌
        self.discarded_tiles = []  # 已打出的牌
        self.turn = 0  # 当前回合数
        self.is_active = False  # 游戏是否进行中
    
    def update_tiles(self, tiles: List[str]) -> None:
        """更新手牌列表
        
        Args:
            tiles: 新的手牌列表
        """
        self.tiles = tiles
    
    def add_discarded_tile(self, tile: str) -> None:
        """添加打出的牌
        
        Args:
            tile: 打出的牌
        """
        self.discarded_tiles.append(tile)
    
    def next_turn(self) -> None:
        """进入下一回合"""
        self.turn += 1
    
    def start_game(self) -> None:
        """开始游戏"""
        self.tiles = []
        self.discarded_tiles = []
        self.turn = 0
        self.is_active = True
    
    def end_game(self) -> None:
        """结束游戏"""
        self.is_active = False
    
    def get_state(self) -> Dict:
        """获取当前状态
        
        Returns:
            Dict: 包含当前状态信息的字典
        """
        return {
            'tiles': self.tiles,
            'discarded_tiles': self.discarded_tiles,
            'turn': self.turn,
            'is_active': self.is_active
        }

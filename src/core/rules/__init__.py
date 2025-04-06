"""
麻将规则引擎模块
=============

定义麻将游戏规则相关的类和函数。
"""

from typing import List, Dict, Optional, Set
from ..game import GameState

class RuleEngine:
    """麻将规则引擎类
    
    实现麻将游戏的规则判断和验证逻辑。
    
    Attributes:
        game_state: 当前游戏状态
        rules: 规则配置字典
    """
    
    def __init__(self, rules: Optional[Dict] = None):
        """初始化规则引擎
        
        Args:
            rules: 规则配置字典，如果为None则使用默认规则
        """
        self.game_state = GameState()
        self.rules = rules or self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """获取默认规则配置
        
        Returns:
            Dict: 默认规则配置
        """
        return {
            'min_tiles': 13,  # 最小手牌数
            'max_tiles': 14,  # 最大手牌数
            'valid_suits': {'万', '筒', '条'},  # 有效花色
            'valid_honors': {'东', '南', '西', '北', '中', '发', '白'},  # 有效字牌
            'max_same_tiles': 4,  # 同一种牌的最大数量
        }
    
    def update_state(self, state: Dict) -> None:
        """更新游戏状态
        
        Args:
            state: 新的游戏状态
        """
        self.game_state.update_tiles(state.get('tiles', []))
        for tile in state.get('discarded_tiles', []):
            self.game_state.add_discarded_tile(tile)
        if state.get('turn') is not None:
            self.game_state.turn = state['turn']
        if state.get('is_active') is not None:
            self.game_state.is_active = state['is_active']
    
    def is_valid_tile(self, tile: str) -> bool:
        """检查牌是否有效
        
        Args:
            tile: 要检查的牌
            
        Returns:
            bool: 牌是否有效
        """
        if not tile:
            return False
            
        # 检查字牌
        if tile in self.rules['valid_honors']:
            return True
            
        # 检查数字牌
        if len(tile) != 2:  # 数字牌应该是"数字+花色"的格式
            return False
            
        number = tile[0]
        suit = tile[1]
        
        # 检查数字是否有效
        if not number.isdigit() or int(number) < 1 or int(number) > 9:
            return False
            
        # 检查花色是否有效
        return suit in self.rules['valid_suits']
    
    def is_valid_hand(self, tiles: List[str]) -> bool:
        """检查手牌是否有效
        
        Args:
            tiles: 手牌列表
            
        Returns:
            bool: 手牌是否有效
        """
        if not tiles:
            return False
            
        # 检查手牌数量
        if len(tiles) < self.rules['min_tiles'] or len(tiles) > self.rules['max_tiles']:
            return False
            
        # 检查每张牌是否有效
        for tile in tiles:
            if not self.is_valid_tile(tile):
                return False
                
        # 检查同一种牌的数量是否超过限制
        tile_counts = {}
        for tile in tiles:
            tile_counts[tile] = tile_counts.get(tile, 0) + 1
            if tile_counts[tile] > self.rules['max_same_tiles']:
                return False
                
        return True
    
    def can_chi(self, tile: str) -> bool:
        """检查是否可以吃牌
        
        Args:
            tile: 要吃的牌
            
        Returns:
            bool: 是否可以吃牌
        """
        if not self.is_valid_tile(tile) or tile in self.rules['valid_honors']:
            return False
            
        number = int(tile[0])
        suit = tile[1]
        
        # 检查是否有相邻的牌
        if number > 1:
            prev_tile = f"{number-1}{suit}"
            if prev_tile in self.game_state.tiles:
                return True
                
        if number < 9:
            next_tile = f"{number+1}{suit}"
            if next_tile in self.game_state.tiles:
                return True
                
        return False
    
    def can_peng(self, tile: str) -> bool:
        """检查是否可以碰牌
        
        Args:
            tile: 要碰的牌
            
        Returns:
            bool: 是否可以碰牌
        """
        if not self.is_valid_tile(tile):
            return False
            
        return self.game_state.tiles.count(tile) >= 2
    
    def can_gang(self, tile: str) -> bool:
        """检查是否可以杠牌
        
        Args:
            tile: 要杠的牌
            
        Returns:
            bool: 是否可以杠牌
        """
        if not self.is_valid_tile(tile):
            return False
            
        return self.game_state.tiles.count(tile) >= 3
    
    def get_valid_actions(self) -> Set[str]:
        """获取当前有效的动作
        
        Returns:
            Set[str]: 有效动作集合
        """
        valid_actions = {'pass'}
        
        # 检查是否可以吃碰杠
        for tile in self.game_state.tiles:
            if self.can_chi(tile):
                valid_actions.add('chi')
            if self.can_peng(tile):
                valid_actions.add('peng')
            if self.can_gang(tile):
                valid_actions.add('gang')
                
        return valid_actions 
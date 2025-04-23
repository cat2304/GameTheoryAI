from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import logging

class GameRound(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class PlayerPosition(Enum):
    SB = "small_blind"
    BB = "big_blind"
    UTG = "under_the_gun"
    MP = "middle_position"
    CO = "cutoff"
    BTN = "button"

@dataclass
class GameState:
    """游戏状态管理类"""
    # 基础信息
    pot_size: int = 0
    current_bet: int = 0
    player_stack: int = 0
    opponent_stack: int = 0
    current_round: GameRound = GameRound.PREFLOP
    player_position: PlayerPosition = PlayerPosition.BTN
    
    # 牌面信息
    hand_cards: List[str] = None
    public_cards: List[str] = None
    last_action: str = ""
    
    # 游戏控制
    is_my_turn: bool = False
    can_check: bool = False
    can_call: bool = False
    can_raise: bool = False
    
    def __post_init__(self):
        if self.hand_cards is None:
            self.hand_cards = []
        if self.public_cards is None:
            self.public_cards = []
        self.logger = logging.getLogger(__name__)
    
    def update_round(self, new_round: GameRound):
        """更新当前轮次"""
        self.logger.info(f"更新轮次: {self.current_round.value} -> {new_round.value}")
        self.current_round = new_round
        self.current_bet = 0
    
    def update_cards(self, hand_cards: List[str], public_cards: List[str]):
        """更新牌面信息"""
        self.logger.info(f"更新手牌: {self.hand_cards} -> {hand_cards}")
        self.logger.info(f"更新公牌: {self.public_cards} -> {public_cards}")
        self.hand_cards = hand_cards
        self.public_cards = public_cards
    
    def update_pot(self, new_pot: int):
        """更新底池大小"""
        self.logger.info(f"更新底池: {self.pot_size} -> {new_pot}")
        self.pot_size = new_pot
    
    def update_stacks(self, player_stack: int, opponent_stack: int):
        """更新玩家筹码"""
        self.logger.info(f"更新玩家筹码: {self.player_stack} -> {player_stack}")
        self.logger.info(f"更新对手筹码: {self.opponent_stack} -> {opponent_stack}")
        self.player_stack = player_stack
        self.opponent_stack = opponent_stack
    
    def update_action(self, action: str):
        """更新最后动作"""
        self.logger.info(f"更新动作: {self.last_action} -> {action}")
        self.last_action = action
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "pot_size": self.pot_size,
            "current_bet": self.current_bet,
            "player_stack": self.player_stack,
            "opponent_stack": self.opponent_stack,
            "current_round": self.current_round.value,
            "player_position": self.player_position.value,
            "hand_cards": self.hand_cards,
            "public_cards": self.public_cards,
            "last_action": self.last_action,
            "is_my_turn": self.is_my_turn,
            "can_check": self.can_check,
            "can_call": self.can_call,
            "can_raise": self.can_raise
        } 
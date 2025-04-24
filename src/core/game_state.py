from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
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
        self.last_hand_cards = []
        self.last_public_cards = []
        self.last_action = None
        self.logger.info("游戏状态初始化完成")
    
    def update_round(self, new_round: GameRound):
        """更新当前轮次"""
        self.logger.info(f"更新轮次: {self.current_round.value} -> {new_round.value}")
        self.current_round = new_round
        self.current_bet = 0
    
    def update_cards(self, result: Dict[str, Any]) -> None:
        """更新游戏状态
        
        Args:
            result: OCR识别结果
        """
        self.last_hand_cards = result.get("handCards", [])
        self.last_public_cards = result.get("publicCards", [])
        
        # 更新游戏状态
        self.update_game_state(
            self.last_hand_cards,
            self.last_public_cards
        )
    
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
    
    def detect_card_changes(self, result: Dict[str, Any]) -> Tuple[bool, bool]:
        """检测牌面变化
        
        Args:
            result: OCR识别结果
            
        Returns:
            Tuple[bool, bool]: (手牌是否变化, 公牌是否变化)
        """
        # 获取当前识别结果中的牌面信息
        current_hand_cards = result.get("handCards", [])
        current_public_cards = result.get("publicCards", [])
        
        # 比较变化
        hand_changed = current_hand_cards != self.last_hand_cards
        public_changed = current_public_cards != self.last_public_cards
        
        if hand_changed or public_changed:
            self.logger.info(f"检测到牌面变化:")
            self.logger.info(f"手牌变化: {self.last_hand_cards} -> {current_hand_cards}")
            self.logger.info(f"公牌变化: {self.last_public_cards} -> {current_public_cards}")
        
        return hand_changed, public_changed

    def update_game_state(self, hand_cards: List[str], public_cards: List[str]) -> None:
        """更新游戏状态
        
        Args:
            hand_cards: 手牌列表
            public_cards: 公牌列表
        """
        self.logger.info(f"更新手牌: {self.hand_cards} -> {hand_cards}")
        self.logger.info(f"更新公牌: {self.public_cards} -> {public_cards}")
        
        # 更新手牌和公牌
        self.hand_cards = hand_cards
        self.public_cards = public_cards
        
        # 根据公牌数量更新当前轮次
        if len(public_cards) == 0:
            self.current_round = GameRound.PREFLOP
        elif len(public_cards) == 3:
            self.current_round = GameRound.FLOP
        elif len(public_cards) == 4:
            self.current_round = GameRound.TURN
        elif len(public_cards) == 5:
            self.current_round = GameRound.RIVER
            
        self.logger.info(f"当前回合: {self.current_round}")
        self.logger.info(f"手牌: {self.hand_cards}")
        self.logger.info(f"公牌: {self.public_cards}")

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
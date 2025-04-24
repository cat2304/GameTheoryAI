import logging
from enum import Enum, auto
from typing import List, Optional
from src.core.game_state import GameState, GameRound


class HandRank(Enum):
    """牌型等级"""
    HIGH_CARD = auto()      # 高牌
    PAIR = auto()           # 对子
    TWO_HIGH_CARDS = auto() # 两张高牌
    STRAIGHT = auto()       # 顺子
    FLUSH = auto()          # 同花
    FOUR_OF_A_KIND = auto() # 四条
    STRAIGHT_FLUSH = auto() # 同花顺


class GameMaker:
    def __init__(self):
        """初始化决策模块"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏决策模块...")

    def make_decision(self, game_state: GameState) -> Optional[str]:
        """根据当前游戏状态做出决策"""
        # 获取当前状态
        current_round = game_state.current_round
        hand = game_state.hand_cards
        public = game_state.public_cards

        self.logger.info(f"当前回合: {current_round}")
        self.logger.info(f"手牌: {hand}")
        self.logger.info(f"公牌: {public}")

        # 评估牌型
        rank = self.evaluate_hand(current_round, hand, public)
        self.logger.info(f"评估牌型: {rank.name}")

        # 根据牌型做出决策
        decision = self._get_decision_by_rank(rank)
        self.logger.info(f"决策: {decision}")
        return decision

    def _get_decision_by_rank(self, rank: HandRank) -> str:
        """根据牌型等级返回决策"""
        # 默认决策：让牌
        decision = "让牌"

        # 有足够强的牌时加注
        if rank in {
            HandRank.STRAIGHT_FLUSH,
            HandRank.FOUR_OF_A_KIND,
            HandRank.FLUSH,
            HandRank.STRAIGHT,
            HandRank.PAIR,
            HandRank.TWO_HIGH_CARDS
        }:
            decision = "加注"

        return decision

    def evaluate_hand(
        self,
        current_round: GameRound,
        hand: List[str],
        public: List[str]
    ) -> HandRank:
        """评估当前牌型"""
        # 翻牌前只看手牌
        if current_round == GameRound.PREFLOP:
            return self._evaluate_preflop(hand)

        # 其余回合组合手牌与公牌
        cards = hand + public
        return self._evaluate_hand(cards)

    def _evaluate_preflop(self, hand: List[str]) -> HandRank:
        """评估翻牌前的牌型"""
        if not hand:  # 如果手牌为空
            return HandRank.HIGH_CARD
            
        if len(hand) >= 2 and hand[0] == hand[1]:
            return HandRank.PAIR
        if len(hand) >= 2 and all(card in {"A", "K", "Q"} for card in hand):
            return HandRank.TWO_HIGH_CARDS
        return HandRank.HIGH_CARD

    def _evaluate_hand(self, cards: List[str]) -> HandRank:
        """评估完整牌型"""
        # 同花顺优先
        if self.check_straight_flush(cards):
            return HandRank.STRAIGHT_FLUSH
        if self.check_four_of_a_kind(cards):
            return HandRank.FOUR_OF_A_KIND
        if self.check_flush(cards):
            return HandRank.FLUSH
        if self.check_straight(cards):
            return HandRank.STRAIGHT
        if self.check_pair(cards):
            return HandRank.PAIR
        return HandRank.HIGH_CARD

    def check_straight(self, cards: List[str]) -> bool:
        """检查是否形成顺子"""
        # TODO: 实现顺子检测逻辑
        return False

    def check_flush(self, cards: List[str]) -> bool:
        """检查是否形成同花"""
        # TODO: 实现同花检测逻辑
        return False

    def check_straight_flush(self, cards: List[str]) -> bool:
        """检查是否形成同花顺"""
        # TODO: 实现同花顺检测逻辑
        return False

    def check_four_of_a_kind(self, cards: List[str]) -> bool:
        """检查是否形成四条"""
        # TODO: 实现四条检测逻辑
        return False

    def check_pair(self, cards: List[str]) -> bool:
        """检查是否存在对子或更高重复"""
        counts = {}
        for card in cards:
            counts[card] = counts.get(card, 0) + 1
            if counts[card] >= 2:
                return True
        return False

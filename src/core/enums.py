from enum import Enum

class RegionType(Enum):
    """区域类型枚举"""
    PUBLIC = 1  # 公牌区域
    HAND = 2    # 手牌区域
    OP = 3      # 操作区域
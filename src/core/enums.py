from enum import Enum

class RegionType(Enum):
    """区域类型枚举"""
    PUBLIC = 1  # 公牌区域
    HAND = 2    # 手牌区域
    OP = 3      # 操作区域

class OCRRegionType(Enum):
    """OCR区域类型枚举"""
    OCR_PUBLIC = 1  # 公牌区域
    OCR_HAND = 2    # 手牌区域
    OCR_OP = 3      # 操作区域 
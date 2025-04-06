"""
游戏核心逻辑模块
=============

本模块包含麻将游戏的核心逻辑和AI组件。

主要组件：
---------
1. 游戏状态管理
    - 游戏状态表示
    - 状态转换
    - 动作验证

2. AI策略
    - 牌面评估
    - 动作决策
    - 概率计算

3. 游戏规则
    - 麻将规则实现
    - 和牌验证
    - 分数计算

主要类：
-------
1. GameState
    - 管理游戏当前状态
    - 跟踪玩家手牌和弃牌
    - 验证游戏动作

2. AIPlayer
    - 实现AI决策
    - 评估可能的动作
    - 计算最优动作

3. RuleEngine
    - 实现麻将规则
    - 验证牌型和组合
    - 计算分数

使用示例：
--------
```python
from core.game import GameState
from core.ai import AIPlayer

# 初始化游戏状态
game_state = GameState()

# 创建AI玩家
ai_player = AIPlayer()

# 获取AI决策
action = ai_player.decide_action(game_state)
```

模块结构：
---------
- game/: 游戏状态和逻辑实现
- ai/: AI策略和决策
- rules/: 游戏规则和验证
- scoring/: 分数计算和牌型评估

详细实现请参考各模块的具体文档。
"""

# 版本信息
__version__ = '1.0.0'

# 模块级导入
from .game import GameState
from .ai import AIPlayer
from .rules import RuleEngine

# 公共接口列表
__all__ = ['GameState', 'AIPlayer', 'RuleEngine'] 
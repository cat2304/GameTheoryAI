"""
核心模块

本模块包含博弈游戏的核心逻辑和AI组件。

包含以下子模块:
- ai: AI决策逻辑
- game: 游戏状态监控
- ocr: OCR识别引擎
- opencv: OpenCV图像处理

主要功能:
- 游戏规则实现
- AI决策算法
- 游戏状态评估
- 图像处理与识别

使用示例:
```python
from src.core.game.game_monitor import GameMonitor
from src.core.ai.decision import AIDecision

# 初始化游戏监控
monitor = GameMonitor()

# 获取游戏状态
game_state = monitor.get_state()

# 使用AI决策
ai = AIDecision()
next_move = ai.decide(game_state)
```

TODO:
- 实现游戏规则
- 扩展AI策略
- 优化图像识别
"""

# 版本信息
__version__ = '0.1.0'

# 模块级导入
from .game.game_monitor import GameMonitor
from .ai import AIPlayer

# 公共接口列表
__all__ = ['GameMonitor', 'AIPlayer'] 
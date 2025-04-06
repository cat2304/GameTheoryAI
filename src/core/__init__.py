"""
核心模块

本模块包含博弈游戏的核心逻辑和AI组件。

子模块
-----
- ai: AI决策逻辑
  - ai_player: AI玩家实现
  - strategy: 策略实现
- game: 游戏状态监控
  - game_monitor: 游戏状态监控
  - state: 游戏状态管理
  - action: 游戏动作管理
- task: 任务管理
  - scheduler: 任务调度
  - executor: 任务执行
- opencv: 图像处理
  - opencv_processor: 图像处理器

主要功能
-------
1. 游戏规则实现
   - 状态评估
   - 动作验证
   - 胜负判定

2. AI决策算法
   - Q-learning
   - 规则树
   - 蒙特卡洛树搜索

3. 游戏状态评估
   - 状态特征提取
   - 状态价值评估
   - 动作价值评估

4. 图像处理与识别
   - 图像预处理
   - 目标检测
   - OCR识别

使用示例
-------
```python
from src.core.game.game_monitor import GameMonitor
from src.core.ai.ai_player import AIPlayer

# 初始化游戏监控
monitor = GameMonitor()

# 初始化AI决策
ai = AIPlayer()

# 获取游戏状态
game_state = monitor.get_state()

# 使用AI决策
next_move = ai.decide(game_state)
```

注意事项
-------
1. 确保游戏监控正确初始化
2. AI决策需要训练数据
3. 任务调度需要合理配置
"""

# 版本信息
__version__ = '0.1.0'

# 模块级导入
from .game.game_monitor import GameMonitor
from .ai.ai_player import AIPlayer

# 公共接口列表
__all__ = [
    'GameMonitor',
    'AIPlayer'
] 
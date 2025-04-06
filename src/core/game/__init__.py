"""
游戏核心模块

提供游戏监控和分析功能。

子模块
-----
- game_monitor: 游戏状态监控
  - 截图捕获
  - 状态识别
  - 状态更新
- state: 游戏状态管理
  - 状态表示
  - 状态转换
  - 状态验证
- action: 游戏动作管理
  - 动作定义
  - 动作执行
  - 动作验证

主要功能
-------
1. 游戏状态监控
   - 实时截图
   - 状态识别
   - 状态更新
   - 异常检测

2. 游戏状态管理
   - 状态表示
   - 状态转换
   - 状态验证
   - 状态持久化

3. 游戏动作管理
   - 动作定义
   - 动作执行
   - 动作验证
   - 动作记录

使用示例
-------
```python
from src.core.game.game_monitor import GameMonitor
from src.core.game.state import GameState
from src.core.game.action import GameAction

# 初始化游戏监控
monitor = GameMonitor()

# 获取游戏状态
game_state = monitor.get_state()

# 创建游戏动作
action = GameAction('play_card', {'card': 'A'})

# 执行游戏动作
result = action.execute(game_state)
```

注意事项
-------
1. 确保游戏窗口可见
2. 监控频率要适中
3. 状态更新要及时
"""

# 版本信息
__version__ = '0.1.0'

# 模块级导入
from .game_monitor import GameMonitor
from .state import GameState
from .action import GameAction

# 公共接口列表
__all__ = [
    'GameMonitor',
    'GameState',
    'GameAction'
]

"""
游戏模块

本模块提供游戏状态监控、状态管理和动作管理功能。

主要功能
-------
1. 游戏状态监控
   - 实时监控游戏状态
   - 状态变化检测
   - 状态数据收集

2. 状态管理
   - 状态数据存储
   - 状态历史记录
   - 状态分析

3. 动作管理
   - 动作执行
   - 动作验证
   - 动作历史

使用示例
-------
```python
from src.core.game.game_monitor import GameMonitor

# 初始化游戏监控
monitor = GameMonitor()

# 获取游戏状态
game_state = monitor.get_state()

# 分析游戏状态
analysis = monitor.analyze_state(game_state)
```

注意事项
-------
1. 确保游戏窗口可见
2. 正确配置游戏区域
3. 定期检查状态更新
"""

# 版本信息
__version__ = '0.1.0'

# 模块级导入
from .game_monitor import GameMonitor

# 公共接口列表
__all__ = ['GameMonitor']

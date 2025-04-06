"""
GameTheoryAI
===========

博弈AI辅助系统，提供游戏状态监控、OCR识别和AI决策功能。

主要功能
-------
- 游戏状态监控
- OCR文字识别
- AI决策支持
- Web界面展示

环境要求
-------
- Python: Conda 3.8
- 依赖包:
  - pytest: 测试框架
  - pyyaml: 配置文件处理
  - opencv: 图像处理
  - pillow: 图像处理
  - numpy: 数值计算
  - pytesseract: OCR文字识别

项目结构
-------
src/
├── core/          # 核心游戏逻辑
│   ├── ai/       # AI决策模块
│   ├── game/     # 游戏监控模块
│   ├── task/     # 任务管理模块
│   └── opencv/   # 图像处理模块
├── utils/         # 工具函数
└── web/          # Web界面

使用示例
-------
    from src.core.game.game_monitor import GameMonitor
    from src.core.ai.decision import AIDecision

    # 初始化游戏监控
    monitor = GameMonitor()

    # 获取游戏状态
    game_state = monitor.get_state()

    # 使用AI决策
    ai = AIDecision()
    next_move = ai.decide(game_state)

环境设置
-------
创建conda环境:
    conda create -n gameai python=3.8

激活环境:
    conda activate gameai

安装依赖:
    conda install pytest pyyaml opencv pillow numpy
    pip install pytesseract

使用说明
-------
1. 确保使用正确的Conda环境:
    conda activate gameai

2. 运行测试:
    python -m pytest tests/

3. 检查环境:
    import sys
    print(sys.version)  # 应该显示Python 3.8.x

注意事项
-------
1. 推荐使用Conda 3.8环境
2. 所有依赖都需要通过conda安装（除了pytesseract）
3. 运行前确保环境激活正确
"""

# 版本信息
__version__ = '1.0.0'

# 模块级导入
from .core.game.game_monitor import GameMonitor
from .core.ai.decision import AIDecision

# 公共接口列表
__all__ = [
    'GameMonitor',
    'AIDecision'
] 
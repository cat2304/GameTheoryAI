#!/usr/bin/env python3
import logging
from src.utils.logger import setup_logging
from src.core.game_controller import GameController

def main():
    # 配置日志
    setup_logging()
    
    # 创建游戏控制器并运行
    controller = GameController()
    controller.run()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import logging
from src.core.game_controller import GameController

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建游戏控制器并运行
    controller = GameController()
    controller.run()

if __name__ == "__main__":
    main()

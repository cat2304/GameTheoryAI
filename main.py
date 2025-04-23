#!/usr/bin/env python3
import os
import logging
from datetime import datetime
from src.vision.screen import ScreenCapture
from src.vision.ocr import recognize_cards
from src.core.game_controller import GameController

def setup_logging():
    """配置日志系统"""
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    root_logger.handlers = []
    
    # 创建文件处理器
    log_file = os.path.join(log_dir, "game.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')  # 使用 'w' 模式覆盖旧日志
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 记录日志系统初始化
    root_logger.info("日志系统初始化完成")
    root_logger.info(f"日志文件路径: {log_file}")


def main():
    # 配置日志
    setup_logging()
    
    # 创建并运行游戏控制器
    controller = GameController()
    controller.run()

if __name__ == "__main__":
    main()

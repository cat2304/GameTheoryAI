"""
日志工具模块
===========

提供项目的日志配置和工具函数。
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config_utils import config

class LogManager:
    """日志管理器类
    
    用于管理项目的日志配置和创建日志记录器。使用单例模式确保全局日志配置一致性。
    
    Attributes:
        _instance: 单例实例
        _loggers: 已创建的日志记录器字典
    """
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._init_logging()
        return cls._instance
    
    def _init_logging(self) -> None:
        """初始化日志系统的基本配置"""
        # 设置日志基本配置
        logging.basicConfig(
            level=getattr(logging, config.get('logging.level', 'INFO')),
            format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    def get_logger(self, name: str, log_file: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
        """获取或创建日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径，默认为None（仅控制台输出）
            level: 日志级别，默认使用配置文件中的级别
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """
        # 如果已存在相同名称的日志记录器，直接返回
        if name in self._loggers:
            return self._loggers[name]
        
        # 创建新的日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(level or getattr(logging, config.get('logging.level', 'INFO')))
        
        # 设置日志格式
        formatter = logging.Formatter(
            config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # 创建文件处理器
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.get('logging.max_bytes', 10*1024*1024),  # 默认10MB
                backupCount=config.get('logging.backup_count', 5),  # 默认保留5个备份
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 缓存并返回日志记录器
        self._loggers[name] = logger
        return logger
    
    def reload_config(self) -> None:
        """重新加载日志配置
        
        重新设置所有日志记录器的级别和格式
        """
        # 更新基本配置
        self._init_logging()
        
        # 更新所有已存在的日志记录器
        new_level = getattr(logging, config.get('logging.level', 'INFO'))
        new_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(new_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        for logger in self._loggers.values():
            logger.setLevel(new_level)
            for handler in logger.handlers:
                handler.setFormatter(formatter)

# 创建全局日志管理器实例
log_manager = LogManager()

def setup_logger(name: str, log_file: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """设置日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，默认为None（仅控制台输出）
        level: 日志级别，默认使用配置文件中的级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return log_manager.get_logger(name, log_file, level)

# 为了兼容性，将 setup_logger 作为 get_logger 导出
get_logger = setup_logger

"""
配置工具模块
===========

提供项目的配置文件管理功能。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

class ConfigManager:
    """配置管理器类
    
    用于加载和管理项目配置。使用单例模式确保全局配置一致性。
    
    Attributes:
        _instance: 单例实例
        _config: 配置数据字典
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """加载配置文件
        
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
        """
        config_path = self._get_config_path()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件格式错误: {e}")
    
    @staticmethod
    def _get_config_path() -> str:
        """获取配置文件路径
        
        Returns:
            str: 配置文件的绝对路径
        """
        return os.path.join(
            Path(__file__).parent.parent.parent,
            'config',
            'app_config.yaml'
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项键名，支持点号分隔的多级键名（如 'adb.path'）
            default: 默认值，当配置项不存在时返回
            
        Returns:
            Any: 配置项的值
        """
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置
        
        Returns:
            Dict[str, Any]: 完整的配置字典
        """
        return self._config.copy()
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()

# 创建全局配置管理器实例
config = ConfigManager()

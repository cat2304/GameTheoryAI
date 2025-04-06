"""
配置管理器测试模块
"""

import os
import sys
import pytest
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_utils import config, ConfigManager

class TestConfigManager:
    """配置管理器测试类"""
    
    def test_singleton(self):
        """测试单例模式"""
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2, "ConfigManager应该是单例"
    
    def test_get_config(self):
        """测试获取配置项"""
        # 测试获取存在的配置项
        adb_path = config.get('adb.path')
        assert adb_path is not None, "应该能获取到ADB路径"
        assert isinstance(adb_path, str), "ADB路径应该是字符串"
        
        # 测试获取嵌套配置项
        screenshot_config = config.get('adb.screenshot')
        assert isinstance(screenshot_config, dict), "截图配置应该是字典"
        assert 'remote_path' in screenshot_config, "截图配置应该包含remote_path"
        
        # 测试获取不存在的配置项（使用默认值）
        default_value = "default"
        non_existent = config.get('non.existent.key', default_value)
        assert non_existent == default_value, "不存在的配置项应该返回默认值"
    
    def test_get_all(self):
        """测试获取所有配置"""
        all_config = config.get_all()
        assert isinstance(all_config, dict), "完整配置应该是字典"
        assert 'adb' in all_config, "配置应该包含ADB部分"
        assert 'logging' in all_config, "配置应该包含日志部分"
    
    def test_config_structure(self):
        """测试配置结构"""
        # 测试ADB配置结构
        adb_config = config.get('adb')
        assert isinstance(adb_config, dict), "ADB配置应该是字典"
        assert 'path' in adb_config, "ADB配置应该包含path"
        assert 'screenshot' in adb_config, "ADB配置应该包含screenshot部分"
        
        # 测试日志配置结构
        logging_config = config.get('logging')
        assert isinstance(logging_config, dict), "日志配置应该是字典"
        assert 'level' in logging_config, "日志配置应该包含level"
        assert 'format' in logging_config, "日志配置应该包含format"
        
    def test_reload(self):
        """测试重新加载配置"""
        original_config = config.get_all()
        config.reload()
        reloaded_config = config.get_all()
        assert original_config == reloaded_config, "重新加载后配置应该保持不变"

if __name__ == "__main__":
    pytest.main([__file__, '-v']) 
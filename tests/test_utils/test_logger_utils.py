"""
日志管理器测试模块
"""

import os
import sys
import logging
import tempfile
import pytest
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.log_utils import log_manager, setup_logger

class TestLogManager:
    """日志管理器测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        # 移除所有处理器
        for logger in log_manager._loggers.values():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        log_manager._loggers.clear()
        
        # 删除临时文件
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        os.rmdir(self.temp_dir)
    
    def test_singleton(self):
        """测试单例模式"""
        manager1 = log_manager
        manager2 = log_manager
        assert manager1 is manager2, "LogManager应该是单例"
    
    def test_logger_creation(self):
        """测试日志记录器创建"""
        logger = setup_logger("test_logger")
        assert logger.name == "test_logger", "日志记录器名称应该正确"
        assert logger.level == logging.INFO, "默认日志级别应该是INFO"
        assert len(logger.handlers) > 0, "日志记录器应该有至少一个处理器"
    
    def test_file_logger(self):
        """测试文件日志记录器"""
        logger = setup_logger("test_file_logger", self.log_file)
        
        # 验证处理器
        handlers = logger.handlers
        assert len(handlers) >= 2, "应该有控制台和文件两个处理器"
        assert any(isinstance(h, logging.StreamHandler) for h in handlers), "应该有控制台处理器"
        assert any(isinstance(h, logging.FileHandler) for h in handlers), "应该有文件处理器"
        
        # 写入日志并验证
        test_message = "测试日志消息"
        logger.info(test_message)
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            assert test_message in log_content, "日志消息应该写入文件"
    
    def test_logger_caching(self):
        """测试日志记录器缓存"""
        logger1 = setup_logger("test_cache")
        logger2 = setup_logger("test_cache")
        assert logger1 is logger2, "相同名称的日志记录器应该复用"
    
    def test_log_levels(self):
        """测试日志级别"""
        logger = setup_logger("test_levels", level=logging.DEBUG)
        assert logger.level == logging.DEBUG, "日志级别应该正确设置"
        
        # 创建临时文件处理器来捕获日志
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            handler = logging.FileHandler(temp_file.name)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 测试不同级别的日志
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            
            # 关闭处理器以确保写入
            handler.close()
            
            # 验证日志内容
            with open(temp_file.name, 'r') as f:
                content = f.read()
                assert "Debug message" in content, "DEBUG消息应该被记录"
                assert "Info message" in content, "INFO消息应该被记录"
                assert "Warning message" in content, "WARNING消息应该被记录"
            
            # 清理临时文件
            os.unlink(temp_file.name)
    
    def test_reload_config(self):
        """测试重新加载配置"""
        logger = setup_logger("test_reload")
        original_level = logger.level
        
        # 重新加载配置
        log_manager.reload_config()
        
        assert logger.level == original_level, "重新加载后日志级别应该保持不变"

if __name__ == "__main__":
    pytest.main([__file__, '-v']) 
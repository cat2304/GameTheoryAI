"""
ADB工具模块

提供Android设备操作相关功能，包括截图等操作。
"""

import os
import subprocess
import time
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# =============================================================================
# 配置管理
# =============================================================================

def load_config() -> Dict[str, Any]:
    """加载配置文件
    
    Returns:
        Dict[str, Any]: 配置信息字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: 配置文件格式错误
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 全局配置
CONFIG = load_config()

# =============================================================================
# 日志管理
# =============================================================================

def setup_logging() -> logging.Logger:
    """配置日志系统
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logging.basicConfig(
        level=getattr(logging, CONFIG['logging']['level']),
        format=CONFIG['logging']['format']
    )
    return logging.getLogger(__name__)

# 全局日志记录器
logger = setup_logging()

# =============================================================================
# ADB工具类
# =============================================================================

class ADBHelper:
    """ADB工具助手类
    
    提供Android设备操作相关功能，包括截图等操作。
    
    Attributes:
        adb_path (str): ADB工具的路径
        logger (logging.Logger): 日志记录器
    """
    
    def __init__(self, adb_path: Optional[str] = None):
        """初始化ADB助手
        
        Args:
            adb_path: ADB工具路径，如果为None则使用配置文件中的路径
            
        Raises:
            subprocess.CalledProcessError: ADB工具验证失败
        """
        self.adb_path = adb_path or CONFIG['adb']['path']
        self.logger = logger
        self._verify_adb()

    def _verify_adb(self) -> None:
        """验证ADB工具是否可用
        
        Raises:
            subprocess.CalledProcessError: ADB工具验证失败
        """
        try:
            subprocess.run([self.adb_path, "version"], check=True, capture_output=True)
            self.logger.info("ADB工具验证成功")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ADB工具验证失败: {e}")
            raise

    def _get_next_number(self, date_dir: Path) -> int:
        """获取下一个文件编号
        
        Args:
            date_dir: 日期目录路径
            
        Returns:
            int: 下一个可用的文件编号
        """
        counter_file = date_dir / CONFIG['adb']['screenshot']['counter_file']
        
        # 如果计数器文件不存在，创建它并返回1
        if not counter_file.exists():
            counter_file.write_text("0")
            return 1
            
        # 读取当前计数并加1
        current = int(counter_file.read_text())
        next_number = current + 1
        counter_file.write_text(str(next_number))
        return next_number

    def take_screenshot(self, local_dir: Optional[str] = None) -> str:
        """执行截图并保存到本地
        
        Args:
            local_dir: 本地保存目录，如果为None则使用配置文件中的路径
            
        Returns:
            str: 保存的截图文件路径
            
        Raises:
            subprocess.CalledProcessError: ADB命令执行失败
            OSError: 文件系统操作失败
            Exception: 其他错误
        """
        try:
            # 准备基础路径
            base_dir = local_dir or CONFIG['adb']['screenshot']['local_dir']
            remote_path = CONFIG['adb']['screenshot']['remote_path']
            
            # 创建日期目录
            date_str = datetime.now().strftime(CONFIG['adb']['screenshot']['date_format'])
            date_dir = Path(base_dir) / date_str
            os.makedirs(date_dir, exist_ok=True)
            
            # 获取下一个文件编号
            number = self._get_next_number(date_dir)
            
            # 生成文件名
            filename = CONFIG['adb']['screenshot']['filename_format'].format(number=number)
            local_path = date_dir / filename
            
            # 执行截图
            self._execute_screenshot(remote_path)
            
            # 拉取文件
            self._pull_screenshot(remote_path, local_path)
            
            self.logger.info(f"截图已保存到: {local_path}")
            return str(local_path)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"执行ADB命令失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"截图过程中发生错误: {e}")
            raise

    def _execute_screenshot(self, remote_path: str) -> None:
        """执行设备截图命令
        
        Args:
            remote_path: 设备端保存路径
            
        Raises:
            subprocess.CalledProcessError: 命令执行失败
        """
        self.logger.info("正在截图...")
        subprocess.run(
            [self.adb_path, "shell", "screencap", "-p", remote_path],
            check=True,
            capture_output=True
        )
        time.sleep(1)  # 等待截图完成

    def _pull_screenshot(self, remote_path: str, local_path: Path) -> None:
        """从设备拉取截图文件
        
        Args:
            remote_path: 设备端文件路径
            local_path: 本地保存路径
            
        Raises:
            subprocess.CalledProcessError: 命令执行失败
        """
        self.logger.info("正在拉取截图...")
        subprocess.run(
            [self.adb_path, "pull", remote_path, str(local_path)],
            check=True,
            capture_output=True
        )
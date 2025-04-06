"""
定时截图工具
===========

提供自动定时截图功能。
"""

import time
from pathlib import Path
from typing import Optional
import threading
import signal
import sys
import subprocess

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.adb_utils import ADBHelper
from src.utils.config_utils import config
from src.utils.log_utils import setup_logger

class AutoScreenshot:
    """自动截图类
    
    提供定时自动截图功能，可以启动、停止截图任务。
    
    Attributes:
        interval (int): 截图间隔时间（秒）
        adb_helper (ADBHelper): ADB工具实例
        is_running (bool): 任务运行状态
        thread (threading.Thread): 截图任务线程
    """
    
    def __init__(self, interval: Optional[int] = None):
        """初始化自动截图工具
        
        Args:
            interval: 截图间隔时间（秒），如果为None则使用配置文件中的值
        """
        self.interval = interval or config.get('adb.screenshot.interval', 5)
        self.adb_helper = ADBHelper()
        self.is_running = False
        self.thread = None
        self.logger = setup_logger(__name__)
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.logger.info("AutoScreenshot 初始化完成")
    
    def _check_device(self) -> bool:
        """检查设备连接状态
        
        Returns:
            bool: 设备是否可用
        """
        try:
            devices = self.adb_helper.list_devices()
            if not devices:
                self.logger.error("未检测到连接的设备，请确保模拟器已启动")
                return False
            self.logger.info(f"检测到设备: {devices}")
            return True
        except Exception as e:
            self.logger.error(f"检查设备状态失败: {str(e)}")
            return False
    
    def _screenshot_task(self):
        """截图任务主循环"""
        self.logger.info(f"开始定时截图任务，间隔时间：{self.interval}秒")
        retry_count = 0
        max_retries = 3
        
        while self.is_running:
            try:
                if not self._check_device():
                    self.logger.warning("等待设备连接...")
                    time.sleep(5)  # 等待5秒后重试
                    continue
                
                self.logger.info("准备截图...")
                screenshot_path = self.adb_helper.take_screenshot()
                self.logger.info(f"截图成功：{screenshot_path}")
                retry_count = 0  # 重置重试计数
            except subprocess.CalledProcessError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    self.logger.error(f"连续截图失败{max_retries}次，停止任务")
                    self.stop()
                    break
                self.logger.warning(f"截图失败，第{retry_count}次重试: {str(e)}")
                time.sleep(5)  # 失败后等待5秒再重试
            except Exception as e:
                self.logger.error(f"截图过程中发生错误: {str(e)}")
                time.sleep(5)  # 发生错误后等待5秒
            
            # 使用更短的睡眠间隔，以便更快响应停止信号
            for _ in range(self.interval):
                if not self.is_running:
                    break
                time.sleep(1)
    
    def start(self) -> bool:
        """启动截图任务
        
        Returns:
            bool: 是否成功启动
        """
        if self.is_running:
            self.logger.warning("截图任务已在运行中")
            return False
        
        try:
            self.logger.info("正在启动截图任务...")
            self.is_running = True
            self.thread = threading.Thread(target=self._screenshot_task)
            self.thread.daemon = True  # 设置为守护线程，这样主程序退出时线程会自动结束
            self.thread.start()
            self.logger.info("截图任务启动成功")
            return True
        except Exception as e:
            self.logger.error(f"启动截图任务失败：{str(e)}")
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """停止截图任务
        
        Returns:
            bool: 是否成功停止
        """
        if not self.is_running:
            self.logger.warning("没有运行中的截图任务")
            return False
        
        try:
            self.logger.info("正在停止截图任务...")
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=5)  # 等待线程结束
            self.logger.info("截图任务已停止")
            return True
        except Exception as e:
            self.logger.error(f"停止截图任务失败：{str(e)}")
            return False
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        self.logger.info(f"收到信号 {signum}，准备停止任务")
        self.stop()

def run_screenshot_task(interval: Optional[int] = None):
    """运行截图任务的便捷函数
    
    Args:
        interval: 截图间隔时间（秒），如果为None则使用配置文件中的值
    """
    logger = setup_logger(__name__)
    logger.info("开始运行截图任务")
    
    screenshot = AutoScreenshot(interval)
    if not screenshot.start():
        logger.error("无法启动截图任务")
        return
    
    try:
        logger.info("截图任务已启动，按 Ctrl+C 停止")
        # 保持主线程运行
        while screenshot.is_running:
            time.sleep(0.1)  # 使用更短的睡眠间隔
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        screenshot.stop()
    except Exception as e:
        logger.error(f"发生错误：{str(e)}")
        screenshot.stop()
    finally:
        logger.info("程序退出")

if __name__ == "__main__":
    run_screenshot_task() 
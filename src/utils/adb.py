import subprocess
import time
import logging
from typing import Optional, Tuple

class ADBController:
    def __init__(self, device_id: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id or self._get_device_id()
        if self.device_id:
            self.logger.info(f"ADB控制器初始化完成，设备ID: {self.device_id}")
        else:
            self.logger.error("ADB控制器初始化失败，未找到设备")

    def _get_device_id(self) -> Optional[str]:
        """获取可用的设备ID"""
        try:
            # 重启 adb 服务器
            subprocess.run(['adb', 'kill-server'], capture_output=True)
            time.sleep(1)
            subprocess.run(['adb', 'start-server'], capture_output=True)
            time.sleep(1)
            
            # 获取设备列表
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            devices = result.stdout.strip().split('\n')[1:]  # 跳过第一行标题
            
            # 查找已连接的设备
            for device in devices:
                if device.strip() and 'device' in device:
                    device_id = device.split('\t')[0]
                    self.logger.info(f"找到设备: {device_id}")
                    return device_id
            
            self.logger.error("未找到已连接的设备")
            return None
            
        except Exception as e:
            self.logger.error(f"获取设备ID失败: {e}")
            return None

    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """获取屏幕尺寸"""
        try:
            result = subprocess.run(
                ['adb', '-s', self.device_id, 'shell', 'wm', 'size'],
                check=True,
                capture_output=True,
                text=True
            )
            size = result.stdout.strip().split(': ')[1]
            width, height = map(int, size.split('x'))
            return width, height
        except Exception as e:
            self.logger.error(f"获取屏幕尺寸失败: {e}")
            return None

    def execute_click(self, x: int, y: int) -> bool:
        """执行点击操作"""
        try:
            command = f'input tap {x} {y}'
            self.logger.info(f"执行点击命令: {command}")
            
            result = subprocess.run(
                ['adb', '-s', self.device_id, 'shell', command],
                check=True,
                capture_output=True,
                text=True
            )
            
            self.logger.info("点击执行成功")
            time.sleep(0.5)  # 点击后等待
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"点击执行失败: {e.stderr if e.stderr else str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"点击执行出错: {str(e)}")
            return False

    def take_screenshot(self, save_path: str) -> bool:
        """获取屏幕截图"""
        try:
            # 使用adb截图
            subprocess.run(
                ['adb', '-s', self.device_id, 'shell', 'screencap', '-p', '/sdcard/screen.png'],
                check=True
            )
            
            # 将截图拉取到本地
            subprocess.run(
                ['adb', '-s', self.device_id, 'pull', '/sdcard/screen.png', save_path],
                check=True
            )
            
            # 删除设备上的截图
            subprocess.run(
                ['adb', '-s', self.device_id, 'shell', 'rm', '/sdcard/screen.png'],
                check=True
            )
            
            return True
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return False 
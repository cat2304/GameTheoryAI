import subprocess
import time
import logging
from typing import Optional, Tuple

class ADBController:
    def __init__(self, device_id: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id or self._get_device_id()
        if self.device_id:
            self.logger.info(f"找到MUMU模拟器ADB设备: {self.device_id}")
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
            
            result = subprocess.run(
                ['adb', '-s', self.device_id, 'shell', command],
                check=True,
                capture_output=True,
                text=True
            )
            
            time.sleep(0.5)  # 点击后等待
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"点击执行失败: {e.stderr if e.stderr else str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"点击执行出错: {str(e)}")
            return False

    def click(self, x: int, y: int) -> Tuple[bool, str]:
        """点击指定坐标
        
        Args:
            x: x坐标
            y: y坐标
            
        Returns:
            Tuple[bool, str]: (是否成功, 成功/错误信息)
        """
        if not self.device_id:
            return False, "未找到设备"
            
        success = self.execute_click(x, y)
        if success:
            return True, "点击成功"
        else:
            return False, "点击失败"

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
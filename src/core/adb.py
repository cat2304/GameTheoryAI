import subprocess
import time
import logging
from typing import Optional, Tuple, List, Dict

class ADBController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device_id = None
        self.logger.info("ADB控制器初始化完成")

    def list_devices(self) -> List[Dict[str, str]]:
        """获取所有已连接的设备列表
        
        Returns:
            List[Dict[str, str]]: 设备列表，每个设备包含 id 和 status
        """
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            devices = result.stdout.strip().split('\n')[1:]  # 跳过第一行标题
            
            device_list = []
            for device in devices:
                if device.strip():
                    device_id, status = device.split('\t')
                    device_list.append({
                        "id": device_id,
                        "status": status
                    })
            
            return device_list
        except Exception as e:
            self.logger.error(f"获取设备列表失败: {e}")
            return []

    def connect_device(self, device_id: str) -> bool:
        """连接指定设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否连接成功
        """
        try:
            if not device_id:
                self.logger.error("未指定设备ID")
                return False
                
            # 检查设备是否已连接
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            devices = result.stdout.strip().split('\n')[1:]
            for device in devices:
                if device.strip() and device_id in device and 'device' in device:
                    self.device_id = device_id
                    self.logger.info(f"设备已连接: {device_id}")
                    return True
            
            # 尝试连接设备
            result = subprocess.run(['adb', 'connect', device_id], capture_output=True, text=True)
            if 'connected' in result.stdout:
                self.device_id = device_id
                self.logger.info(f"设备连接成功: {device_id}")
                return True
            else:
                self.logger.error(f"设备连接失败: {result.stdout}")
                return False
                
        except Exception as e:
            self.logger.error(f"连接设备失败: {e}")
            return False

    def disconnect_device(self) -> bool:
        """断开当前设备连接
        
        Returns:
            bool: 是否断开成功
        """
        try:
            if not self.device_id:
                self.logger.error("当前没有连接的设备")
                return False
                
            # 断开设备连接
            result = subprocess.run(['adb', 'disconnect', self.device_id], capture_output=True, text=True)
            if 'disconnected' in result.stdout:
                self.device_id = None
                self.logger.info("设备断开连接成功")
                return True
            else:
                self.logger.error(f"设备断开连接失败: {result.stdout}")
                return False
                
        except Exception as e:
            self.logger.error(f"断开设备连接失败: {e}")
            return False

    def get_screen_size(self, device_id: str) -> Optional[Tuple[int, int]]:
        """获取屏幕尺寸"""
        try:
            result = subprocess.run(
                ['adb', '-s', device_id, 'shell', 'wm', 'size'],
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

    def click(self, device_id: str, x: int, y: int) -> Tuple[bool, str]:
        """点击指定坐标
        
        Args:
            device_id: 设备ID
            x: x坐标
            y: y坐标
            
        Returns:
            Tuple[bool, str]: (是否成功, 成功/错误信息)
        """
        try:
            command = f'input tap {x} {y}'
            
            result = subprocess.run(
                ['adb', '-s', device_id, 'shell', command],
                check=True,
                capture_output=True,
                text=True
            )
            
            time.sleep(0.5)  # 点击后等待
            return True, "点击成功"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"点击执行失败: {e.stderr if e.stderr else str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"点击执行出错: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def take_screenshot(self, device_id: str, save_path: str) -> bool:
        """获取屏幕截图"""
        try:
            # 使用adb截图
            subprocess.run(
                ['adb', '-s', device_id, 'shell', 'screencap', '-p', '/sdcard/screen.png'],
                check=True
            )
            
            # 将截图拉取到本地
            subprocess.run(
                ['adb', '-s', device_id, 'pull', '/sdcard/screen.png', save_path],
                check=True
            )
            
            # 删除设备上的截图
            subprocess.run(
                ['adb', '-s', device_id, 'shell', 'rm', '/sdcard/screen.png'],
                check=True
            )
            
            return True
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return False 
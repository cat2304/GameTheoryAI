import os
import subprocess
import time
import logging
from typing import Optional, Tuple, List, Dict

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 设置目标分辨率
WIDTH = 900
HEIGHT = 1600
DENSITY = 320  # DPI，默认推荐 320

class ADBController:
    def __init__(self):
        self.device_id = None
        self.width = WIDTH
        self.height = HEIGHT
        self.density = DENSITY
        self.logger = logging.getLogger(__name__)
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

    def connect_device(self, device_id: Optional[str] = None) -> bool:
        """连接设备"""
        try:
            if device_id:
                self.device_id = device_id
                cmd = f"adb connect {device_id}"
            else:
                cmd = "adb devices"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if device_id:
                if "connected" in result.stdout:
                    logger.info(f"成功连接到设备: {device_id}")
                    self._set_resolution()
                    return True
                else:
                    logger.error(f"连接设备失败: {device_id}")
                    return False
            else:
                # 获取第一个可用设备
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    device = lines[1].split('\t')[0]
                    if device != "List of devices attached":
                        self.device_id = device
                        logger.info(f"使用设备: {device}")
                        self._set_resolution()
                        return True
                logger.error("未找到可用设备")
                return False
                
        except Exception as e:
            logger.error(f"连接设备时发生错误: {str(e)}")
            return False

    def _set_resolution(self) -> None:
        """设置设备分辨率"""
        try:
            # 设置分辨率
            cmd = f"adb -s {self.device_id} shell wm size {self.width}x{self.height}"
            subprocess.run(cmd, shell=True, check=True)
            
            # 设置DPI
            cmd = f"adb -s {self.device_id} shell wm density {self.density}"
            subprocess.run(cmd, shell=True, check=True)
            
            logger.info(f"已设置分辨率: {self.width}x{self.height}, DPI: {self.density}")
        except Exception as e:
            logger.error(f"设置分辨率时发生错误: {str(e)}")

    def get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕尺寸"""
        return self.width, self.height

    def take_screenshot(self, save_path: str) -> bool:
        """截图"""
        try:
            if not self.device_id:
                logger.error("未连接设备")
                return False
                
            # 截图
            cmd = f"adb -s {self.device_id} shell screencap -p /sdcard/screenshot.png"
            subprocess.run(cmd, shell=True, check=True)
            
            # 拉取截图
            cmd = f"adb -s {self.device_id} pull /sdcard/screenshot.png {save_path}"
            subprocess.run(cmd, shell=True, check=True)
            
            # 删除设备上的截图
            cmd = f"adb -s {self.device_id} shell rm /sdcard/screenshot.png"
            subprocess.run(cmd, shell=True, check=True)
            
            logger.info(f"截图已保存到: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"截图时发生错误: {str(e)}")
            return False

    def tap(self, x: int, y: int) -> bool:
        """点击屏幕"""
        try:
            if not self.device_id:
                logger.error("未连接设备")
                return False
                
            # 使用列表形式构建命令
            cmd = ['adb', '-s', self.device_id, 'shell', 'input', 'tap', str(x), str(y)]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"点击坐标: ({x}, {y})")
                return True
            else:
                logger.error(f"点击失败: {result.stderr if result.stderr else '未知错误'}")
                return False
            
        except Exception as e:
            logger.error(f"点击屏幕时发生错误: {str(e)}")
            return False

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """滑动屏幕"""
        try:
            if not self.device_id:
                logger.error("未连接设备")
                return False
                
            # 使用列表形式构建命令
            cmd = ['adb', '-s', self.device_id, 'shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(duration)]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"滑动: ({x1}, {y1}) -> ({x2}, {y2}), 持续时间: {duration}ms")
                return True
            else:
                logger.error(f"滑动失败: {result.stderr if result.stderr else '未知错误'}")
                return False
            
        except Exception as e:
            logger.error(f"滑动屏幕时发生错误: {str(e)}")
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

    def click(self, x: int, y: int, device_id: Optional[str] = None) -> Tuple[bool, str]:
        """点击指定坐标
        
        Args:
            x: x坐标
            y: y坐标
            device_id: 设备ID，如果为None则使用当前连接的设备
            
        Returns:
            Tuple[bool, str]: (是否成功, 成功/错误信息)
        """
        try:
            # 使用传入的device_id或当前连接的设备
            target_device = device_id if device_id else self.device_id
            if not target_device:
                return False, "未指定设备ID且当前无连接设备"
                
            # 确保所有参数都是字符串
            x_str = str(x)
            y_str = str(y)
            
            # 使用列表形式构建命令
            cmd = ['adb', '-s', target_device, 'shell', 'input', 'tap', x_str, y_str]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                time.sleep(0.5)  # 点击后等待
                self.logger.info(f"点击坐标: ({x}, {y})")
                return True, "点击成功"
            else:
                error_msg = f"点击执行失败: {result.stderr if result.stderr else '未知错误'}"
                self.logger.error(error_msg)
                return False, error_msg
            
        except Exception as e:
            error_msg = f"点击执行出错: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

if __name__ == "__main__":
    # 测试代码
    adb = ADBController()
    if adb.connect_device():
        # 获取屏幕尺寸
        width, height = adb.get_screen_size()
        print(f"屏幕尺寸: {width}x{height}")
        
        # 截图
        adb.take_screenshot("test_screenshot.png")
        
        # 测试点击
        print("\n测试点击功能:")
        # 点击屏幕中心
        success, message = adb.click(width//2, height//2)
        print(f"点击屏幕中心: {'成功' if success else '失败'} - {message}")
        
        # 点击左上角
        success, message = adb.click(100, 100)
        print(f"点击左上角: {'成功' if success else '失败'} - {message}")
        
        # 点击右下角
        success, message = adb.click(width-100, height-100)
        print(f"点击右下角: {'成功' if success else '失败'} - {message}")
        
        # 滑动屏幕
        adb.swipe(width//2, height*3//4, width//2, height//4) 
import os
import time
import subprocess
import logging
import cv2
import numpy as np
from typing import Tuple, Optional

class ScreenCapture:
    """屏幕捕获类，用于获取Android设备屏幕截图"""
    
    def __init__(self, output_dir: str = "data/screenshots"):
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 设置输出目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取设备ID
        self.device_id = self._get_device_id()
        if not self.device_id:
            raise RuntimeError("未找到可用的设备")
            
        # 等待ADB连接完全建立
        time.sleep(2)
    
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
    
    def take_screenshot(self) -> Tuple[bool, str]:
        """获取屏幕截图
        
        Returns:
            Tuple[bool, str]: (是否成功, 图片路径或错误信息)
        """
        try:
            # 执行截图命令
            result = subprocess.run([
                "adb", "-s", self.device_id, "exec-out", "screencap -p"
            ], capture_output=True)
            
            if result.returncode != 0:
                self.logger.error(f"截图失败: {result.stderr.decode()}")
                return False, result.stderr.decode()
            
            # 转换图像数据
            nparr = np.frombuffer(result.stdout, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                self.logger.error("无法解码图像数据")
                return False, "无法解码图像数据"
            
            # 使用固定的文件名
            filepath = os.path.join(self.output_dir, "latest.png")
            cv2.imwrite(filepath, frame)
            self.logger.info(f"截图已保存: {filepath}")
            return True, filepath
            
        except Exception as e:
            error_msg = f"截图失败: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        # 创建截图实例
        screen_capture = ScreenCapture()
        
        # 等待截图线程启动
        time.sleep(2)
        
        # 获取截图
        success, result = screen_capture.take_screenshot()
        if success:
            print(f"成功获取截图: {result}")
        else:
            print(f"截图失败: {result}")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()

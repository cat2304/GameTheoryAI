import os
import time
import subprocess
import logging
import cv2
import numpy as np
from typing import Tuple, Optional

class ScreenCapture:
    """屏幕捕获类"""
    
    def __init__(self, screenshot_dir: str = "data/screenshots"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = screenshot_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def capture(self, device_id: str) -> Tuple[bool, str]:
        """获取屏幕截图"""
        try:
            # 执行截图命令
            result = subprocess.run([
                "adb", "-s", device_id, "exec-out", "screencap -p"
            ], capture_output=True, check=True)
            
            # 转换图像数据
            nparr = np.frombuffer(result.stdout, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return False, "无法解码图像数据"
            
            # 保存截图
            filepath = os.path.join(self.output_dir, "latest.png")
            cv2.imwrite(filepath, frame)
            self.logger.info(f"截图已保存: {filepath}")
            return True, filepath
            
        except subprocess.CalledProcessError as e:
            error_msg = f"截图失败: {e.stderr.decode()}"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"截图失败: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def capture_region(self, device_id: str, x: int, y: int, width: int, height: int) -> Tuple[bool, str]:
        """获取指定区域截图"""
        try:
            # 获取全屏截图
            success, full_image = self.capture(device_id)
            if not success:
                return False, full_image
            
            # 读取并裁剪图片
            img = cv2.imread(full_image)
            if img is None:
                return False, "无法读取截图"
            
            roi = img[y:y+height, x:x+width]
            region_path = os.path.join(self.output_dir, f"region_{x}_{y}_{width}_{height}.png")
            cv2.imwrite(region_path, roi)
            
            return True, region_path
            
        except Exception as e:
            error_msg = f"区域截图失败: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        screen_capture = ScreenCapture()
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, check=True)
        devices = result.stdout.strip().split('\n')[1:]
        
        if not devices:
            print("未找到已连接的设备")
            return
            
        device_id = devices[0].split('\t')[0]
        success, result = screen_capture.capture(device_id)
        if success:
            print(f"成功获取截图: {result}")
        else:
            print(f"截图失败: {result}")
            
    except subprocess.CalledProcessError as e:
        print(f"ADB命令执行失败: {e.stderr}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()

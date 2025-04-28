import os
import time
import subprocess
import logging
import cv2
import numpy as np
from typing import Tuple, Optional

class ScreenCapture:
    """屏幕捕获类，用于获取Android设备屏幕截图"""
    
    def __init__(self, screenshot_dir: str = "data/screenshots"):
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 设置输出目录
        self.output_dir = screenshot_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def capture(self, device_id: str) -> Tuple[bool, str]:
        """获取屏幕截图
        
        Args:
            device_id: 设备ID
            
        Returns:
            Tuple[bool, str]: (是否成功, 图片路径或错误信息)
        """
        try:
            # 执行截图命令
            result = subprocess.run([
                "adb", "-s", device_id, "exec-out", "screencap -p"
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

    def capture_region(self, device_id: str, x: int, y: int, width: int, height: int) -> Tuple[bool, str]:
        """获取指定区域截图
        
        Args:
            device_id: 设备ID
            x: 起始x坐标
            y: 起始y坐标
            width: 宽度
            height: 高度
            
        Returns:
            Tuple[bool, str]: (是否成功, 图片路径或错误信息)
        """
        try:
            # 先获取全屏截图
            success, full_image = self.capture(device_id)
            if not success:
                return False, full_image
            
            # 读取图片
            img = cv2.imread(full_image)
            if img is None:
                return False, "无法读取截图"
            
            # 裁剪指定区域
            roi = img[y:y+height, x:x+width]
            
            # 保存区域截图
            region_path = os.path.join(self.output_dir, f"region_{x}_{y}_{width}_{height}.png")
            cv2.imwrite(region_path, roi)
            
            return True, region_path
            
        except Exception as e:
            error_msg = f"区域截图失败: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        # 创建截图实例
        screen_capture = ScreenCapture()
        
        # 获取设备列表
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = result.stdout.strip().split('\n')[1:]
        
        if not devices:
            print("未找到已连接的设备")
            return
            
        # 使用第一个设备
        device_id = devices[0].split('\t')[0]
        
        # 获取截图
        success, result = screen_capture.capture(device_id)
        if success:
            print(f"成功获取截图: {result}")
        else:
            print(f"截图失败: {result}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()

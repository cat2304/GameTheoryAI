import os
import time
import subprocess
import logging
import threading
import queue
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

class ScreenCapture:
    """屏幕捕获类，用于获取Android设备屏幕截图"""
    
    def __init__(self, output_dir: str = "data/screenshots"):
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 设置输出目录
        date_folder = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.join(output_dir, date_folder)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化图像队列
        self.frame_queue = queue.Queue(maxsize=10)
        
        # 获取设备ID
        self.device_id = self._get_device_id()
        if not self.device_id:
            raise RuntimeError("未找到可用的设备")
        
        # 启动截图线程
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
    
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
    
    def _capture_frames(self):
        """持续捕获屏幕帧"""
        while True:
            try:
                # 执行截图命令
                result = subprocess.run([
                    "adb", "-s", self.device_id, "exec-out", "screencap -p"
                ], capture_output=True)
                
                if result.returncode != 0:
                    self.logger.error(f"截图失败: {result.stderr.decode()}")
                    time.sleep(2)
                    continue
                
                # 转换图像数据
                nparr = np.frombuffer(result.stdout, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.logger.error("无法解码图像数据")
                    time.sleep(2)
                    continue
                
                # 更新队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
                
                # 控制截图频率
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"捕获帧失败: {e}")
                time.sleep(2)
    
    def take_screenshot(self) -> Tuple[bool, Optional[str]]:
        """获取一张截图"""
        try:
            # 从队列获取最新帧
            frame = self.frame_queue.get(timeout=1.0)
            
            # 生成文件名
            filename = f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            self.logger.info(f"截图已保存: {filepath}")
            return True, filepath
            
        except queue.Empty:
            self.logger.error("无法获取视频帧")
            return False, "无法获取视频帧"
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return False, str(e)

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

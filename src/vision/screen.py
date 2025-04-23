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
from typing import Tuple, Optional, Dict, Any
import json

class ScreenCapture:
    def __init__(self,
                 device_id: str = "emulator-5554",
                 output_dir: str = "data/screenshots"):
        self.device_id = device_id
        
        # 准备输出目录
        date_folder = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.join(output_dir, date_folder)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        
        # 初始化图像队列
        self.frame_queue = queue.Queue(maxsize=10)
        
        # 启动图像捕获线程
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
    
    def _capture_frames(self):
        """捕获视频帧"""
        try:
            while True:
                # 使用 adb screencap 命令捕获屏幕
                result = subprocess.run([
                    "adb", "-s", self.device_id, "exec-out", "screencap -p"
                ], capture_output=True)
                
                if result.returncode != 0:
                    self.logger.error("截图失败: %s", result.stderr.decode())
                    time.sleep(5)  # 失败后等待5秒
                    continue
                
                # 将截图数据转换为图像
                nparr = np.frombuffer(result.stdout, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.logger.error("无法解码图像数据")
                    time.sleep(5)  # 解码失败后等待5秒
                    continue
                
                # 将帧放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # 如果队列满了，移除最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
                
                # 控制帧率，每5秒截图一次
                time.sleep(5)
                
        except Exception as e:
            self.logger.error("捕获视频帧失败", exc_info=e)
    
    def take_screenshot(self) -> Tuple[bool, Optional[str]]:
        """获取当前帧作为截图"""
        try:
            # 从队列获取最新帧
            frame = self.frame_queue.get(timeout=1.0)
            
            # 生成文件名
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"screen_{ts}.png"
            fp = os.path.join(self.output_dir, fn)
            
            # 保存图像
            cv2.imwrite(fp, frame)
            
            self.logger.info(f"截图已保存: {fp}")
            return True, fp
            
        except queue.Empty:
            self.logger.error("无法获取视频帧")
            return False, "无法获取视频帧"
        except Exception as e:
            self.logger.error("截图异常", exc_info=e)
            return False, str(e)
    
    def start_capture_loop(self, interval: int = 1):
        """循环截图主流程"""
        try:
            self.logger.info(f"开始截图循环，间隔 {interval} 秒")
            self.logger.info(f"截图将保存到: {self.output_dir}")
            self.logger.info("按 Ctrl+C 停止")
            
            while True:
                success, result = self.take_screenshot()
                if not success:
                    self.logger.error(f"截图失败: {result}")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n收到停止信号，程序退出")
        except Exception as e:
            self.logger.error(f"发生错误: {str(e)}")
            raise

    def get_frame(self) -> Optional[str]:
        """获取一帧图像"""
        try:
            return self.frame_queue.get()
        except queue.Empty:
            return None

    def is_frame_available(self) -> bool:
        """检查是否有可用的图像帧"""
        return not self.frame_queue.empty()

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # 创建屏幕捕获实例
    screen_capture = ScreenCapture(device_id="emulator-5554")

    success, image_path = screen_capture.take_screenshot()
    if success:
        print(f"成功获取截图: {image_path}")
    else:
        print(f"截图失败: {image_path}")
 
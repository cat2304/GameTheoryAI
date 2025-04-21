import os
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

class MuMuScreenshot:
    def __init__(self, output_dir: str = "/Users/mac/ai/adb"):
        """初始化截图工具
        
        Args:
            output_dir: 截图保存目录
        """
        # 创建以当前日期命名的文件夹
        date_str = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.join(output_dir, date_str)
        self.adb_path = "/Applications/MuMuPlayer.app/Contents/MacOS/MuMuEmulator.app/Contents/MacOS/tools/adb"
        self.screenshot_count = 0  # 从0开始，第一次截图会变成1
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"截图将保存到: {self.output_dir}")
        
    def _run_adb_command(self, command: list) -> tuple:
        """执行 ADB 命令"""
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            return process.returncode, process.stdout, process.stderr
        except Exception as e:
            print(f"执行 ADB 命令出错: {str(e)}")
            return -1, "", str(e)
            
    def take_screenshot(self) -> bool:
        """执行截图操作"""
        try:
            # 生成文件名
            self.screenshot_count += 1
            filename = f"{self.screenshot_count}.png"  # 直接使用数字作为文件名
            remote_path = f"/sdcard/{filename}"
            local_path = os.path.join(self.output_dir, filename)
            
            # 使用最高质量参数截图
            code, _, stderr = self._run_adb_command([
                self.adb_path, "shell", "screencap", "-p", remote_path
            ])
            
            if code != 0:
                print(f"截图失败: {stderr}")
                return False
                
            # 拉取截图到本地
            code, _, stderr = self._run_adb_command([
                self.adb_path, "pull", remote_path, local_path
            ])
            
            if code != 0:
                print(f"拉取截图失败: {stderr}")
                return False
                
            # 删除设备上的临时文件
            self._run_adb_command([
                self.adb_path, "shell", "rm", remote_path
            ])
            
            print(f"已保存第 {self.screenshot_count} 张截图")
            return True
            
        except Exception as e:
            print(f"截图过程出错: {str(e)}")
            return False
            
    def start_screenshot_loop(self, interval: int = 5):
        """开始循环截图
        
        Args:
            interval: 截图间隔（秒）
        """
        print(f"开始循环截图，间隔: {interval}秒")
        
        try:
            while True:
                if self.take_screenshot():
                    time.sleep(interval)
                else:
                    time.sleep(interval)
        except KeyboardInterrupt:
            print("\n停止截图")
            
def main():
    # 创建截图工具实例
    screenshot = MuMuScreenshot()
    
    # 开始循环截图
    screenshot.start_screenshot_loop()
    
if __name__ == "__main__":
    main() 
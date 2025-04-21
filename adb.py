#!/usr/bin/env python3
# scrcpy_screenshot.py

import os
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# ==============================
# Configuration
# ==============================
DEVICE_ID = "127.0.0.1:16384"  # MuMu 设备 TCP/IP 地址
ADB_PATH = "/Applications/MuMuPlayer.app/Contents/MacOS/MuMuEmulator.app/Contents/MacOS/tools/adb"
SCREENSHOT_DIR = "/Users/mac/ai/adb"
SCREENSHOT_INTERVAL = 5       # 秒
SCRCPY_WINDOW_NAME = "scrcpy"  # Scrcpy 窗口名称
SCRCPY_WINDOW_X = 100        # 窗口位置 X
SCRCPY_WINDOW_Y = 100        # 窗口位置 Y
SCRCPY_WINDOW_WIDTH = 800    # 窗口宽度
SCRCPY_WINDOW_HEIGHT = 600   # 窗口高度

# ==============================
# Scrcpy Screenshot Class
# ==============================
class ScrcpyScreenshot:
    def __init__(self,
                 device_id: str = DEVICE_ID,
                 adb_path: str = ADB_PATH,
                 output_dir: str = SCREENSHOT_DIR):
        self.device_id = device_id
        self.adb_path = adb_path

        # 准备输出目录
        date_folder = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.join(output_dir, date_folder)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 日志配置
        self.logger = logging.getLogger("ScrcpyScreenshot")
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(self.output_dir, "scrcpy_screenshot.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        self.scrcpy_proc = None
        self.logger.info(f"输出目录: {self.output_dir}")

        # 1) 确保 ADB TCP/IP 连接
        self._ensure_adb_tcp()
        # 2) 启动 scrcpy 窗口
        self._start_scrcpy()

    def _ensure_adb_tcp(self):
        """将设备切到 TCP/IP 模式并连接"""
        try:
            self.logger.debug("启用 ADB TCP/IP 模式 (5555)...")
            subprocess.run([self.adb_path, "-s", self.device_id, "tcpip", "5555"],
                           check=True, stdout=subprocess.DEVNULL)
            self.logger.debug(f"连接到设备 {self.device_id}...")
            subprocess.run([self.adb_path, "connect", self.device_id],
                           check=True, stdout=subprocess.DEVNULL)
            self.logger.info(f"ADB 已连接到 {self.device_id}")
        except subprocess.CalledProcessError as e:
            self.logger.error("ADB TCP/IP 连接失败", exc_info=e)
            raise RuntimeError("ADB TCP/IP 连接失败")

    def _start_scrcpy(self):
        """启动 scrcpy 窗口"""
        try:
            args = [
                "scrcpy",
                "-s", self.device_id,
                "--window-title", SCRCPY_WINDOW_NAME,
                "--window-x", str(SCRCPY_WINDOW_X),
                "--window-y", str(SCRCPY_WINDOW_Y),
                "--window-width", str(SCRCPY_WINDOW_WIDTH),
                "--window-height", str(SCRCPY_WINDOW_HEIGHT),
                "--no-control",
                "--no-audio",
                "--stay-awake",
                "--render-driver", "metal"
            ]
            self.logger.debug(f"启动 scrcpy: {' '.join(args)}")
            self.scrcpy_proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)  # 等待窗口显示
            self.logger.info("scrcpy 窗口已启动")
        except Exception as e:
            self.logger.error("启动 scrcpy 失败", exc_info=e)
            raise RuntimeError("scrcpy 启动失败")

    def _stop_scrcpy(self):
        """停止 scrcpy 进程"""
        if self.scrcpy_proc and self.scrcpy_proc.poll() is None:
            self.logger.debug("终止 scrcpy 进程…")
            self.scrcpy_proc.terminate()
            try:
                self.scrcpy_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.logger.warning("scrcpy 未及时退出，强制杀死")
                self.scrcpy_proc.kill()
        self.scrcpy_proc = None

    def take_screenshot(self) -> Tuple[bool, Optional[str]]:
        """使用 ADB 命令截取屏幕"""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"screen_{ts}.png"
            fp = os.path.join(self.output_dir, fn)
            
            # 使用 ADB 截图
            subprocess.run([
                self.adb_path,
                "-s", self.device_id,
                "exec-out",
                "screencap",
                "-p"
            ], check=True, stdout=open(fp, "wb"))
            
            self.logger.info(f"截图已保存: {fp}")
            return True, fp

        except Exception as e:
            self.logger.error("截图异常", exc_info=e)
            return False, str(e)

    def start_loop(self, interval: int = SCREENSHOT_INTERVAL):
        """循环截图主流程"""
        self.logger.info(f"开始每 {interval}s 截图，按 Ctrl+C 停止")
        try:
            while True:
                self.take_screenshot()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("用户中断循环")
        finally:
            self._stop_scrcpy()
            self.logger.info("清理完成，退出")

# ==============================
# Main Entry
# ==============================
def main():
    logging.basicConfig(level=logging.INFO)
    
    # 检查依赖
    if subprocess.run(["which", "scrcpy"], capture_output=True).returncode != 0:
        print("Error: 未检测到 scrcpy，请先安装")
        return

    screener = ScrcpyScreenshot()
    screener.start_loop()

if __name__ == "__main__":
    main()

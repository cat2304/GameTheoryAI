import os
import time
import subprocess
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 截图保存路径
SCREENSHOT_BASE_DIR = "/Users/mac/ai/adb"

def get_screenshot_dir():
    """获取当前日期的截图目录"""
    today = datetime.now().strftime("%Y%m%d")
    screenshot_dir = os.path.join(SCREENSHOT_BASE_DIR, today)
    os.makedirs(screenshot_dir, exist_ok=True)
    return screenshot_dir

def get_timestamp_filename():
    """获取时间戳文件名"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17] + ".png"

def take_screenshot():
    """获取屏幕截图"""
    try:
        # 获取设备列表
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = result.stdout.strip().split('\n')[1:]
        
        if not devices:
            logger.error("未找到已连接的设备")
            return None
            
        # 使用第一个设备
        device_id = devices[0].split('\t')[0]
        
        # 执行截图命令
        result = subprocess.run([
            "adb", "-s", device_id, "exec-out", "screencap -p"
        ], capture_output=True)
        
        if result.returncode != 0:
            logger.error(f"截图失败: {result.stderr.decode()}")
            return None
        
        # 获取保存路径
        screenshot_dir = get_screenshot_dir()
        filename = get_timestamp_filename()
        screenshot_path = os.path.join(screenshot_dir, filename)
        
        # 保存截图
        with open(screenshot_path, 'wb') as f:
            f.write(result.stdout)
        
        logger.info(f"截图已保存: {screenshot_path}")
        return screenshot_path
        
    except Exception as e:
        logger.error(f"截图失败: {str(e)}")
        return None

def main():
    """主函数"""
    logger.info("截图程序已启动，输入1进行截图，输入q退出程序")
    
    while True:
        user_input = input("请输入命令 (1: 截图, q: 退出): ").strip().lower()
        
        if user_input == 'q':
            logger.info("程序退出")
            break
        elif user_input == '1':
            logger.info("开始截图...")
            screenshot_path = take_screenshot()
            if screenshot_path:
                logger.info(f"截图完成: {screenshot_path}")
            else:
                logger.error("截图失败")
        else:
            logger.warning("无效的命令，请输入1进行截图或q退出程序")

if __name__ == "__main__":
    main()

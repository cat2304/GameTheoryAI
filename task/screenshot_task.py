import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import time
import yaml
import logging
from fetch.adb import ADBHelper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_screenshot_task():
    """运行定时截图任务"""
    try:
        # 加载配置
        config = load_config()
        interval = config['adb']['screenshot']['interval']
        
        # 创建ADB助手实例
        adb = ADBHelper()
        
        logger.info(f"开始定时截图任务，间隔时间：{interval}秒")
        
        while True:
            try:
                # 执行截图
                screenshot_path = adb.take_screenshot()
                logger.info(f"截图成功：{screenshot_path}")
            except Exception as e:
                logger.error(f"截图失败：{str(e)}")
            
            # 等待指定时间
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("任务已停止")
    except Exception as e:
        logger.error(f"任务异常：{str(e)}")

if __name__ == "__main__":
    run_screenshot_task() 
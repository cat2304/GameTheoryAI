import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fetch.adb import ADBHelper

def test_screenshot():
    try:
        # 创建ADB助手实例
        adb = ADBHelper()
        
        # 执行截图
        screenshot_path = adb.take_screenshot()
        print(f"截图成功！保存在: {screenshot_path}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_screenshot() 
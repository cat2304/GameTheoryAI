import os
import sys
import pytest
from pathlib import Path
import subprocess
from unittest.mock import patch, MagicMock

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.adb_utils import ADBHelper

def is_device_connected():
    """检查是否有设备连接"""
    try:
        result = subprocess.run(
            ['/Applications/NemuPlayer.app/Contents/MacOS/adb', 'devices'],
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip().split('\n')) > 1
    except Exception:
        return False

class TestADBHelper:
    """ADBHelper类的测试用例"""
    
    @pytest.fixture
    def adb_helper(self):
        """创建ADBHelper实例的fixture"""
        return ADBHelper()
    
    @pytest.mark.skipif(not is_device_connected(), reason="没有设备连接")
    def test_screenshot_creation(self, adb_helper):
        """测试截图创建功能"""
        try:
            # 执行截图
            screenshot_path = adb_helper.take_screenshot()
            
            # 验证截图文件是否创建成功
            assert os.path.exists(screenshot_path), f"截图文件未创建: {screenshot_path}"
            assert os.path.getsize(screenshot_path) > 0, "截图文件是空的"
            
            print(f"截图成功！保存在: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            pytest.fail(f"截图测试失败: {str(e)}")
    
    @pytest.mark.skipif(not is_device_connected(), reason="没有设备连接")
    def test_screenshot_path_format(self, adb_helper):
        """测试截图路径格式是否正确"""
        screenshot_path = self.test_screenshot_creation(adb_helper)
        
        # 验证文件名格式
        assert screenshot_path.endswith('.png'), "截图文件扩展名不是.png"
        assert 'screenshot_' in os.path.basename(screenshot_path), "文件名不包含'screenshot_'前缀"
        
        # 验证文件路径
        assert os.path.isabs(screenshot_path), "不是绝对路径"
        assert os.path.dirname(screenshot_path).endswith('screenshots'), "截图不在screenshots目录中"
    
    def test_adb_verification(self, adb_helper):
        """测试ADB工具验证功能"""
        # 验证ADB工具路径是否正确设置
        assert adb_helper.adb_path is not None, "ADB路径未设置"
        assert isinstance(adb_helper.adb_path, str), "ADB路径类型错误"
    
    @patch('subprocess.run')
    def test_adb_command_error_handling(self, mock_run, adb_helper):
        """测试ADB命令错误处理"""
        # 模拟命令执行失败
        mock_run.side_effect = subprocess.CalledProcessError(1, 'adb devices')
        
        # 验证是否正确处理异常
        with pytest.raises(subprocess.CalledProcessError):
            adb_helper._verify_adb()

if __name__ == "__main__":
    pytest.main([__file__, '-v']) 
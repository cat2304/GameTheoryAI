"""
GameTheoryAI 工具模块
===================

本模块提供项目所需的各种工具函数和类。

主要组件：
---------
1. 配置管理
    - 配置文件加载和访问
    - 配置项的获取和验证
    - 配置重载功能

2. 日志管理
    - 日志记录器配置
    - 文件和控制台输出
    - 日志级别控制

3. ADB工具
    - 设备连接和验证
    - 截图功能
    - 文件传输

4. 自动截图
    - 定时自动截图
    - 可配置间隔时间
    - 错误处理和日志记录

使用示例：
---------
1. 配置管理：
    ```python
    from utils.config_utils import config
    
    adb_path = config.get('adb.path')
    log_level = config.get('logging.level', 'INFO')
    ```

2. 日志记录：
    ```python
    from utils.log_utils import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("操作成功")
    ```

3. ADB操作：
    ```python
    from utils.adb_utils import ADBHelper
    
    adb = ADBHelper()
    screenshot_path = adb.take_screenshot()
    ```

4. 自动截图：
    ```python
    from utils.auto_screenshot import AutoScreenshot
    
    screenshot = AutoScreenshot(interval=5)
    screenshot.start()
    ```

目录结构：
---------
- config_utils.py: 配置管理工具
- log_utils.py: 日志工具
- adb_utils.py: ADB操作工具
- auto_screenshot.py: 自动截图工具

详细文档请参考各模块的具体文件。
"""

# 版本信息
__version__ = '1.0.0'

# 模块级导入
from .config_utils import config
from .log_utils import setup_logger
from .adb_utils import ADBHelper
from .config_utils import ConfigManager
from .log_utils import LogManager
from .auto_screenshot import AutoScreenshot, run_screenshot_task

# 公共接口列表
__all__ = [
    'config',
    'setup_logger',
    'ADBHelper',
    'ConfigManager',
    'LogManager',
    'AutoScreenshot',
    'run_screenshot_task'
]

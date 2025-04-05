"""
ADB工具包 (Android Debug Bridge Tools)
====================================

这个包提供了一系列用于Android设备自动化操作的工具。

主要功能:
--------
1. 设备截图获取
   - 支持自定义保存路径
   - 自动生成时间戳文件名
   - 错误重试机制

2. ADB工具管理
   - 自动验证ADB可用性
   - 支持自定义ADB路径
   - 异常处理和日志记录

配置说明:
--------
配置文件位置: config/config.yaml
主要配置项:
- adb.path: ADB工具路径
- adb.screenshot.remote_path: 设备端截图路径
- adb.screenshot.local_dir: 本地保存目录
- adb.screenshot.filename_format: 文件名格式
- logging: 日志配置

使用示例:
--------
>>> from fetch.adb import ADBHelper
>>> adb = ADBHelper()
>>> screenshot_path = adb.take_screenshot()
>>> print(f"截图保存在: {screenshot_path}")

依赖要求:
--------
- Python >= 3.6
- pyyaml
- adb工具

作者: GameTheoryAI Team
版本: 1.0.0
许可: MIT
"""

__version__ = "1.0.0"
__author__ = "GameTheoryAI Team"
__license__ = "MIT"

from .adb import ADBHelper

__all__ = ['ADBHelper'] 
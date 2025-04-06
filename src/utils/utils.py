"""
工具模块 - 系统工具和辅助功能

提供系统工具、ADB工具和配置管理功能。
"""

import os, cv2, yaml, numpy as np, logging, subprocess, time, threading, signal, sys, json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging.handlers
try:
    import pngquant  # 尝试导入pngquant库
    PNGQUANT_AVAILABLE = True
except ImportError:
    PNGQUANT_AVAILABLE = False

# 导入OCR相关功能
from .ocr import GameOCR, handle_ocr_test, find_latest_screenshot

# 定义日志级别常量
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40

# 默认日志级别
DEFAULT_LOG_LEVELS = {
    "default": "INFO",
    "adb": "INFO",
    "screenshot": "INFO", 
    "ocr": "DEBUG"
}

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置加载错误: {str(e)}")
            # 返回默认配置
            return {
                "app": {"name": "GameTheoryAI"},
                "adb": {
                    "path": "/Applications/MuMuPlayer.app/Contents/MacOS/MuMuEmulator.app/Contents/MacOS/tools/adb",
                    "screenshot": {
                        "local_temp_dir": "/tmp/screenshots",
                        "remote_path": "/sdcard/screenshot.png",
                        "date_format": '%Y%m%d',
                        "interval": 1,
                        "clear_target_dir": False
                    }
                },
                "environment": {
                    "temp_dir": "/tmp",
                    "log_dir": "logs"
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            }
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """使用点分隔的路径获取配置值
        
        Args:
            key_path: 点分隔的键路径，例如 "adb.screenshot.interval"
            default: 如果键不存在，返回的默认值
            
        Returns:
            找到的配置值或默认值
        """
        keys = key_path.split('.')
        value = self.config
        
        # 尝试按照路径访问嵌套的字典
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """使用点分隔的路径设置配置值
        
        Args:
            key_path: 点分隔的键路径，例如 "adb.screenshot.interval"
            value: 要设置的值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 遍历路径
        for i, k in enumerate(keys[:-1]):
            # 如果键不存在，创建一个新的字典
            if k not in config:
                config[k] = {}
            # 如果值不是字典，无法继续嵌套，返回
            elif not isinstance(config[k], dict):
                return
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def save(self) -> bool:
        """保存配置到文件
        
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"配置保存错误: {str(e)}")
            return False

class LogManager:
    """日志管理器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置根日志记录器
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)  # 根记录器设置最低级别
        
        # 控制台处理器
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s', 
                           datefmt='%Y-%m-%d %H:%M:%S')
        )
        self.console_handler.setLevel(logging.INFO)
        
        if not self.root_logger.handlers:
            self.root_logger.addHandler(self.console_handler)
        
        # 记录器缓存
        self.loggers = {}
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取预配置的日志记录器"""
        if name in self.loggers:
            return self.loggers[name]
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        
        # 确定日志级别
        level_name = DEFAULT_LOG_LEVELS.get(name.split('.')[0], DEFAULT_LOG_LEVELS['default'])
        level = getattr(logging, level_name)
        logger.setLevel(level)
        
        # 添加文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, f"{name.replace('.', '_')}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
                           datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(file_handler)
        
        # 缓存并返回日志记录器
        self.loggers[name] = logger
        return logger

class ADBHelper:
    """ADB工具辅助类"""
    
    def __init__(self, adb_path):
        self.logger = get_logger("adb.helper")  # 使用模块化日志名称
        self.adb_path = adb_path
        self._screenshot_counter = 0
        self._device_connected = False
        self._init_device_session()
    
    def _init_device_session(self):
        """初始化设备会话"""
        try:
            self.logger.info(f"开始初始化ADB会话，ADB路径: {self.adb_path}")
            devices = self._get_devices()
            if devices:
                # 使用第一个设备
                self._device_connected = True
                self.logger.info(f"已连接设备: {devices[0]}")
            else:
                self._device_connected = False
                self.logger.warning("没有检测到已连接的设备，请连接设备后重试")
                print("设备连接检测: 没有发现可用设备，请检查USB连接或打开USB调试")
        except Exception as e:
            self._device_connected = False
            self.logger.error(f"初始化ADB会话失败: {str(e)}")
    
    def _get_devices(self) -> List[str]:
        """获取连接的设备列表"""
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5  # 增加超时处理
            )
            
            if result.returncode != 0:
                self.logger.error(f"获取设备列表失败: {result.stderr}")
                return []
            
            # 解析设备列表，跳过第一行（标题行）
            devices = []
            for line in result.stdout.splitlines()[1:]:
                if line.strip() and not line.strip().startswith('*') and '\t' in line:
                    device_id = line.split('\t')[0].strip()
                    if device_id and "offline" not in line:
                        devices.append(device_id)
            
            return devices
            
        except subprocess.TimeoutExpired:
            self.logger.error("获取设备列表超时")
            return []
        except Exception as e:
            self.logger.error(f"获取设备列表异常: {str(e)}")
            return []
    
    def check_device_connection(self) -> bool:
        """检查设备连接状态"""
        try:
            devices = self._get_devices()
            self._device_connected = len(devices) > 0
            return self._device_connected
        except Exception as e:
            self.logger.error(f"检查设备连接失败: {str(e)}")
            self._device_connected = False
            return False
    
    def take_screenshot(self, local_dir: Optional[str] = None) -> Optional[str]:
        """获取屏幕截图
        
        Args:
            local_dir: 本地保存目录，None表示使用默认路径
            
        Returns:
            str: 截图保存路径，失败时返回None
        """
        if not self.check_device_connection():
            self.logger.warning("无法截图: 设备未连接")
            return None
            
        try:
            # 使用adb命令执行截图
            remote_path = "/sdcard/screenshot.png"
            
            # 执行截图命令
            result = subprocess.run(
                [self.adb_path, "shell", "screencap", "-p", remote_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.error(f"截图命令执行失败: {result.stderr.decode('utf-8')}")
                return None
            
            # 生成本地保存路径
            if local_dir is None:
                local_dir = "screenshots"
            
            # 创建日期子目录
            date_str = datetime.now().strftime('%Y%m%d')
            target_dir = os.path.join(local_dir, date_str)
            os.makedirs(target_dir, exist_ok=True)
            
            # 增加计数器，确保文件名唯一
            self._screenshot_counter += 1
            local_path = os.path.join(target_dir, f"{self._screenshot_counter}.png")
            
            # 拉取截图文件到本地
            pull_result = subprocess.run(
                [self.adb_path, "pull", remote_path, local_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            if pull_result.returncode != 0:
                self.logger.error(f"拉取截图失败: {pull_result.stderr.decode('utf-8')}")
                return None
            
            self.logger.info(f"截图已保存至: {local_path}")
            return local_path
            
        except subprocess.TimeoutExpired:
            self.logger.error("截图操作超时")
            return None
        except Exception as e:
            self.logger.error(f"截图异常: {str(e)}")
            return None
    
    def execute_command(self, command: List[str]) -> Tuple[int, str, str]:
        """执行ADB命令
        
        Args:
            command: ADB命令参数列表，不包含adb路径
            
        Returns:
            Tuple[int, str, str]: 返回码、标准输出、标准错误
        """
        try:
            cmd = [self.adb_path] + command
            self.logger.debug(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            return (
                result.returncode,
                result.stdout.decode('utf-8'),
                result.stderr.decode('utf-8')
            )
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"命令执行超时: {' '.join([self.adb_path] + command)}")
            return -1, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"命令执行异常: {str(e)}")
            return -1, "", str(e)

class ScreenshotManager:
    """截图管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.logger = get_logger("screenshot.manager")  # 使用模块化日志名称
        self.config_manager = config_manager
        self.adb_helper = ADBHelper(config_manager.get_config()["adb"]["path"])
        self._screenshot_tasks = {}
        self._counter_lock = threading.Lock()
        self._last_screenshot_time = 0  # 记录上次截图时间，控制速率
        self._min_interval = 0.1  # 最小截图间隔（秒），防止过度截图
    
    def take_screenshot(self, save_dir: Optional[str] = None) -> Optional[str]:
        """获取单张截图"""
        self.logger.debug("尝试获取截图...")
        try:
            # 控制截图速率，避免过度截图
            current_time = time.time()
            with self._counter_lock:
                if current_time - self._last_screenshot_time < self._min_interval:
                    # 如果间隔太短，适当延迟
                    time.sleep(self._min_interval - (current_time - self._last_screenshot_time))
                self._last_screenshot_time = time.time()
            
            # 检查设备连接状态
            if not self.adb_helper._device_connected:
                self.logger.error("设备未连接，请连接设备后重试")
                print("设备未连接，请连接设备后重试")
                return None
            
            # 执行截图
            path = self.adb_helper.take_screenshot(local_dir=save_dir)
            if path:
                self.logger.info(f"截图已保存: {path}")
            else:
                self.logger.warning("截图失败，请检查设备连接")
                print("截图失败，请检查设备连接")
            return path
        except Exception as e:
            self.logger.error("截图操作异常", exc_info=True)
            return None
    
    def start_screenshot_task(self, interval: int = 5, max_count: Optional[int] = None, 
                             save_dir: Optional[str] = None) -> str:
        """启动定时截图任务"""
        # 检查设备连接状态
        if not self.adb_helper._device_connected:
            self.logger.error("设备未连接，无法启动截图任务")
            print("设备未连接，无法启动截图任务")
            return "ERROR_NO_DEVICE"
        
        task_id = str(uuid.uuid4())
        stop_event = threading.Event()
        
        # 确定保存目录
        temp_dir = self.config_manager.get('environment.temp_dir', '/Users/mac/ai/temp')
        target_dir = save_dir or os.path.join(temp_dir, 'screenshots')
        
        # 检查是否需要清空目标目录
        clear_target_dir = self.config_manager.get('adb.screenshot.clear_target_dir', False)
        self.logger.info(f"清空目标目录配置: {clear_target_dir}")
        
        if clear_target_dir:
            self.logger.info(f"准备清空目标目录: {target_dir}")
            try:
                # 获取日期目录路径
                date_str = datetime.now().strftime(self.config_manager.get('adb.screenshot.date_format', '%Y%m%d'))
                date_dir = os.path.join(target_dir, date_str)
                
                # 创建日期目录（如果不存在）
                os.makedirs(date_dir, exist_ok=True)
                
                # 清空日期目录中的文件
                file_count = 0
                for file in os.listdir(date_dir):
                    file_path = os.path.join(date_dir, file)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            file_count += 1
                        except Exception as e:
                            self.logger.warning(f"删除文件失败: {file_path}, 错误: {e}")
                
                self.logger.info(f"已清空日期目录: {date_dir}, 共删除{file_count}个文件")
                
                # 重置ADB助手的截图计数器
                self.adb_helper._screenshot_counter = 0
                
            except Exception as e:
                self.logger.error(f"清空目标目录时出错: {str(e)}")
        else:
            self.logger.info("未启用目标目录清空功能")
        
        def screenshot_task():
            count = 0
            self.logger.info(f"截图任务[{task_id}]已启动, 间隔: {interval}秒")
            
            try:
                last_time = 0
                while not stop_event.is_set():
                    # 检查是否达到最大次数
                    if max_count and count >= max_count:
                        self.logger.info(f"截图任务[{task_id}]已完成, 共截图{count}张")
                        break
                    
                    # 检查设备连接状态
                    if not self.adb_helper._device_connected:
                        self.logger.warning("设备未连接，等待设备连接...")
                        print("设备未连接，等待设备连接...")
                        # 等待一段时间后重试
                        if stop_event.wait(timeout=5):
                            break
                        continue
                    
                    current_time = time.sleep()
                    # 动态调整等待时间，确保截图间隔准确
                    if current_time - last_time < interval:
                        # 使用短间隔等待以提高响应性
                        wait_time = min(0.5, interval - (current_time - last_time))
                        if stop_event.wait(timeout=wait_time):
                            break
                        continue
                    
                    # 获取截图
                    screenshot_path = self.take_screenshot(save_dir=save_dir)
                    if screenshot_path:
                        count += 1
                        last_time = time.time()
                    else:
                        # 截图失败时等待较长时间
                        self.logger.warning("截图失败，等待重试...")
                        if stop_event.wait(timeout=5):
                            break
                        
            except Exception as e:
                self.logger.error(f"截图任务[{task_id}]执行失败: {str(e)}")
            finally:
                # 任务结束时从字典中移除
                with self._counter_lock:
                    if task_id in self._screenshot_tasks:
                        del self._screenshot_tasks[task_id]
                self.logger.info(f"截图任务[{task_id}]已停止")
        
        # 创建并启动线程
        thread = threading.Thread(target=screenshot_task, daemon=True)
        with self._counter_lock:
            self._screenshot_tasks[task_id] = {
                "thread": thread,
                "stop_event": stop_event,
                "interval": interval,
                "max_count": max_count,
                "start_time": datetime.now(),
                "save_dir": save_dir
            }
        thread.start()
        
        return task_id
    
    def stop_screenshot_task(self, task_id: str) -> bool:
        """停止截图任务"""
        with self._counter_lock:
            if task_id not in self._screenshot_tasks:
                self.logger.warning(f"截图任务[{task_id}]不存在")
                return False
                
            task_info = self._screenshot_tasks[task_id]
            task_info["stop_event"].set()
            
            # 等待线程结束
            if task_info["thread"].is_alive():
                task_info["thread"].join(timeout=3)
                
            # 从字典中移除任务
            del self._screenshot_tasks[task_id]
            
        self.logger.info(f"截图任务[{task_id}]已停止")
        return True
    
    def stop_all_tasks(self) -> None:
        """停止所有截图任务"""
        task_ids = list(self._screenshot_tasks.keys())
        for task_id in task_ids:
            self.stop_screenshot_task(task_id)
        self.logger.info("所有截图任务已停止")
    
    def get_task_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """获取任务状态"""
        with self._counter_lock:
            if task_id:
                if task_id not in self._screenshot_tasks:
                    return {"success": False, "message": f"截图任务[{task_id}]不存在"}
                    
                task_info = self._screenshot_tasks[task_id]
                return {
                    "success": True,
                    "task_id": task_id,
                    "running": task_info["thread"].is_alive(),
                    "interval": task_info["interval"],
                    "max_count": task_info["max_count"],
                    "start_time": task_info["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed": (datetime.now() - task_info["start_time"]).total_seconds()
                }
            else:
                # 返回所有任务的状态
                tasks = {}
                for tid, info in self._screenshot_tasks.items():
                    tasks[tid] = {
                        "running": info["thread"].is_alive(),
                        "interval": info["interval"],
                        "max_count": info["max_count"],
                        "start_time": info["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed": (datetime.now() - info["start_time"]).total_seconds()
                    }
                return {
                    "success": True,
                    "task_count": len(self._screenshot_tasks),
                    "tasks": tasks
                }
    
    def start_multi_screenshot_task(self, tasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """启动多个截图任务，返回任务ID字典
        
        Args:
            tasks: 任务列表，每个任务是一个字典，包含：
                  - interval: 截图间隔（秒）
                  - max_count: 最大截图数（可选）
                  - save_dir: 保存目录（可选）
                  - name: 任务名称（可选）
        
        Returns:
            Dict[str, str]: 任务名称到任务ID的映射
        """
        task_ids = {}
        
        self.logger.info(f"准备启动{len(tasks)}个截图任务")
        
        # 首先清空目标目录（如果需要）
        clear_target_dir = self.config_manager.get('adb.screenshot.clear_target_dir', False)
        if clear_target_dir:
            # 获取日期目录路径
            date_str = datetime.now().strftime(self.config_manager.get('adb.screenshot.date_format', '%Y%m%d'))
            save_dirs = set()
            
            # 收集所有不同的保存目录
            for task in tasks:
                save_dir = task.get('save_dir') or self.config_manager.get('adb.screenshot.local_temp_dir')
                save_dirs.add(save_dir)
            
            # 清空每个目录
            for save_dir in save_dirs:
                self.logger.info(f"准备清空目标目录: {save_dir}")
                try:
                    date_dir = os.path.join(save_dir, date_str)
                    os.makedirs(date_dir, exist_ok=True)
                    
                    # 清空日期目录中的文件
                    file_count = 0
                    for file in os.listdir(date_dir):
                        file_path = os.path.join(date_dir, file)
                        if os.path.isfile(file_path):
                            try:
                                os.remove(file_path)
                                file_count += 1
                            except Exception as e:
                                self.logger.warning(f"删除文件失败: {file_path}, 错误: {e}")
                    
                    self.logger.info(f"已清空日期目录: {date_dir}, 共删除{file_count}个文件")
                except Exception as e:
                    self.logger.error(f"清空目标目录时出错: {str(e)}")
        
        # 依次启动每个任务
        for i, task in enumerate(tasks):
            interval = task.get('interval', 5)
            max_count = task.get('max_count', None)
            save_dir = task.get('save_dir', None)
            name = task.get('name', f"任务{i+1}")
            
            # 启动任务并保存任务ID
            task_id = self.start_screenshot_task(
                interval=interval,
                max_count=max_count,
                save_dir=save_dir
            )
            
            task_ids[name] = task_id
            self.logger.info(f"已启动截图任务: {name} (ID: {task_id}), 间隔: {interval}秒, 最大数量: {max_count or '无限'}")
            
            # 添加短暂延迟，防止任务同时启动
            time.sleep(0.2)
        
        return task_ids

def run_project(config: Dict[str, Any]) -> None:
    """运行项目"""
    logger = get_logger(__name__)
    logger.info("启动项目")
    
    try:
        # 初始化截图管理器
        screenshot_manager = ScreenshotManager(ConfigManager(config))
        
        # 运行一次截图
        screenshot_path = screenshot_manager.take_screenshot()
        if screenshot_path:
            logger.info(f"初始截图已保存: {screenshot_path}")
        
        # 启动定时截图任务，从配置读取间隔时间
        interval = config.get("adb", {}).get("screenshot", {}).get("interval", 1)
        task_id = screenshot_manager.start_screenshot_task(interval=interval)
        logger.info(f"截图任务已启动: {task_id}, 间隔: {interval}秒")
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        finally:
            # 停止所有任务
            screenshot_manager.stop_all_tasks()
            logger.info("项目已停止")
    
    except Exception as e:
        logger.error(f"项目运行出错: {str(e)}")
        

# 创建全局实例
try:
    config_manager = ConfigManager("config/app_config.yaml")
    log_manager = LogManager()  # 使用新的日志管理器
except Exception as e:
    print(f"初始化配置或日志失败: {str(e)}")
    sys.exit(1)

# 获取预配置的日志记录器
def get_logger(name: str) -> logging.Logger:
    """获取预配置的日志记录器"""
    return log_manager.get_logger(name)

def main():
    """主函数"""
    try:
        # 初始化服务
        config_manager, screenshot_manager, game_ocr = initialize_services()
        
        # 创建菜单项
        menu_items = [
            {'name': '检查设备连接状态', 'handler': lambda: handle_device_check(screenshot_manager)},
            {'name': '启动ADB服务', 'handler': lambda: handle_adb_server_start(config_manager, screenshot_manager)},
            {'name': '执行单次截图', 'handler': lambda: handle_single_screenshot(screenshot_manager)},
            {'name': '启动定时截图任务', 'handler': lambda: handle_scheduled_screenshot(config_manager, screenshot_manager)},
            {'name': '单张OCR识别', 'handler': lambda: handle_ocr_test(game_ocr, config_manager)},
            {'name': '打印当前配置', 'handler': lambda: handle_show_config(config_manager)}
        ]
        
        # 获取默认任务配置
        default_task = config_manager.get('app.default_task', 0)
        
        # 如果配置了默认任务且有效，直接执行
        if default_task and 1 <= default_task <= len(menu_items):
            logger = get_logger(__name__)
            logger.info(f"执行默认任务: {default_task}")
            
            try:
                if default_task == 1:
                    handle_device_check(screenshot_manager)
                elif default_task == 2:
                    handle_adb_server_start(config_manager, screenshot_manager)
                elif default_task == 3:
                    handle_single_screenshot(screenshot_manager)
                elif default_task == 4:
                    handle_scheduled_screenshot(config_manager, screenshot_manager)
                elif default_task == 5:
                    handle_ocr_test(game_ocr, config_manager)
                elif default_task == 6:
                    handle_show_config(config_manager)
            except Exception as e:
                logger = get_logger(__name__)
                logger.error(f"默认任务执行失败: {str(e)}", exc_info=True)
                print(f"\n默认任务执行失败: {str(e)}")
                print("将显示主菜单...")
        
        # 主循环
        while True:
            print("\n==== 游戏辅助工具 ====")
            for i, item in enumerate(menu_items, 1):
                print(f"{i}. {item['name']}")
            print("====================")
            
            choice = input("请选择功能 (输入q退出): ").strip()
            
            if choice.lower() == 'q':
                print("\n正在退出程序...")
                break
                
            try:
                choice = int(choice)
                if 1 <= choice <= len(menu_items):
                    menu_items[choice-1]['handler']()
                else:
                    print("无效的选择，请重试")
            except ValueError:
                print("无效的输入，请重试")
                
    except KeyboardInterrupt:
        handle_keyboard_interrupt(screenshot_manager)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"\n程序运行出错: {str(e)}")
    finally:
        if screenshot_manager:
            screenshot_manager.stop_all_tasks()

# 以下是拆分出的子函数 --------------------------------------------

def initialize_services():
    """初始化配置和服务实例"""
    try:
        config_manager = ConfigManager("config/app_config.yaml")
        screenshot_manager = ScreenshotManager(config_manager)
        game_ocr = GameOCR("config/app_config.yaml")
        return config_manager, screenshot_manager, game_ocr
    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        sys.exit(1)

def display_main_menu() -> str:
    """显示主菜单并获取用户输入"""
    print("\n==== 游戏辅助工具 ====")
    menu_items = [
        "1. 检查设备连接状态",
        "2. 启动ADB服务",
        "3. 执行单次截图",
        "4. 启动定时截图任务",
        "5. 单张OCR识别",
        "6. 打印当前配置"
    ]
    print("\n".join(menu_items))
    print("====================")
    return input("请选择功能 (输入q退出): ").strip()

# 各功能处理函数
def handle_single_screenshot(screenshot_manager):
    """处理单次截图"""
    if not screenshot_manager.adb_helper.check_device_connection():
        print("无法执行截图: 没有检测到已连接的设备")
        return
    
    print("正在执行截图...")
    if path := screenshot_manager.take_screenshot():
        print(f"截图已保存至: {path}")
    else:
        print("截图失败，请检查设备连接和ADB配置")

def handle_scheduled_screenshot(config_manager, screenshot_manager):
    """处理定时截图任务"""
    if not screenshot_manager.adb_helper.check_device_connection():
        print("无法启动截图任务: 没有检测到已连接的设备")
        return
    
    interval = config_manager.get('adb.screenshot.interval', 1)
    print(f"启动截图任务，间隔: {interval}秒...")
    
    if (task_id := screenshot_manager.start_screenshot_task(interval=interval)) == "ERROR_NO_DEVICE":
        print("截图任务启动失败: 设备未连接")
        return
    
    print(f"截图任务已启动，任务ID: {task_id}\n按Ctrl+C停止...")
    monitor_screenshot_task(screenshot_manager)

def monitor_screenshot_task(screenshot_manager):
    """监控截图任务状态"""
    try:
        while True:
            if not screenshot_manager.adb_helper.check_device_connection():
                print("警告: 设备连接已断开，等待重新连接...")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
    finally:
        screenshot_manager.stop_all_tasks()
        print("任务已停止")

def handle_device_check(screenshot_manager):
    """处理设备检查"""
    print("开始检查设备连接状态...")
    if screenshot_manager.adb_helper.check_device_connection():
        print("设备已连接，可以开始截图任务")
    else:
        print("设备未连接，请连接设备后重试")
        print("提示: 请确保USB调试已开启，并且已授权此电脑连接")

def handle_adb_server_start(config_manager, screenshot_manager):
    """处理启动ADB服务"""
    print("正在启动ADB服务...")
    try:
        adb_path = config_manager.get_config()["adb"]["path"]
        start_cmd = [adb_path, 'start-server']
        logger.info(f"执行命令: {' '.join(start_cmd)}")
        
        result = subprocess.run(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        
        if result.returncode == 0:
            print("ADB服务启动成功")
            logger.info("ADB服务启动成功")
            # 重新检查设备连接
            time.sleep(1)  # 等待服务完全启动
            if screenshot_manager.adb_helper.check_device_connection():
                print("设备已连接，可以开始截图任务")
            else:
                print("设备未连接，请连接设备后重试")
        else:
            error_msg = result.stderr.decode('utf-8')
            print(f"ADB服务启动失败: {error_msg}")
            logger.error(f"ADB服务启动失败: {error_msg}")
    except Exception as e:
        print(f"启动ADB服务时出错: {str(e)}")
        logger.error(f"启动ADB服务时出错: {str(e)}")

def handle_show_config(config_manager):
    """处理显示当前配置"""
    print("\n当前配置:")
    print(f"ADB路径: {config_manager.get('adb.path')}")
    temp_dir = config_manager.get('environment.temp_dir', '/Users/mac/ai/temp')
    screenshot_dir = os.path.join(temp_dir, 'screenshots')
    print(f"临时文件目录: {temp_dir}")
    print(f"截图保存目录: {screenshot_dir}")
    print(f"单任务截图间隔: {config_manager.get('adb.screenshot.interval')}秒")
    print(f"清空目标目录: {config_manager.get('adb.screenshot.clear_target_dir')}")
    print(f"日期格式: {config_manager.get('adb.screenshot.date_format')}")

def handle_keyboard_interrupt(screenshot_manager):
    """处理键盘中断"""
    print("\n收到中断信号，正在停止...")
    if screenshot_manager:
        screenshot_manager.stop_all_tasks()
    print("程序已退出")
    sys.exit(0)

def handle_unexpected_error(e):
    """处理意外错误"""
    logger.error(f"执行任务异常: {str(e)}")
    print(f"发生未预期错误: {str(e)}\n程序已退出")

if __name__ == "__main__":
    # 初始化日志
    log_manager = LogManager()
    logger = log_manager.get_logger(__name__)
    
    # 初始化配置
    try:
        config_manager = ConfigManager("config/app_config.yaml")
        config = config_manager.get_config()
    except Exception as e:
        logger.error(f"配置加载失败: {str(e)}")
        sys.exit(1)
    
    # 调用主函数
    main() 
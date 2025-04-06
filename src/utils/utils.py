"""
工具模块 - 系统工具和辅助功能

提供系统工具、ADB工具和配置管理功能。
"""

import os, yaml, logging, subprocess, time, threading, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging.handlers
import uuid
try:
    PNGQUANT_AVAILABLE = True
except ImportError:
    PNGQUANT_AVAILABLE = False

# 从配置文件加载日志级别常量
def load_log_levels(config_path="config/app_config.yaml"):
    """从配置文件加载日志级别常量，如果找不到则抛出异常"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        if 'log_levels' not in config:
            raise ValueError("配置文件中未找到'log_levels'配置项")
            
        log_levels = config['log_levels']
        
        # 提取常量 - 不使用默认值
        if 'DEBUG' not in log_levels:
            raise ValueError("日志级别配置中未找到'DEBUG'项")
        if 'INFO' not in log_levels:
            raise ValueError("日志级别配置中未找到'INFO'项")
        if 'WARNING' not in log_levels:
            raise ValueError("日志级别配置中未找到'WARNING'项")
        if 'ERROR' not in log_levels:
            raise ValueError("日志级别配置中未找到'ERROR'项")
        
        DEBUG = log_levels['DEBUG']
        INFO = log_levels['INFO']
        WARNING = log_levels['WARNING']
        ERROR = log_levels['ERROR']
        
        # 提取模块日志级别 - 不使用默认值
        if 'default' not in log_levels:
            raise ValueError("日志级别配置中未找到'default'项")
        if 'adb' not in log_levels:
            raise ValueError("日志级别配置中未找到'adb'项")
        if 'screenshot' not in log_levels:
            raise ValueError("日志级别配置中未找到'screenshot'项")
        if 'ocr' not in log_levels:
            raise ValueError("日志级别配置中未找到'ocr'项")
        
        DEFAULT_LOG_LEVELS = {
            "default": log_levels['default'],
            "adb": log_levels['adb'],
            "screenshot": log_levels['screenshot'],
            "ocr": log_levels['ocr']
        }
        
        return DEBUG, INFO, WARNING, ERROR, DEFAULT_LOG_LEVELS

# 加载日志级别常量
DEBUG, INFO, WARNING, ERROR, DEFAULT_LOG_LEVELS = load_log_levels()

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """从YAML文件加载配置，如果加载失败则抛出异常"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"配置文件为空或格式不正确: {self.config_path}")
            return config
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config
    
    def get(self, key_path: str) -> Any:
        """使用点分隔的路径获取配置值，如果不存在则抛出异常"""
        keys = key_path.split('.')
        value = self.config
        
        # 尝试按照路径访问嵌套的字典
        for i, k in enumerate(keys):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                current_path = '.'.join(keys[:i+1])
                raise KeyError(f"配置项不存在: '{current_path}'")
                
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
        """
        初始化日志管理器
        
        Args:
            config: 日志配置，不能为None
            
        Raises:
            ValueError: 当config为None时
        """
        if config is None:
            raise ValueError("LogManager初始化错误: 配置不能为None")
        
        self.config = config
        self.log_dir = config['log_dir']  # 不提供默认值
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
        """获取屏幕截图"""
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
            else:
                # 确保使用绝对路径
                local_dir = os.path.abspath(local_dir)
            
            # 打印确认使用的保存目录
            print(f"ADB Helper使用的截图保存目录: {local_dir}")
            
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
            
            # 返回绝对路径
            absolute_path = os.path.abspath(local_path)
            # 打印完整路径以方便查看
            print(f"截图完整路径: {absolute_path}")
            self.logger.info(f"截图已保存至: {absolute_path}")
            return absolute_path
            
        except subprocess.TimeoutExpired:
            self.logger.error("截图操作超时")
            return None
        except Exception as e:
            self.logger.error(f"截图异常: {str(e)}")
            return None
    
    def execute_command(self, command: List[str]) -> Tuple[int, str, str]:
        """执行ADB命令"""
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
        self.logger = get_logger("screenshot.manager")
        self.config_manager = config_manager
        self.adb_helper = ADBHelper(config_manager.get_config()["adb"]["path"])
        self._screenshot_tasks = {}
        self._counter_lock = threading.Lock()
        self._last_screenshot_time = 0
        self._min_interval = 0.1
    
    def _log_and_print(self, message: str, level: str = "info", show_time: bool = False) -> None:
        """优化的日志输出格式"""
        if show_time:
            time_str = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{time_str}] {message}"
        else:
            formatted_message = message
        
        print(formatted_message)
        getattr(self.logger, level)(message)

    def take_screenshot(self, save_dir: Optional[str] = None) -> Optional[str]:
        """优化后的截图输出"""
        try:
            # 控制截图速率
            current_time = time.time()
            with self._counter_lock:
                if current_time - self._last_screenshot_time < self._min_interval:
                    time.sleep(self._min_interval - (current_time - self._last_screenshot_time))
                self._last_screenshot_time = time.time()
            
            if not self.adb_helper._device_connected:
                self._log_and_print("设备未连接，请连接设备后重试", level="error")
                return None
            
            save_dir = save_dir or self.config_manager.get('environment.screenshots_dir')
            print(f"\n使用截图保存目录: {save_dir}")
            
            path = self.adb_helper.take_screenshot(local_dir=save_dir)
            if path:
                filename = os.path.basename(path)
                dir_name = os.path.dirname(path)
                file_size = os.path.getsize(path) / 1024  # 转换为KB
                self._log_and_print(f"✓ 截图成功 | 文件名: {filename} | 大小: {file_size:.1f}KB | 路径: {dir_name}", "info")
            else:
                self._log_and_print("✗ 截图失败，请检查设备连接", "warning")
            return path
        except Exception as e:
            self._log_and_print(f"✗ 截图异常: {str(e)}", "error")
            return None

    def start_screenshot_task(self, interval: int = 5, max_count: Optional[int] = None, 
                             save_dir: Optional[str] = None) -> str:
        """启动定时截图任务"""
        if not self.adb_helper._device_connected:
            self._log_and_print("设备未连接，无法启动截图任务", level="error")
            return "ERROR_NO_DEVICE"
        
        task_id = str(uuid.uuid4())
        target_dir = save_dir or self.config_manager.get('environment.screenshots_dir')
        
        # 处理目录清理
        if self.config_manager.get('adb.screenshot.clear_target_dir'):
            self._clear_target_directory(target_dir)
        
        # 创建并启动任务
        task_info = self._create_task_info(interval, max_count, save_dir)
        self._start_task_thread(task_id, task_info)
        
        return task_id

    def _clear_target_directory(self, target_dir: str) -> None:
        """清理目标目录"""
        try:
            date_str = datetime.now().strftime(self.config_manager.get('adb.screenshot.date_format'))
            date_dir = os.path.join(target_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            file_count = sum(1 for f in os.listdir(date_dir) 
                           if os.path.isfile(os.path.join(date_dir, f)) 
                           and os.remove(os.path.join(date_dir, f)) is None)
            
            self.logger.info(f"已清空日期目录: {date_dir}, 共删除{file_count}个文件")
            self.adb_helper._screenshot_counter = 0
            
        except Exception as e:
            self.logger.error(f"清空目标目录时出错: {str(e)}")

    def _create_task_info(self, interval: int, max_count: Optional[int], save_dir: Optional[str]) -> dict:
        """创建任务信息"""
        stop_event = threading.Event()
        return {
            "thread": None,  # 将在_start_task_thread中设置
            "stop_event": stop_event,
            "interval": interval,
            "max_count": max_count,
            "start_time": datetime.now(),
            "save_dir": save_dir
        }

    def _start_task_thread(self, task_id: str, task_info: dict) -> None:
        """启动任务线程"""
        thread = threading.Thread(
            target=self._screenshot_task_worker,
            args=(task_id, task_info),
            daemon=True
        )
        task_info["thread"] = thread
        
        with self._counter_lock:
            self._screenshot_tasks[task_id] = task_info
        thread.start()

    def _screenshot_task_worker(self, task_id: str, task_info: dict) -> None:
        """截图任务工作线程"""
        count = 0
        self.logger.info(f"截图任务[{task_id}]已启动, 间隔: {task_info['interval']}秒")
        
        try:
            last_time = 0
            while not task_info['stop_event'].is_set():
                # 检查是否达到最大次数
                if task_info['max_count'] and count >= task_info['max_count']:
                    self.logger.info(f"截图任务[{task_id}]已完成, 共截图{count}张")
                    break
                
                # 检查设备连接状态
                if not self.adb_helper._device_connected:
                    self._log_and_print("设备未连接，等待设备连接...", level="warning")
                    # 等待一段时间后重试
                    if task_info['stop_event'].wait(timeout=5):
                        break
                    continue
                
                current_time = time.time()
                # 动态调整等待时间，确保截图间隔准确
                if current_time - last_time < task_info['interval']:
                    wait_time = min(0.5, task_info['interval'] - (current_time - last_time))
                    if task_info['stop_event'].wait(timeout=wait_time):
                        break
                    continue
                
                # 获取截图
                screenshot_path = self.take_screenshot(save_dir=task_info['save_dir'])
                if screenshot_path:
                    count += 1
                    last_time = time.time()
                else:
                    # 截图失败时等待较长时间
                    self._log_and_print("截图失败，等待重试...", level="warning")
                    if task_info['stop_event'].wait(timeout=5):
                        break
                    
        except Exception as e:
            self.logger.error(f"截图任务[{task_id}]执行失败: {str(e)}")
        finally:
            # 任务结束时从字典中移除
            with self._counter_lock:
                if task_id in self._screenshot_tasks:
                    del self._screenshot_tasks[task_id]
            self.logger.info(f"截图任务[{task_id}]已停止")

    def stop_all_tasks(self) -> None:
        """停止所有截图任务"""
        with self._counter_lock:
            for task_id, task_info in list(self._screenshot_tasks.items()):
                task_info['stop_event'].set()
                if task_info['thread'].is_alive():
                    task_info['thread'].join(timeout=3)
                del self._screenshot_tasks[task_id]
        self.logger.info("所有截图任务已停止")

# 创建全局实例
config_manager = ConfigManager("config/app_config.yaml")
log_config = {
    'log_dir': config_manager.get('environment.log_dir')
}
log_manager = LogManager(log_config)

# 获取预配置的日志记录器
def get_logger(name: str) -> logging.Logger:
    """获取预配置的日志记录器"""
    return log_manager.get_logger(name)

logger = get_logger(__name__)

def main():
    """主函数"""
    screenshot_manager = None
    
    try:
        # 初始化服务
        config_manager, screenshot_manager = initialize_services()
        
        # 创建菜单项
        menu_items = [
            {'name': '检查设备连接状态', 'handler': lambda: handle_device_check(screenshot_manager)},
            {'name': '启动ADB服务', 'handler': lambda: handle_adb_server_start(config_manager, screenshot_manager)},
            {'name': '执行单次截图', 'handler': lambda: handle_single_screenshot(screenshot_manager)},
            {'name': '启动定时截图任务', 'handler': lambda: handle_scheduled_screenshot(config_manager, screenshot_manager)},
            {'name': '打印当前配置', 'handler': lambda: handle_show_config(config_manager)}
        ]
        
        # 获取默认任务配置
        default_task = config_manager.get('app.default_task')
        
        # 如果配置了默认任务且有效，直接执行
        if default_task and 1 <= default_task <= len(menu_items):
            logger = get_logger(__name__)
            logger.info(f"执行默认任务: {default_task}")
            
            try:
                menu_items[default_task-1]['handler']()
            except Exception as e:
                logger = get_logger(__name__)
                logger.error(f"默认任务执行失败: {str(e)}", exc_info=True)
                print(f"\n默认任务执行失败: {str(e)}")
        
        # 主循环
        while True:
            choice = display_menu()  # display_menu 函数已经包含了菜单显示
            
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
        # 使用全局配置管理器
        global config_manager
        screenshot_manager = ScreenshotManager(config_manager)
        return config_manager, screenshot_manager
    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        sys.exit(1)

# 各功能处理函数
def handle_single_screenshot(screenshot_manager):
    """处理单次截图"""
    if not screenshot_manager.adb_helper.check_device_connection():
        print("无法执行截图: 没有检测到已连接的设备")
        return
    
    print("正在执行截图...")
    screenshot_path = screenshot_manager.take_screenshot()
    if screenshot_path:
        print(f"截图成功，文件路径: {screenshot_path}")

def handle_scheduled_screenshot(config_manager, screenshot_manager):
    """处理定时截图任务"""
    if not screenshot_manager.adb_helper.check_device_connection():
        print("无法启动截图任务: 没有检测到已连接的设备")
        return
    
    interval = config_manager.get('adb.screenshot.interval')
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

        result = subprocess.run(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        
        if result.returncode == 0:
            print("ADB服务启动成功")
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
    screenshot_dir = config_manager.get('environment.screenshots_dir')
    print(f"截图保存目录: {screenshot_dir}")
    print(f"单任务截图间隔: {config_manager.get('adb.screenshot.interval')}秒")
    print(f"清空目标目录: {'是' if config_manager.get('adb.screenshot.clear_target_dir') else '否'}")
    print(f"日期格式: {config_manager.get('adb.screenshot.date_format')}")

def handle_keyboard_interrupt(screenshot_manager):
    print("\n收到中断信号，正在停止...")
    if screenshot_manager:
        screenshot_manager.stop_all_tasks()
    print("程序已退出")
    sys.exit(0)

def display_menu():
    """显示简化的菜单"""
    menu = """
==== 游戏辅助工具 ====
1. 检查设备连接状态
2. 启动ADB服务
3. 执行单次截图
4. 启动定时截图任务
5. 打印当前配置
====================
请选择功能 (q退出): """
    return input(menu)

if __name__ == "__main__":
    try:
        config_manager = ConfigManager("config/app_config.yaml")
        config = config_manager.get_config()
        log_config = {
            'log_dir': config_manager.get('environment.log_dir')
        }
        log_manager = LogManager(log_config)
        logger = get_logger(__name__)
    except Exception as e:
        print(f"配置加载失败: {str(e)}")
        sys.exit(1)
    main() 
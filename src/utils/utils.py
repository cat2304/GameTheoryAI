"""
工具模块

提供系统工具、ADB工具和游戏OCR功能。
"""

import os, cv2, yaml, numpy as np, logging, subprocess, time, threading, signal, sys, json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import uuid
import io
import logging.handlers  # 添加缺失的导入
try:
    import pngquant  # 尝试导入pngquant库
    PNGQUANT_AVAILABLE = True
except ImportError:
    PNGQUANT_AVAILABLE = False

# 在LogManager类上方添加以下常量定义
LOG_CONFIG = {
    "base_format": "%(asctime)s [%(levelname).4s] %(name)s:%(lineno)d - %(message)s",
    "file_format": "%(asctime)s [%(levelname).4s] %(name)s:%(lineno)d - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_levels": {
        "root": "INFO",
        "utils": "DEBUG",
        "adb": "INFO",
        "ocr": "DEBUG"
    },
    "log_file": {
        "enabled": True,
        "path": "logs/app.log",
        "max_size": 10,  # MB
        "backup_count": 7,
        "encoding": "utf-8"
    }
}

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path="config/app_config.yaml"):
        self.config_path = config_path
        self.config = {}
        
        try:
            self._load_config()
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            self.config = {}
    
    def _load_config(self):
        """加载配置文件"""
        if not os.path.isfile(self.config_path):
            print(f"配置文件不存在: {self.config_path}")
            return
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
    
    def get(self, key=None, default=None):
        """获取配置值"""
        if key is None:
            return self.config
            
        # 支持点号分隔的键
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key, value):
        """设置配置值"""
        if key is None:
            return False
            
        # 支持点号分隔的键
        keys = key.split('.')
        config = self.config
        
        # 遍历键路径
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # 设置最终值
        config[keys[-1]] = value
        return True
    
    def get_config(self):
        """获取完整配置"""
        return self.config
        
    def save(self):
        """保存配置"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        return True

class LogManager:
    """增强版日志管理器"""
    
    def __init__(self, config_manager=None):
        self.config = LOG_CONFIG  # 默认使用内置配置
        self._configure_logging()
        
    def _configure_logging(self):
        """配置日志系统"""
        # 创建根日志器
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(self.config["log_levels"]["root"])
        
        # 清除已有处理器
        if self.root_logger.handlers:
            for handler in self.root_logger.handlers[:]:
                self.root_logger.removeHandler(handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            fmt=self.config["base_format"],
            datefmt=self.config["date_format"]
        )
        console_handler.setFormatter(console_formatter)
        self.root_logger.addHandler(console_handler)
        
        # 文件处理器（如果启用）
        if self.config["log_file"]["enabled"]:
            self._setup_file_handler()
        
        # 设置子模块日志级别
        self._set_module_levels()
    
    def _setup_file_handler(self):
        """配置文件日志处理器"""
        try:
            log_path = self.config["log_file"]["path"]
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=self.config["log_file"]["max_size"] * 1024 * 1024,
                backupCount=self.config["log_file"]["backup_count"],
                encoding=self.config["log_file"]["encoding"]
            )
            
            file_formatter = logging.Formatter(
                fmt=self.config["file_format"],
                datefmt=self.config["date_format"]
            )
            file_handler.setFormatter(file_formatter)
            self.root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"初始化文件日志失败: {str(e)}")
            raise
    
    def _set_module_levels(self):
        """设置模块级日志级别"""
        for module, level in self.config["log_levels"].items():
            if module == "root":
                continue
            logger = logging.getLogger(module)
            logger.setLevel(level)
    
    def get_logger(self, name):
        """获取预配置的日志记录器"""
        logger = logging.getLogger(name)
        
        # 添加NullHandler防止无处理器警告
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
            
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
            print(f"设备连接初始化失败: {str(e)}")
    
    def check_device_connection(self) -> bool:
        """检查设备连接状态
        
        Returns:
            bool: 设备是否已连接
        """
        self.logger.debug("开始检查设备连接状态...")
        try:
            command = [self.adb_path, 'devices']
            command_str = ' '.join(command)
            self.logger.debug(f"执行检测命令: {command_str}")
            
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8')
                self.logger.error(f"设备检测命令执行失败: {error_msg}")
                print(f"设备检测命令执行失败: {error_msg}")
                print(f"您可以手动在终端执行以下命令验证ADB是否工作正常:")
                print(f"  {command_str}")
                return False
            
            output = result.stdout.decode('utf-8').strip()
            self.logger.debug(f"ADB设备列表输出:\n{output}")
            
            devices = self._get_devices()
            was_connected = self._device_connected
            
            if devices:
                self._device_connected = True
                if not was_connected:
                    self.logger.info(f"设备已连接: {devices[0]}")
                    print(f"设备已连接: {devices[0]}")
                return True
            else:
                self._device_connected = False
                self.logger.warning("设备检测结果: 没有找到已连接的设备")
                print("\n没有找到已连接的设备，请尝试以下操作:")
                print("1. 确保USB线缆正常连接")
                print("2. 在设备上启用USB调试模式")
                print("3. 在设备上允许USB调试授权")
                print("4. 检查ADB服务是否正常运行")
                print("\n您可以在终端执行以下命令进行手动检测:")
                print(f"  {command_str}")
                print(f"  {self.adb_path} kill-server")
                print(f"  {self.adb_path} start-server")
                print(f"  {self.adb_path} devices")
                
                return False
        except subprocess.TimeoutExpired:
            self.logger.exception("ADB设备检测超时")
            return False
        except Exception as e:
            self.logger.error("设备检测发生未预期错误", exc_info=True)
            self._device_connected = False
            print(f"设备连接检测失败: {str(e)}")
            print("\n请尝试手动在终端执行以下命令重启ADB服务:")
            print(f"  {self.adb_path} kill-server")
            print(f"  {self.adb_path} start-server")
            print(f"  {self.adb_path} devices")
            return False
    
    def _get_devices(self) -> List[str]:
        """获取已连接的设备列表"""
        try:
            command = [self.adb_path, 'devices']
            self.logger.debug(f"执行命令: {' '.join(command)}")
            
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8')
                self.logger.error(f"获取设备失败，命令返回错误: {error_msg}")
                return []
            
            output = result.stdout.decode('utf-8').strip()
            
            devices = []
            lines = output.split('\n')
            for line in lines[1:]:  # 跳过标题行
                if line.strip() and not line.strip().startswith('*') and '\tdevice' in line:
                    devices.append(line.split('\t')[0])
            
            self.logger.debug(f"检测到{len(devices)}个设备: {devices}")
            return devices
        except subprocess.TimeoutExpired:
            self.logger.error("获取设备列表超时，可能是ADB服务无响应")
            return []
        except Exception as e:
            self.logger.error(f"获取设备列表失败: {str(e)}")
            return []
    
    def take_screenshot(self, local_dir: Optional[str] = None) -> Optional[str]:
        """获取设备截图
        
        Args:
            local_dir: 可选，截图保存目录
            
        Returns:
            str: 截图本地路径，如果失败则返回None
        """
        if not self.check_device_connection():
            self.logger.error("截图失败: 未检测到已连接设备")
            return None
            
        try:
            # 确保截图目录存在
            date_str = datetime.now().strftime('%Y%m%d')
            # 如果未提供目录，使用配置文件中的设置
            temp_dir = config_manager.get('environment.temp_dir', '/Users/mac/ai/temp')
            local_dir = local_dir or os.path.join(temp_dir, 'screenshots', date_str)
            os.makedirs(local_dir, exist_ok=True)
            
            # 生成截图保存路径 - 使用简单的递增数字作为文件名
            self._screenshot_counter += 1
            screenshot_filename = f"{self._screenshot_counter}.png"
            local_path = os.path.join(local_dir, screenshot_filename)
            
            # 执行截图命令
            self.logger.debug(f"开始执行截图命令")
            screenshot_cmd = [self.adb_path, 'exec-out', 'screencap', '-p']
            self.logger.debug(f"执行命令: {' '.join(screenshot_cmd)}")
            
            # 增加超时时间，有些设备可能需要更长时间
            with open(local_path, 'wb') as f:
                process = subprocess.run(
                    screenshot_cmd, 
                    stdout=f,
                    stderr=subprocess.PIPE,
                    timeout=10  # 增加超时时间到10秒
                )
            
            if process.returncode != 0:
                error_msg = process.stderr.decode('utf-8') if process.stderr else "未知错误"
                self.logger.error(f"截图命令执行失败: {error_msg}")
                print(f"截图命令执行失败: {error_msg}")
                return None
                
            # 验证截图是否成功保存
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                self.logger.info(f"截图成功: {local_path}")
                return local_path
            else:
                self.logger.error(f"截图保存失败: 文件不存在或大小为0")
                if os.path.exists(local_path):
                    os.remove(local_path)  # 清理空文件
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("截图命令执行超时")
            print("截图命令执行超时，请检查设备连接是否稳定")
            return None
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
            print(f"截图过程中出现异常: {str(e)}")
            return None

class GameOCR:
    """游戏OCR工具类"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.logger = get_logger("ocr.core")  # 使用模块化日志名称
        
        if isinstance(config_path, str):
            config_path = Path(config_path)
            
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        self.config = {
            "preprocessing": {"blur_kernel": (3, 3), "threshold_min": 200, "threshold_max": 255, "canny_min": 30, "canny_max": 120, "dilate_kernel": (2, 2), "dilate_iterations": 1, "debug_output": True},
            "detection": {"min_area": 3000, "max_area": 50000, "aspect_ratio_min": 0.6, "aspect_ratio_max": 1.5, "min_width": 30, "max_width": 300, "y_threshold": 150, "expected_elements": 13, "element_gap": 5, "merge_threshold": 20, "max_retries": 3, "threshold_reduction": 0.8},
            "recognition": {"text_area_ratio": 0.4, "text_threshold": 150, "red_threshold": 0.02, "green_threshold": 0.02, "circle_threshold": 0.7, "debug_features": True}
        }
        
        self.logger.info("游戏状态识别工具初始化完成")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图片"""
        try:
            # 如果是单通道图像，转换为三通道
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊去噪
            blur = cv2.GaussianBlur(gray, 
                                  self.config['preprocessing']['blur_kernel'],
                                  0)
            
            # 自适应二值化
            binary = cv2.adaptiveThreshold(blur, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Canny边缘检测
            edges = cv2.Canny(binary,
                            self.config['preprocessing']['canny_min'],
                            self.config['preprocessing']['canny_max'])
            
            # 膨胀操作，连接边缘
            kernel = np.ones(self.config['preprocessing']['dilate_kernel'], np.uint8)
            dilated = cv2.dilate(edges, kernel, 
                               iterations=self.config['preprocessing']['dilate_iterations'])
            
            return dilated
            
        except Exception as e:
            self.logger.error(f"图片预处理失败: {str(e)}")
            raise
    
    def recognize_text(self, image_path: str) -> str:
        """识别图片中的文字"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 预处理图片
            processed = self.preprocess_image(image)
            
            # 使用Tesseract进行OCR识别
            text = pytesseract.image_to_string(processed, lang='chi_sim+eng')
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"文字识别失败: {str(e)}")
            raise
    
    def recognize_image(self, image_path: str) -> Dict[str, Any]:
        """识别游戏截图"""
        self.logger.info(f"开始识别图片: {image_path}")
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 预处理图片
            processed = self.preprocess_image(image)
            
            # 查找游戏元素
            elements = self.find_game_elements(processed)
            
            # 对每个元素进行OCR识别
            results = []
            for x, y, w, h in elements:
                # 提取元素区域
                roi = image[y:y+h, x:x+w]
                # 识别文字
                text = pytesseract.image_to_string(roi, lang='chi_sim+eng')
                results.append({
                    'region': (x, y, w, h),
                    'result': text.strip()
                })
            
            self.logger.debug(f"识别到{len(elements)}个元素")
            return {
                'success': True,
                'elements': results
            }
            
        except Exception as e:
            self.logger.error(f"图片识别失败: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def find_game_elements(self, image: np.ndarray, retry_count: int = 0) -> List[Tuple[int, int, int, int]]:
        """查找游戏元素区域
        
        Args:
            image: 输入图片
            retry_count: 当前重试次数
            
        Returns:
            List[Tuple[int, int, int, int]]: 游戏元素区域列表
        """
        try:
            height, width = image.shape[:2]
            self.logger.debug(f"图片尺寸: {width}x{height}")
            
            # 只处理图片底部区域
            y_threshold = self.config['detection']['y_threshold']
            bottom_region = image[height - y_threshold:, :]
            
            # 查找轮廓
            contours, hierarchy = cv2.findContours(bottom_region,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            
            elements = []
            config = self.config['detection']
            
            # 计算当前阈值
            current_min_area = config['min_area'] * (config['threshold_reduction'] ** retry_count)
            current_min_width = config['min_width'] * (config['threshold_reduction'] ** retry_count)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < current_min_area or area > config['max_area']:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                y = height - y_threshold + y  # 调整y坐标
                
                aspect_ratio = w / float(h)
                if (aspect_ratio < config['aspect_ratio_min'] or 
                    aspect_ratio > config['aspect_ratio_max']):
                    continue
                
                if w < current_min_width or w > config['max_width']:
                    continue
                
                elements.append((x, y, w, h))
            
            # 检查识别到的元素数是否合理
            if len(elements) < 10 and retry_count < config['max_retries']:  # 最少应该有10个元素
                self.logger.warning(f"识别到的游戏元素数量过少: {len(elements)}")
                self.logger.info(f"尝试第{retry_count + 1}次调整参数重新识别...")
                return self.find_game_elements(image, retry_count + 1)
            
            # 按x坐标排序
            elements.sort(key=lambda x: x[0])
            
            # 检查元素的间距
            for i in range(len(elements) - 1):
                gap = elements[i+1][0] - (elements[i][0] + elements[i][2])
                if gap > config['element_gap']:
                    self.logger.warning(f"检测到异常间距: {gap}像素 (位置: {i+1})")
            
            return elements
            
        except Exception as e:
            self.logger.error(f"查找游戏元素区域失败: {str(e)}")
            raise

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
            {'name': '打印当前配置', 'handler': lambda: handle_show_config(config_manager)},
            {'name': '批量OCR识别', 'handler': lambda: handle_batch_ocr(game_ocr, config_manager)}
        ]
        
        # 获取默认任务配置
        default_task = config_manager.get('app.default_task', 0)
        
        # 如果配置了默认任务且有效，直接执行
        if default_task and 1 <= default_task <= 7:
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
                elif default_task == 7:
                    handle_batch_ocr(game_ocr, config_manager)
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
        "6. 打印当前配置",
        "7. 批量OCR识别"
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

def handle_ocr_test(game_ocr, config_manager):
    """OCR图片识别功能（自动获取最新截图）"""
    logger = get_logger(__name__)
    logger.info("启动OCR识别流程")

    try:
        # 从配置获取截图目录
        temp_dir = config_manager.get('environment.temp_dir', '/Users/mac/ai/temp')
        screenshot_dir = os.path.join(temp_dir, 'screenshots')
        
        # 自动查找最新截图
        latest_img = find_latest_screenshot(screenshot_dir)
        
        if not latest_img:
            print("\n警告：未找到任何截图文件")
            print("请先执行截图操作或手动指定图片路径")
            return

        # 执行识别
        print(f"\n正在自动识别最新截图: {latest_img}")
        start_time = time.time()
        
        result = game_ocr.recognize_image(latest_img)
        
        elapsed_time = time.time() - start_time
        print(f"\n识别完成（耗时{elapsed_time:.2f}秒）")

        # 输出结果
        print(f"图片路径: {latest_img}")
        if result['success']:
            # 生成结果文件路径
            txt_path = os.path.splitext(latest_img)[0] + ".txt"
            
            # 构建结果内容
            content = [
                f"图片路径: {latest_img}",
                f"识别耗时: {elapsed_time:.2f}秒",
                f"识别元素数量: {len(result['elements'])}",
                "\n识别结果:"
            ]
            
            for idx, element in enumerate(result['elements'], 1):
                content.append(f"{idx}. 位置: {element['region']}")
                content.append(f"   结果: {element['result']}")
                content.append("")  # 空行分隔
            
            # 写入文件
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(content))
                print(f"\n识别结果已保存至: {txt_path}")
                logger.info(f"OCR结果已保存到 {txt_path}")
            except Exception as e:
                print(f"\n警告：结果文件保存失败 - {str(e)}")
                logger.error(f"结果文件保存失败: {str(e)}")
        else:
            print(f"识别失败: {result.get('error', '未知错误')}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        logger.error(f"OCR识别异常: {str(e)}", exc_info=True)

def find_latest_screenshot(base_dir: str) -> Optional[str]:
    """查找最新截图文件"""
    try:
        # 按日期目录排序（格式：YYYYMMDD）
        date_dirs = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()],
            reverse=True
        )
        
        # 遍历所有日期目录
        for date_dir in date_dirs:
            dir_path = os.path.join(base_dir, date_dir)
            # 按文件名排序（数字.png）
            screenshots = sorted(
                [f for f in os.listdir(dir_path) if f.lower().endswith('.png')],
                key=lambda x: int(x.split('.')[0]),
                reverse=True
            )
            
            if screenshots:
                return os.path.join(dir_path, screenshots[0])
                
        return None
        
    except Exception as e:
        logger.error(f"查找截图失败: {str(e)}")
        return None

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

def handle_batch_ocr(game_ocr, config_manager):
    """批量OCR识别功能"""
    logger = get_logger(__name__)
    logger.info("启动批量OCR识别流程")
    
    try:
        # 从配置获取基础目录
        base_dir = config_manager.get('environment.temp_dir', '/Users/mac/ai/temp')
        screenshot_dir = os.path.join(base_dir, 'screenshots')
        
        # 获取所有图片文件（按修改时间排序）
        all_files = find_all_screenshots(screenshot_dir)
        
        if not all_files:
            print("\n警告：未找到任何截图文件")
            return

        # 创建汇总文件
        summary_path = os.path.join(base_dir, "ocr_summary.txt")
        success_count = 0
        fail_count = 0
        
        print(f"\n开始批量处理 {len(all_files)} 个文件...")
        
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write(f"OCR识别汇总报告 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            summary_file.write("="*50 + "\n\n")
            
            for idx, img_path in enumerate(all_files, 1):
                try:
                    print(f"\n[{idx}/{len(all_files)}] 正在处理: {img_path}")
                    start_time = time.time()
                    
                    # 执行识别
                    result = game_ocr.recognize_image(img_path)
                    elapsed_time = time.time() - start_time
                    
                    # 构建结果内容
                    log_lines = [
                        f"文件: {img_path}",
                        f"状态: {'成功' if result['success'] else '失败'}",
                        f"耗时: {elapsed_time:.2f}秒",
                    ]
                    
                    if result['success']:
                        success_count += 1
                        log_lines.append(f"识别元素数量: {len(result['elements'])}")
                        # 保存独立结果文件
                        save_single_result(img_path, result, elapsed_time)
                    else:
                        fail_count += 1
                        log_lines.append(f"错误信息: {result.get('error', '未知错误')}")
                    
                    # 打印并记录日志
                    current_log = "\n".join(log_lines)
                    print(current_log)
                    summary_file.write(current_log + "\n\n")
                    
                except Exception as e:
                    fail_count += 1
                    error_msg = f"处理异常: {str(e)}"
                    print(error_msg)
                    summary_file.write(f"文件: {img_path}\n状态: 异常失败\n错误信息: {error_msg}\n\n")
                    logger.error(f"文件处理异常: {img_path} - {str(e)}", exc_info=True)
                    continue

        # 生成最终汇总
        final_summary = (
            f"\n处理完成！成功: {success_count} 个，失败: {fail_count} 个\n"
            f"详细结果请查看: {summary_path}"
        )
        print(final_summary)
        logger.info(final_summary)

    except Exception as e:
        print(f"批量处理发生错误: {str(e)}")
        logger.error(f"批量OCR异常: {str(e)}", exc_info=True)

def find_all_screenshots(base_dir: str) -> List[str]:
    """查找所有截图文件（按修改时间排序）"""
    try:
        all_files = []
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(base_dir):
            # 按修改时间排序目录（最新修改的优先）
            dirs.sort(key=lambda d: os.path.getmtime(os.path.join(root, d)), reverse=True)
            
            # 收集图片文件（支持PNG/JPG）
            for file in sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f)), reverse=True):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)
                    
        return all_files
        
    except Exception as e:
        logger.error(f"查找截图文件失败: {str(e)}")
        return []

def save_single_result(img_path: str, result: dict, elapsed_time: float):
    """保存单个文件结果"""
    try:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        content = [
            f"图片路径: {img_path}",
            f"识别耗时: {elapsed_time:.2f}秒",
            f"识别元素数量: {len(result['elements'])}",
            "\n识别结果:"
        ]
        
        for idx, element in enumerate(result['elements'], 1):
            content.append(f"{idx}. 位置: {element['region']}")
            content.append(f"   结果: {element['result']}")
            content.append("")
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
            
    except Exception as e:
        logger.error(f"结果文件保存失败: {str(e)}")

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
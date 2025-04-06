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
try:
    import pngquant  # 尝试导入pngquant库
    PNGQUANT_AVAILABLE = True
except ImportError:
    PNGQUANT_AVAILABLE = False

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
    """日志管理器"""
    
    def __init__(self, config_file=None):
        self.log_level = logging.INFO
        self.log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        self.log_date_format = '%Y-%m-%d %H:%M:%S'
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        # 配置根日志器
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            datefmt=self.log_date_format
        )
        
    def get_logger(self, name):
        """获取日志记录器"""
        return logging.getLogger(name)

class ADBHelper:
    """ADB工具辅助类"""
    
    def __init__(self, adb_path):
        self.logger = log_manager.get_logger(__name__)
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
        try:
            self.logger.info("开始检查设备连接状态...")
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
        except Exception as e:
            self.logger.error(f"检查设备连接失败: {str(e)}")
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
        self.logger = log_manager.get_logger(__name__)
        
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
            
            return {
                'success': True,
                'elements': results
            }
            
        except Exception as e:
            self.logger.error(f"图片识别失败: {str(e)}")
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
        self.logger = log_manager.get_logger(__name__)
        self.config_manager = config_manager
        self.adb_helper = ADBHelper(config_manager.get_config()["adb"]["path"])
        self._screenshot_tasks = {}
        self._counter_lock = threading.Lock()
        self._last_screenshot_time = 0  # 记录上次截图时间，控制速率
        self._min_interval = 0.1  # 最小截图间隔（秒），防止过度截图
    
    def take_screenshot(self, save_dir: Optional[str] = None) -> Optional[str]:
        """获取单张截图"""
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
                self.logger.error("截图失败，请检查设备连接")
                print("截图失败，请检查设备连接")
            return path
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
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
                    
                    current_time = time.time()
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
    logger = log_manager.get_logger(__name__)
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
    log_manager = LogManager()
except Exception as e:
    print(f"初始化配置或日志失败: {str(e)}")
    # 创建默认实例
    config_manager = ConfigManager()
    log_manager = LogManager()

# 导出常用函数
def get_config(key=None, default=None):
    """获取配置值"""
    return config_manager.get(key, default)

def set_config(key, value):
    """设置配置值"""
    return config_manager.set(key, value)

def get_logger(name):
    """获取日志记录器"""
    return log_manager.get_logger(name)

def main():
    """主函数"""
    # 配置和实例化工具类
    try:
        config_manager = ConfigManager("config/app_config.yaml")
        config = config_manager.get_config()
    except Exception as e:
        logger.error(f"配置加载失败: {str(e)}")
        sys.exit(1)
    
    screenshot_manager = ScreenshotManager(config_manager)
    
    # 验证ADB工具是否可用
    try:
        logger.info("开始验证ADB工具...")
        adb_path = config_manager.get_config()["adb"]["path"]
        logger.info(f"ADB工具路径: {adb_path}")
        
        result = subprocess.run(
            [adb_path, 'version'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=5
        )
        
        if result.returncode == 0:
            version_info = result.stdout.decode('utf-8').strip()
            logger.info(f"ADB工具验证成功: {version_info}")
            print(f"ADB工具验证成功")
        else:
            error_msg = result.stderr.decode('utf-8')
            logger.error(f"ADB工具验证失败: {error_msg}")
            print(f"ADB工具验证失败: {error_msg}")
            return
    except subprocess.TimeoutExpired:
        logger.error("ADB工具验证超时")
        print("ADB工具验证超时，请检查ADB服务是否响应")
        return
    except Exception as e:
        logger.error(f"ADB工具验证异常: {str(e)}")
        print(f"ADB工具验证异常: {str(e)}")
        return
    
    # 检查设备连接
    logger.info("开始检查设备连接状态...")
    if not screenshot_manager.adb_helper.check_device_connection():
        logger.warning("首次检测未发现设备，请确保设备已正确连接并启用USB调试")
    
    # 显示菜单
    print("\n==== 游戏辅助工具 ====")
    print("1. 执行单次截图")
    print("2. 启动定时截图任务 (按Ctrl+C停止)")
    print("3. 启动多间隔截图任务 (按Ctrl+C停止)")
    print("4. 检查设备连接状态")
    print("5. 启动ADB服务")
    print("6. 打印当前配置")
    print("====================\n")
    
    try:
        choice = input("请选择功能: ")
        
        if choice == "1":
            # 单次截图
            if not screenshot_manager.adb_helper.check_device_connection():
                print("无法执行截图: 没有检测到已连接的设备")
                return
                
            print("正在执行截图...")
            path = screenshot_manager.take_screenshot()
            if path:
                print(f"截图已保存至: {path}")
            else:
                print("截图失败，请检查设备连接和ADB配置")
        
        elif choice == "2":
            # 定时截图任务
            if not screenshot_manager.adb_helper.check_device_connection():
                print("无法启动截图任务: 没有检测到已连接的设备")
                return
            
            # 从配置文件读取间隔时间    
            interval = config_manager.get('adb.screenshot.interval', 1)
            print(f"从配置文件读取截图间隔: {interval}秒")
            print(f"启动截图任务，间隔: {interval}秒...")
            task_id = screenshot_manager.start_screenshot_task(interval=interval)
            
            if task_id == "ERROR_NO_DEVICE":
                print("截图任务启动失败: 设备未连接")
                return
                
            print(f"截图任务已启动，任务ID: {task_id}")
            print("按Ctrl+C停止...")
            
            try:
                while True:
                    # 定期检查设备连接状态
                    if not screenshot_manager.adb_helper.check_device_connection():
                        print("警告: 设备连接已断开，等待重新连接...")
                    time.sleep(3)
            except KeyboardInterrupt:
                print("\n收到中断信号，正在停止...")
            finally:
                screenshot_manager.stop_all_tasks()
                logger.info("所有截图任务已停止")
                print("任务已停止")
        
        elif choice == "3":
            # 多间隔截图任务
            if not screenshot_manager.adb_helper.check_device_connection():
                print("无法启动截图任务: 没有检测到已连接的设备")
                return
            
            # 从配置文件读取间隔时间列表
            default_intervals = [1, 3, 5]  # 默认值
            intervals = config_manager.get('adb.screenshot.multi_intervals', default_intervals)
            
            if not isinstance(intervals, list) or not intervals:
                intervals = default_intervals
                print(f"配置文件中未找到有效的多间隔配置，使用默认值: {intervals}秒")
            else:
                print(f"从配置文件读取多间隔配置: {intervals}秒")
            
            task_ids = []
            has_error = False
            
            for interval in intervals:
                task_id = screenshot_manager.start_screenshot_task(interval=interval)
                if task_id == "ERROR_NO_DEVICE":
                    has_error = True
                    print(f"间隔为{interval}秒的任务启动失败: 设备未连接")
                else:
                    task_ids.append(task_id)
                    print(f"已启动间隔为{interval}秒的截图任务: {task_id}")
            
            if not task_ids:
                print("所有任务启动失败，请检查设备连接")
                return
                
            if has_error:
                print("部分任务启动失败，只有已连接的设备会执行截图")
                
            print(f"已启动{len(task_ids)}个截图任务，按Ctrl+C停止...")
            
            try:
                while True:
                    # 定期检查设备连接状态
                    if not screenshot_manager.adb_helper.check_device_connection():
                        print("警告: 设备连接已断开，等待重新连接...")
                    time.sleep(3)
            except KeyboardInterrupt:
                print("\n收到中断信号，正在停止...")
            finally:
                screenshot_manager.stop_all_tasks()
                logger.info("所有截图任务已停止")
                print("所有任务已停止")
                
        elif choice == "4":
            # 检查设备连接
            print("开始检查设备连接状态...")
            if screenshot_manager.adb_helper.check_device_connection():
                print("设备已连接，可以开始截图任务")
            else:
                print("设备未连接，请连接设备后重试")
                print("提示: 请确保USB调试已开启，并且已授权此电脑连接")
                
        elif choice == "5":
            # 启动ADB服务
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
                
        elif choice == "6":
            # 打印当前配置
            print("\n当前配置:")
            print(f"ADB路径: {config_manager.get('adb.path')}")
            temp_dir = config_manager.get('environment.temp_dir', '/Users/mac/ai/temp')
            screenshot_dir = os.path.join(temp_dir, 'screenshots')
            print(f"临时文件目录: {temp_dir}")
            print(f"截图保存目录: {screenshot_dir}")
            print(f"单任务截图间隔: {config_manager.get('adb.screenshot.interval')}秒")
            print(f"多任务截图间隔: {config_manager.get('adb.screenshot.multi_intervals')}秒")
            print(f"清空目标目录: {config_manager.get('adb.screenshot.clear_target_dir')}")
            print(f"日期格式: {config_manager.get('adb.screenshot.date_format')}")
                
        else:
            print("无效的选择，请输入1-6之间的数字")
            
    except Exception as e:
        logger.error(f"执行任务异常: {str(e)}")
        print(f"执行任务异常: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n程序已中断")
        
    finally:
        print("程序已退出")

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
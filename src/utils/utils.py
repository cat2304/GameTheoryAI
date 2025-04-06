"""
工具模块

提供系统工具、ADB工具和游戏OCR功能。
"""

import os
import cv2
import yaml
import numpy as np
import logging
import subprocess
import time
import threading
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件格式错误: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项键名，支持点号分隔的多级键名
            default: 默认值，当配置项不存在时返回
            
        Returns:
            Any: 配置项值
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项
        
        Args:
            key: 配置项键名，支持点号分隔的多级键名
            value: 配置项值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        # 保存配置
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)

class LogManager:
    """日志管理器类"""
    
    def __init__(self, config: ConfigManager):
        """初始化日志管理器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        # 获取日志配置
        log_level = self.config.get('logging.level', 'INFO')
        log_format = self.config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 设置根日志记录器
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            logging.Logger: 日志记录器实例
        """
        logger = logging.getLogger(name)
        
        # 如果启用了文件日志，添加文件处理器
        if self.config.get('logging.file.enabled', True):
            log_dir = self.config.get('logging.file.dir', 'data/logs')
            log_name = self.config.get('logging.file.name_format', '{name}_%Y%m%d.log').format(name=name)
            
            # 创建日志目录
            os.makedirs(log_dir, exist_ok=True)
            
            # 创建文件处理器
            file_handler = logging.FileHandler(
                os.path.join(log_dir, log_name),
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(self.config.get('logging.format')))
            logger.addHandler(file_handler)
        
        return logger

class ADBHelper:
    """ADB工具助手类
    
    提供Android设备操作相关功能，包括截图等操作。
    """
    
    def __init__(self, adb_path: Optional[str] = None):
        """初始化ADB助手
        
        Args:
            adb_path: ADB工具路径，如果为None则使用配置文件中的路径
        """
        self.adb_path = adb_path or config_manager.get('adb.path')
        self.logger = log_manager.get_logger(__name__)
        self._screenshot_counter = 0  # 截图计数器，用于生成递增的文件名
        self.use_mock = not self._is_adb_available()
        
        if self.use_mock:
            self.logger.warning("ADB工具不可用，将使用模拟模式")
            # 检查模拟脚本是否存在
            if not os.path.exists('mock_adb.py'):
                self.logger.error("模拟脚本不存在: mock_adb.py")
        else:
            self._verify_adb()
    
    def _is_adb_available(self) -> bool:
        """检查ADB是否可用"""
        try:
            result = subprocess.run(['which', self.adb_path], 
                                  capture_output=True, 
                                  text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _verify_adb(self) -> None:
        """验证ADB工具是否可用"""
        try:
            subprocess.run([self.adb_path, "version"], check=True, capture_output=True)
            self.logger.info("ADB工具验证成功")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ADB工具验证失败: {e}")
            raise
        except FileNotFoundError as e:
            self.logger.error(f"ADB工具路径错误: {e}")
            raise

    def take_screenshot(self, local_dir: Optional[str] = None) -> str:
        """执行截图并保存到本地
        
        Args:
            local_dir: 本地保存目录，如果为None则使用配置文件中的路径
            
        Returns:
            str: 保存的截图文件路径
        """
        try:
            # 如果启用模拟模式
            if self.use_mock:
                return self._take_mock_screenshot(local_dir)
            
            # 准备基础路径
            base_dir = local_dir or config_manager.get('adb.screenshot.local_temp_dir')
            remote_path = config_manager.get('adb.screenshot.remote_path')
            
            # 创建日期目录
            date_str = datetime.now().strftime(config_manager.get('adb.screenshot.date_format'))
            date_dir = Path(base_dir) / date_str
            os.makedirs(date_dir, exist_ok=True)
            
            # 生成文件名（递增序号）
            self._screenshot_counter += 1
            filename = f"{self._screenshot_counter}.png"
            local_path = date_dir / filename
            
            # 执行截图
            self._execute_screenshot(remote_path)
            
            # 拉取文件
            self._pull_screenshot(remote_path, local_path)
            
            self.logger.info(f"截图已保存到: {local_path}")
            return str(local_path)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"执行ADB命令失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"截图过程中发生错误: {e}")
            raise
    
    def _take_mock_screenshot(self, local_dir: Optional[str] = None) -> str:
        """执行模拟截图
        
        Args:
            local_dir: 本地保存目录，如果为None则使用配置文件中的路径
            
        Returns:
            str: 保存的截图文件路径
        """
        try:
            # 准备基础路径
            base_dir = local_dir or config_manager.get('adb.screenshot.local_temp_dir')
            
            # 创建日期目录
            date_str = datetime.now().strftime(config_manager.get('adb.screenshot.date_format', '%Y%m%d'))
            date_dir = Path(base_dir) / date_str
            os.makedirs(date_dir, exist_ok=True)
            
            # 生成文件名（递增序号）
            self._screenshot_counter += 1
            filename = f"{self._screenshot_counter}.png"
            save_path = date_dir / filename
            
            # 创建一个空白图片
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # 添加一些文字和图形，模拟游戏界面
            cv2.putText(img, 'Mock Screenshot', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Time: {datetime.now().strftime("%H:%M:%S")}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.rectangle(img, (100, 200), (400, 400), (0, 255, 0), 2)
            cv2.circle(img, (800, 300), 50, (0, 0, 255), -1)
            
            # 保存图片
            cv2.imwrite(str(save_path), img)
            
            self.logger.info(f"生成模拟截图: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"生成模拟截图失败: {str(e)}")
            return ""

    def _execute_screenshot(self, remote_path: str) -> None:
        """执行设备截图命令"""
        try:
            self.logger.info("正在截图...")
            # 首先检查设备是否在线
            result = subprocess.run(
                [self.adb_path, "shell", "getprop", "sys.boot_completed"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    result.stdout,
                    result.stderr
                )
            
            # 检查设备是否完全启动
            if "1" not in result.stdout.strip():
                self.logger.warning("设备未完全启动，等待中...")
                time.sleep(5)  # 等待设备完全启动
                return self._execute_screenshot(remote_path)
            
            # 执行截图命令
            result = subprocess.run(
                [self.adb_path, "shell", "screencap", "-p", remote_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"截图命令失败: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    result.stdout,
                    result.stderr
                )
            
            time.sleep(config_manager.get('adb.screenshot.interval', 1))  # 等待截图完成
        except subprocess.CalledProcessError as e:
            self.logger.error(f"执行ADB命令失败: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"截图过程中发生错误: {str(e)}")
            raise

    def _pull_screenshot(self, remote_path: str, local_path: Path) -> None:
        """从设备拉取截图文件"""
        self.logger.info("正在拉取截图...")
        subprocess.run(
            [self.adb_path, "pull", remote_path, str(local_path)],
            check=True,
            capture_output=True
        )

    def list_devices(self) -> List[str]:
        """获取已连接的设备列表
        
        Returns:
            List[str]: 设备序列号列表
        """
        try:
            result = subprocess.run([self.adb_path, 'devices'], capture_output=True, text=True)
            if result.returncode == 0:
                # 解析设备列表，跳过第一行（标题行）
                devices = []
                for line in result.stdout.splitlines()[1:]:
                    if line.strip():
                        device_id = line.split('\t')[0]
                        if device_id:
                            devices.append(device_id)
                return devices
            else:
                self.logger.error(f"获取设备列表失败: {result.stderr}")
                return []
        except Exception as e:
            self.logger.error(f"获取设备列表异常: {str(e)}")
            return []

class GameOCR:
    """游戏OCR工具类"""
    
    def __init__(self, config_path: Union[str, Path]):
        """初始化游戏状态识别工具"""
        self.logger = log_manager.get_logger(__name__)
        
        if isinstance(config_path, str):
            config_path = Path(config_path)
            
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        self.config = {
            # 预处理参数
            'preprocessing': {
                'blur_kernel': (3, 3),     # 高斯模糊核大小
                'threshold_min': 200,       # 二值化最小阈值
                'threshold_max': 255,       # 二值化最大阈值
                'canny_min': 30,           # 边缘检测最小阈值
                'canny_max': 120,          # 边缘检测最大阈值
                'dilate_kernel': (2, 2),   # 膨胀核大小
                'dilate_iterations': 1,     # 膨胀次数
                'debug_output': True       # 是否输出调试图片
            },
            # 游戏元素检测参数
            'detection': {
                'min_area': 3000,          # 最小面积
                'max_area': 50000,         # 最大面积
                'aspect_ratio_min': 0.6,   # 最小宽高比
                'aspect_ratio_max': 1.5,   # 最大宽高比
                'min_width': 30,           # 最小宽度
                'max_width': 300,          # 最大宽度
                'y_threshold': 150,        # 底部区域阈值
                'expected_elements': 13,    # 期望的游戏元素数量
                'element_gap': 5,          # 游戏元素间隔阈值
                'merge_threshold': 20,      # 合并相近区域的阈值
                'max_retries': 3,          # 最大重试次数
                'threshold_reduction': 0.8  # 每次重试的阈值降低比例
            },
            # 文字识别参数
            'recognition': {
                'text_area_ratio': 0.4,    # 文字区域占比阈值
                'text_threshold': 150,      # 二值化阈值
                'red_threshold': 0.02,      # 红色像素占比阈值
                'green_threshold': 0.02,    # 绿色像素占比阈值
                'circle_threshold': 0.7,    # 圆形度阈值
                'debug_features': True     # 是否输出特征调试信息
            }
        }
        
        self.logger.info(f"游戏状态识别工具初始化完成")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图片
        
        Args:
            image: 输入图片
            
        Returns:
            np.ndarray: 预处理后的图片
        """
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
        """识别图片中的文字
        
        Args:
            image_path: 图片路径
            
        Returns:
            str: 识别到的文字
        """
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
        """识别游戏截图
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 识别结果，包含success和elements字段
        """
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
            
            # 合并相近的区域
            elements = self._merge_nearby_regions(elements)
            
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
    
    def _merge_nearby_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """合并相近的区域"""
        if not regions:
            return []
            
        # 按x坐标排序
        sorted_regions = sorted(regions, key=lambda x: x[0])
        merged = []
        current = list(sorted_regions[0])
        
        for region in sorted_regions[1:]:
            # 如果两个区域足够接近
            if region[0] - (current[0] + current[2]) < self.config['detection']['merge_threshold']:
                # 更新当前区域
                current[2] = region[0] + region[2] - current[0]  # 新宽度
                current[3] = max(current[3], region[3])  # 新高度
            else:
                merged.append(tuple(current))
                current = list(region)
        
        merged.append(tuple(current))
        return merged

class ScreenshotManager:
    """截图管理器，使用线程池进行异步调度"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化截图管理器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger(__name__)
        self.config = config
        adb_path = config.get('adb', {}).get('path')
        self.logger.info(f"使用ADB路径: {adb_path}")
        self.adb_helper = ADBHelper(adb_path)
        self.is_running = False
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.screenshot_interval = config.get('adb', {}).get('screenshot', {}).get('interval', 5.0)
        
    def start(self):
        """启动截图任务"""
        if self.is_running:
            self.logger.warning("截图任务已在运行中")
            return
            
        self.is_running = True
        self._schedule_next_screenshot()
        self.logger.info(f"截图任务已启动，间隔 {self.screenshot_interval} 秒")
        
    def stop(self):
        """停止截图任务"""
        if not self.is_running:
            self.logger.warning("截图任务未在运行")
            return
            
        self.is_running = False
        self.thread_pool.shutdown(wait=False)
        self.logger.info("截图任务已停止")
        
    def _schedule_next_screenshot(self):
        """调度下一次截图"""
        if not self.is_running:
            return
            
        self.thread_pool.submit(self._take_screenshot)
        
    def _take_screenshot(self):
        """执行截图任务"""
        try:
            # 准备基础路径
            base_dir = self.config.get('adb', {}).get('screenshot', {}).get('local_temp_dir', 'data/screenshots')
            
            # 执行截图
            screenshot_path = self.adb_helper.take_screenshot(base_dir)
                
            if screenshot_path:
                self.logger.debug(f"截屏成功: {screenshot_path}")
            else:
                self.logger.warning("截屏失败")
                
        except Exception as e:
            self.logger.error(f"截屏出错: {str(e)}")
        finally:
            # 调度下一次截图
            if self.is_running:
                threading.Timer(self.screenshot_interval, self._schedule_next_screenshot).start()

def run_project(config: Dict[str, Any]) -> None:
    """运行项目主函数
    
    Args:
        config: 项目配置字典
    """
    logger = get_logger(__name__)
    logger.info("开始运行 GameTheoryAI 项目...")
    
    try:
        # 初始化游戏监控
        monitor = GameMonitor(config)
        logger.info("游戏监控初始化完成")
        
        # 初始化AI决策
        ai = AIPlayer(config)
        logger.info("AI决策初始化完成")
        
        # 启动监控
        monitor.start()
        logger.info("游戏监控已启动")
        
        print("\n游戏监控已启动，按 Ctrl+C 停止...")
        
        # 定时截屏和状态更新
        def screenshot_and_update():
            try:
                # 执行截屏
                screenshot_path = monitor.take_screenshot()
                if screenshot_path:
                    logger.debug(f"截屏成功: {screenshot_path}")
                    
                    # 获取游戏状态
                    game_state = monitor.get_state()
                    if game_state:
                        # 更新AI状态
                        ai.update_state(game_state)
                        
                        # 使用AI决策
                        next_move = ai.decide_action()
                        logger.info(f"AI决策结果: {next_move}")
                        
                        # 执行动作
                        if next_move:
                            monitor.execute_action(next_move)
                    else:
                        logger.warning("获取游戏状态失败")
                else:
                    logger.warning("截屏失败")
                    
            except Exception as e:
                logger.error(f"截屏和状态更新出错: {str(e)}")
            finally:
                # 5秒后再次执行
                threading.Timer(5.0, screenshot_and_update).start()
        
        # 启动定时任务
        screenshot_and_update()
        
        # 保持主线程运行
        while True:
            time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("收到终止信号，正在停止...")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise
    finally:
        # 清理资源
        if 'monitor' in locals():
            try:
                monitor.stop()
                logger.info("游戏监控已停止")
            except Exception as e:
                logger.error(f"停止监控时出错: {str(e)}")
        
        if 'ai' in locals():
            try:
                ai.cleanup()
                logger.info("AI资源已清理")
            except Exception as e:
                logger.error(f"清理AI资源时出错: {str(e)}")
        
        logger.info("程序已停止")
        print("\n程序已停止，按回车键返回主菜单...")
        input()

def start_screenshot_task(monitor: 'GameMonitor', config: Dict[str, Any]) -> ScreenshotManager:
    """启动截图任务
    
    Args:
        monitor: 游戏监控器实例
        config: 配置字典
        
    Returns:
        ScreenshotManager: 截图管理器实例
    """
    screenshot_manager = ScreenshotManager(config)
    screenshot_manager.start()
    return screenshot_manager

def test_screenshot(adb_path: Optional[str] = None, screenshot_dir: Optional[str] = None) -> None:
    """测试ADB截图功能
    
    Args:
        adb_path: ADB工具路径，如果为None则使用配置文件中的路径
        screenshot_dir: 本地保存目录，如果为None则使用配置文件中的路径
    """
    logger = get_logger(__name__)
    
    try:
        # 加载配置
        config = config_manager
        logger.info("配置加载成功")
        
        # 获取ADB路径
        if adb_path is None:
            adb_path = config.get('adb.path')
        logger.info(f"ADB路径: {adb_path}")
        
        # 获取截图目录
        if screenshot_dir is None:
            screenshot_dir = config.get('adb.screenshot.local_temp_dir')
        logger.info(f"截图目录: {screenshot_dir}")
        
        # 创建ADB助手
        adb_helper = ADBHelper(adb_path)
        logger.info(f"ADB助手初始化成功，路径: {adb_helper.adb_path}")
        
        # 确保截图目录存在
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # 尝试截图
        logger.info("开始执行截图...")
        screenshot_path = adb_helper.take_screenshot(screenshot_dir)
        
        if screenshot_path:
            logger.info(f"截图成功，保存到: {screenshot_path}")
            print(f"\n截图已保存到: {screenshot_path}")
            return screenshot_path
        else:
            logger.error("截图失败")
            print("\n截图失败，请检查ADB连接和配置")
            return None
            
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        print(f"\n发生错误: {str(e)}")
        return None

def test_timer_screenshot(duration: int = 30) -> None:
    """测试定时截图功能
    
    Args:
        duration: 测试持续时间（秒）
    """
    import signal
    
    # 初始化变量
    running = True
    
    # 处理 Ctrl+C 信号
    def signal_handler(sig, frame):
        """处理 Ctrl+C 信号"""
        print("\n收到停止信号，正在停止...")
        nonlocal running
        running = False
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化日志
    logger = get_logger(__name__)
    
    try:
        # 加载配置
        config_dict = config_manager.config
        logger.info("配置加载成功")
        
        # 获取配置
        adb_path = config_manager.get('adb.path')
        logger.info(f"ADB路径: {adb_path}")
        
        screenshot_dir = config_manager.get('adb.screenshot.local_temp_dir')
        logger.info(f"截图保存目录: {screenshot_dir}")
        
        # 设置截图间隔为 5 秒
        config_dict['adb']['screenshot']['interval'] = 5.0
        logger.info(f"设置截图间隔: 5秒")
        
        # 确保截图目录存在
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # 初始化游戏监控器
        try:
            from src.core.game.game_monitor import GameMonitor
            monitor = GameMonitor(config_dict)
            logger.info("游戏监控器初始化成功")
        except Exception as e:
            # 如果无法初始化 GameMonitor，创建一个模拟的监控器
            logger.warning(f"无法初始化游戏监控器: {str(e)}")
            
            class MockMonitor:
                def take_screenshot(self, local_dir=None):
                    nonlocal adb_path, screenshot_dir
                    if local_dir is None:
                        local_dir = screenshot_dir
                    adb_helper = ADBHelper(adb_path)
                    return adb_helper.take_screenshot(local_dir)
            
            monitor = MockMonitor()
            logger.info("使用模拟监控器")
        
        # 启动定时截图任务
        logger.info("开始启动定时截图任务...")
        screenshot_manager = start_screenshot_task(monitor, config_dict)
        logger.info("定时截图任务已启动")
        
        # 运行指定时间后停止
        logger.info(f"定时截图任务将运行{duration}秒后停止...")
        print(f"\n定时截图任务已启动，每5秒执行一次。按 Ctrl+C 可提前停止...")
        
        start_time = time.time()
        
        while running and (time.time() - start_time) < duration:
            # 每秒检查一次是否需要退出
            time.sleep(1)
            
            # 显示运行时间
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            if elapsed % 5 == 0 and elapsed > 0:
                print(f"已运行 {elapsed} 秒，还剩 {remaining} 秒...")
            
        # 停止定时任务
        logger.info("准备停止定时截图任务...")
        screenshot_manager.stop()
        logger.info("定时截图任务已停止")
        
        print("\n定时截图任务已完成。")
        return True
        
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        print(f"\n发生错误: {str(e)}")
        return False

# 创建全局配置管理器和日志管理器实例
config_manager = ConfigManager()
log_manager = LogManager(config_manager)

# 导出常用函数
get_config = config_manager.get
set_config = config_manager.set
get_logger = log_manager.get_logger

if __name__ == "__main__":
    """命令行接口，用于测试
    
    用法：
        python -m src.utils.utils [options]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="GameTheoryAI 工具测试")
    parser.add_argument("--adb-path", help="ADB工具路径")
    parser.add_argument("--screenshot-dir", help="截图保存目录")
    parser.add_argument("--timer", action="store_true", help="测试定时截图功能")
    parser.add_argument("--duration", type=int, default=30, help="定时测试持续时间（秒）")
    args = parser.parse_args()
    
    if args.timer:
        test_timer_screenshot(args.duration)
    else:
        test_screenshot(args.adb_path, args.screenshot_dir) 
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
        self._verify_adb()

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
            # 准备基础路径
            base_dir = local_dir or config_manager.get('adb.screenshot.local_temp_dir')
            remote_path = config_manager.get('adb.screenshot.remote_path')
            
            # 创建日期目录
            date_str = datetime.now().strftime(config_manager.get('adb.screenshot.date_format'))
            date_dir = Path(base_dir) / date_str
            os.makedirs(date_dir, exist_ok=True)
            
            # 生成文件名
            filename = f"screenshot_{int(time.time())}.png"
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
    """游戏状态识别类
    
    提供游戏状态识别功能，包括：
    - 游戏区域检测
    - 游戏状态识别
    - OCR文字识别
    """
    
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
                'merge_threshold': 20      # 合并相近区域的阈值
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
    
    def find_game_elements(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """查找游戏元素区域"""
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
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < config['min_area'] or area > config['max_area']:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                y = height - y_threshold + y  # 调整y坐标
                
                aspect_ratio = w / float(h)
                if (aspect_ratio < config['aspect_ratio_min'] or 
                    aspect_ratio > config['aspect_ratio_max']):
                    continue
                
                if w < config['min_width'] or w > config['max_width']:
                    continue
                
                elements.append((x, y, w, h))
            
            # 合并相近的区域
            elements = self._merge_nearby_regions(elements)
            
            # 检查识别到的元素数是否合理
            if len(elements) < 10:  # 最少应该有10个元素
                self.logger.warning(f"识别到的游戏元素数量过少: {len(elements)}")
                # 尝试调整参数重新识别
                if len(elements) < config['expected_elements']:
                    self.logger.info("尝试调整参数重新识别...")
                    # 临时降低阈值
                    config['min_area'] *= 0.8
                    config['min_width'] *= 0.8
                    return self.find_game_elements(image)
            
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

# 创建全局配置管理器和日志管理器实例
config_manager = ConfigManager()
log_manager = LogManager(config_manager)

# 导出常用函数
get_config = config_manager.get
set_config = config_manager.set
get_logger = log_manager.get_logger 
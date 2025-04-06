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
    """ADB工具助手"""
    
    def __init__(self, adb_path: Optional[str] = None):
        self.adb_path = adb_path or config_manager.get('adb.path')
        self.logger = log_manager.get_logger(__name__)
        self._screenshot_counter = 0  # 截图计数器，用于生成递增的文件名
        self._verify_adb()
        # 初始化ADB设备会话
        self._init_device_session()
    
    def _is_adb_available(self) -> bool:
        """检查ADB是否可用"""
        try:
            result = subprocess.run(['which', self.adb_path], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _verify_adb(self) -> None:
        """验证ADB工具是否可用"""
        try:
            subprocess.run([self.adb_path, "version"], check=True, capture_output=True)
            self.logger.info("ADB工具验证成功")
        except Exception as e:
            self.logger.error(f"ADB工具验证失败: {e}")
            raise
    
    def _init_device_session(self) -> None:
        """初始化与设备的会话，提前检查设备连接状态"""
        try:
            # 检查设备连接状态
            result = subprocess.run(
                [self.adb_path, "devices"], 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # 验证是否有设备连接
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:
                self.logger.warning("没有检测到已连接的设备")
            else:
                self.logger.info(f"检测到 {len(lines)-1} 个已连接设备")
                
            # 预热设备通信
            subprocess.run(
                [self.adb_path, "shell", "echo", "ready"],
                capture_output=True,
                check=False
            )
        except Exception as e:
            self.logger.warning(f"设备会话初始化失败: {e}")

    def take_screenshot(self, local_dir: Optional[str] = None) -> str:
        """执行截图并保存到本地"""
        try:
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
            
            # 使用内存直接压缩的方式截图 (不依赖pngquant)
            compressed_path = self._direct_memory_compressed_screenshot(local_path)
            if compressed_path:
                self.logger.info(f"截图已直接压缩并保存到: {compressed_path}")
                return str(compressed_path)
            
            # 如果直接压缩失败，回退到普通截图
            self.logger.debug("直接压缩失败，使用普通截图方式")
            self._execute_optimized_screenshot(remote_path, local_path)
            
            # 压缩图片以减小文件大小
            compressed_path = self._compress_image(local_path)
            
            self.logger.info(f"截图已保存到: {compressed_path}")
            return str(compressed_path)
            
        except Exception as e:
            self.logger.error(f"截图过程中发生错误: {e}")
            raise
    
    def _direct_memory_compressed_screenshot(self, output_path: Path) -> Optional[Path]:
        """通过内存操作直接获取并压缩截图，无需中间文件"""
        try:
            self.logger.debug("获取截图并直接在内存中压缩...")
            
            # 获取截图数据到内存
            process = subprocess.Popen(
                [self.adb_path, "exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout_data, stderr_data = process.communicate(timeout=5)
            
            if process.returncode != 0:
                stderr_output = stderr_data.decode('utf-8', errors='ignore')
                self.logger.warning(f"获取截图数据失败: {stderr_output}")
                return None
            
            # 使用PIL处理图像
            with io.BytesIO(stdout_data) as data:
                # 读取原始图像
                image = Image.open(data)
                original_size = len(stdout_data)
                
                # 准备输出路径
                jpg_path = output_path.with_suffix('.jpg')
                
                # 检查是否为RGBA模式
                if image.mode == 'RGBA':
                    # 转换RGBA到RGB（用白色背景）
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])  # 3是alpha通道
                    image = background
                
                # 使用PIL的内置压缩功能直接保存为高质量JPEG
                image.save(jpg_path, format='JPEG', quality=85, optimize=True)
                
                # 检查压缩后大小
                compressed_size = os.path.getsize(jpg_path)
                compression_ratio = compressed_size / original_size
                
                self.logger.info(f"截图内存直接压缩: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB ({compression_ratio:.1%})")
                return jpg_path
                
        except Exception as e:
            self.logger.warning(f"内存直接压缩截图失败: {e}")
            return None

    def _execute_optimized_screenshot(self, remote_path: str, local_path: Path) -> None:
        """优化的截图执行函数，合并命令减少延迟"""
        try:
            # 直接使用管道处理，避免中间文件读写
            self.logger.debug("正在截图...")  # 降低日志级别，减少输出
            
            # 方法1: 直接从设备捕获截图并保存到本地
            # 使用超时控制，避免长时间等待
            process = subprocess.Popen(
                [self.adb_path, "exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024*1024  # 使用更大的缓冲区
            )
            
            # 设置超时时间为5秒（增加超时时间避免超时错误）
            try:
                stdout_data, stderr_data = process.communicate(timeout=5)
                
                # 直接将数据写入文件
                with open(local_path, 'wb') as f:
                    f.write(stdout_data)
                
                # 检查命令是否成功执行
                if process.returncode != 0:
                    stderr_output = stderr_data.decode('utf-8', errors='ignore')
                    self.logger.error(f"截图命令失败: {stderr_output}")
                    raise subprocess.CalledProcessError(process.returncode, "adb exec-out screencap", stderr_output)
                    
            except subprocess.TimeoutExpired:
                # 超时后终止进程
                process.kill()
                self.logger.warning("截图命令超时，尝试备选方法")
                raise TimeoutError("截图命令执行超时")
                
        except Exception as e:
            self.logger.warning(f"优化截图方法失败: {str(e)}")
            # 失败时回退到传统方法
            self._fallback_screenshot(remote_path, local_path)

    def _fallback_screenshot(self, remote_path: str, local_path: Path) -> None:
        """传统的截图方法，作为备选方案"""
        try:
            self.logger.warning("优化截图方法失败，使用传统方法...")
            # 执行截图命令
            subprocess.run(
                [self.adb_path, "shell", "screencap", "-p", remote_path],
                check=True,
                capture_output=True
            )
            
            # 拉取截图文件
            self.logger.info("正在拉取截图...")
            subprocess.run(
                [self.adb_path, "pull", remote_path, str(local_path)],
                check=True,
                capture_output=True
            )
        except Exception as e:
            self.logger.error(f"备选截图方法也失败: {str(e)}")
            raise

    def _execute_screenshot(self, remote_path: str) -> None:
        """执行设备截图命令 (旧方法，保留兼容性)"""
        try:
            self.logger.info("正在截图...")
            # 检查设备是否在线
            result = subprocess.run(
                [self.adb_path, "shell", "getprop", "sys.boot_completed"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, result.stdout, result.stderr
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
                    result.returncode, result.args, result.stdout, result.stderr
                )
        except Exception as e:
            self.logger.error(f"执行ADB命令失败: {str(e)}")
            raise

    def _pull_screenshot(self, remote_path: str, local_path: Path) -> None:
        """从设备拉取截图文件 (旧方法，保留兼容性)"""
        self.logger.info("正在拉取截图...")
        subprocess.run(
            [self.adb_path, "pull", remote_path, str(local_path)],
            check=True,
            capture_output=True
        )

    def _compress_image(self, image_path: Path) -> Path:
        """压缩图片以减小文件大小，同时保持图像质量"""
        try:
            # 检查原图大小
            original_size = os.path.getsize(image_path)
            
            # 打开图片
            img = Image.open(image_path)
            
            # 如果是RGBA模式，转换为RGB模式
            if img.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 将原图合并到背景上
                background.paste(img, mask=img.split()[3])  # 3是alpha通道
                img = background
            
            # 先将原图备份
            original_path = image_path.with_suffix('.original.png')
            os.rename(image_path, original_path)
            
            # 压缩为高质量JPEG (比PNG小很多但保持较好质量)
            compressed_path = image_path.with_suffix('.jpg')
            img.save(compressed_path, format='JPEG', quality=85, optimize=True)
            
            # 获取压缩后的大小
            compressed_size = os.path.getsize(compressed_path)
            
            # 计算压缩比
            compression_ratio = compressed_size / original_size
            
            # 如果新文件显著小于原文件，则使用压缩版本
            if compression_ratio <= 0.7:  # 节省30%以上空间才使用压缩版
                self.logger.info(f"图片已压缩: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB ({compression_ratio:.1%})")
                os.remove(original_path)  # 删除原始备份
                return compressed_path
            else:
                # 压缩效果不明显，恢复原始文件
                os.remove(compressed_path)
                os.rename(original_path, image_path)
                self.logger.debug(f"压缩效果不明显 ({compression_ratio:.1%})，保留原始图片")
                return image_path
                
        except Exception as e:
            self.logger.warning(f"压缩图片失败: {e}")
            # 如果有原始备份文件，恢复它
            original_backup = image_path.with_suffix('.original.png')
            if original_backup.exists():
                os.rename(original_backup, image_path)
            return image_path  # 如果压缩失败，返回原始路径

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
            
            # 执行截图
            path = self.adb_helper.take_screenshot(local_dir=save_dir)
            self.logger.info(f"截图已保存: {path}")
            return path
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
            return None
    
    def start_screenshot_task(self, interval: int = 5, max_count: Optional[int] = None, 
                             save_dir: Optional[str] = None) -> str:
        """启动定时截图任务"""
        task_id = str(uuid.uuid4())
        stop_event = threading.Event()
        
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
                        # 截图失败时短暂等待
                        time.sleep(0.5)
                        
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
    
    # 打印主菜单
    print("\n==== 游戏辅助工具 ====")
    print("1. 执行单次截图")
    print("2. 启动定时截图任务 (按Ctrl+C停止)")
    print("====================\n")
    
    try:
        choice = input("请选择功能: ")
        
        # 初始化ADB助手
        adb_helper = ADBHelper(config["adb"]["path"])
        
        if choice == "1":
            # 单次截图
            screenshot_path = adb_helper.take_screenshot()
            if screenshot_path:
                print(f"截图已保存: {screenshot_path}")
            else:
                print("截图失败")
                
        elif choice == "2":
            # 定时截图
            screenshot_manager = ScreenshotManager(config_manager)
            try:
                # 从配置读取间隔时间
                interval = config.get("adb", {}).get("screenshot", {}).get("interval", 1)
                task_id = screenshot_manager.start_screenshot_task(interval=interval)
                print(f"截图任务已启动，间隔: {interval}秒，按Ctrl+C停止...")
                
                # 等待用户中断
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n收到中断信号，正在停止...")
                screenshot_manager.stop_all_tasks()
                print("任务已停止")
        else:
            print("无效选择")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        print(f"程序执行出错: {str(e)}")
    except KeyboardInterrupt:
        print("\n程序已退出") 
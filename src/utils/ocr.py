"""
OCR工具模块

提供游戏OCR功能，支持图像预处理、元素检测和文字识别。
"""

import os
import cv2
import numpy as np
import logging
import pytesseract
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 日志记录
def get_logger(name: str) -> logging.Logger:
    """获取预配置的日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # 添加文件处理器
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 设置级别
        logger.setLevel(logging.DEBUG)
    
    return logger

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

def find_latest_screenshot(base_dir: str) -> Optional[str]:
    """查找最新截图文件"""
    logger = get_logger("ocr.utils")
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

def handle_ocr_test(game_ocr, config_manager):
    """单张OCR图片识别功能"""
    logger = get_logger(__name__)
    logger.info("启动OCR识别流程")

    try:
        # 从配置获取截图目录
        temp_dir = config_manager.get('environment.temp_dir')
        screenshot_dir = os.path.join(temp_dir, 'screenshots')
        
        # 自动查找最新截图
        latest_img = find_latest_screenshot(screenshot_dir)
        
        if not latest_img:
            print("\n警告：未找到任何截图文件")
            print("请先执行截图操作或手动指定图片路径")
            return

        # 执行识别
        print(f"\n正在自动识别最新截图: {latest_img}")
        start_time = datetime.now().timestamp()
        
        result = game_ocr.recognize_image(latest_img)
        
        elapsed_time = datetime.now().timestamp() - start_time
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
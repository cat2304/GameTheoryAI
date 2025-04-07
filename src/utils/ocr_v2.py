"""
OCR工具模块

提供游戏OCR功能，支持图像预处理、元素检测和文字识别。
"""

import os
import sys
import cv2
import numpy as np
import logging
import pytesseract
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging.handlers
import glob
import time
from sklearn.cluster import KMeans
import re
import traceback

# 导入配置管理器
from src.utils.utils import ConfigManager

# 定义常量
IMAGE_WIDTH = 1600   # 默认图像宽度
IMAGE_HEIGHT = 900   # 默认图像高度

# 日志记录
def get_logger(name: str) -> logging.Logger:
    """获取预配置的日志记录器"""
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.hasHandlers():
        return logger
        
    try:
        # 获取配置
        config = ConfigManager("config/app_config.yaml").get_config()
        
        # 创建logs目录
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # 添加控制台处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # 添加文件处理器
        log_file = os.path.join(logs_dir, f"{name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 设置日志级别
        if "log_level" in config:
            log_level = getattr(logging, config["log_level"].upper(), logging.DEBUG)
        else:
            log_level = logging.DEBUG
            logger.warning("配置文件中缺少log_level，使用默认值DEBUG")
        
        logger.setLevel(log_level)
        
        # 记录初始化信息
        logger.info(f"日志初始化完成，日志文件: {log_file}")
        
    except Exception as e:
        # 如果配置失败，设置基本的控制台日志
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.error(f"日志配置失败: {str(e)}")
    
    return logger

class OCR:
    """OCR核心类"""
    def __init__(self, config_path: str = "config/app_config.yaml", debug_mode: bool = False):
        self.logger = get_logger("ocr.core")
        
        # 加载并验证配置
        config_manager = ConfigManager(config_path)
        self.config = config_manager.get_config()
        
        # 验证必需的配置项
        required_configs = [
            "screenshots_dir",
            "template_dir",
            "debug_dir",
            "output_dir",
            "log_dir",
            "preprocess_params",
            "detection_params",
            "ocr_params",
            "name_map",
            "number_map"
        ]
        
        missing_configs = []
        for config_name in required_configs:
            if config_name not in self.config:
                missing_configs.append(config_name)
        
        if missing_configs:
            error_msg = f"配置文件错误：缺少以下配置项：{', '.join(missing_configs)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.debug_mode = debug_mode
        self.templates = {}
        
        # 初始化debug目录
        if self.debug_mode:
            self.debug_dir = Path(self.config["debug_dir"])
            self.debug_dir.mkdir(exist_ok=True)
        
        # 获取配置参数
        self.preprocess_params = self.config["preprocess_params"]
        self.detection_params = self.config["detection_params"]
        self.ocr_params = self.config["ocr_params"]
        self.name_map = self.config["name_map"]
        self.number_map = self.config["number_map"]
        
        # 设置图像缩放比例
        self.resize_ratio = 0.5  # 默认缩放比例为0.5
        
        # 加载模板
        self._load_templates()
        
        self.logger.info("OCR初始化完成")

    def recognize_file(self, image_path: str) -> Dict:
        """识别图片文件"""
        try:
            self.logger.info(f"开始读取图片文件: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            self.logger.info("图片读取成功，开始识别")
            result = self.recognize_image(image_path)
            self.logger.info("图片识别完成")
            return result
        except Exception as e:
            self.logger.error(f"识别图片文件失败: {e}", exc_info=True)
            return {"error": str(e)}

    def recognize_image(self, image_path: str) -> List[Dict]:
        """识别图片中的所有元素"""
        try:
            # 读取图片
            self.logger.info(f"开始读取图片文件: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图片: {image_path}")
                return []
            
            # 保存图片尺寸
            self.image_height, self.image_width = image.shape[:2]
            
            self.logger.info("图片读取成功，开始识别")
            
            # 预处理图像
            processed = self._preprocess_image(image)
            if processed is None:
                self.logger.error("图像预处理失败")
                return []
            
            # 检测元素
            elements = self._detect_elements(processed)
            if not elements:
                self.logger.warning("未检测到任何元素")
                return []
            
            self.logger.info(f"检测到 {len(elements)} 个元素，开始识别")
            
            # 对每个检测到的元素进行识别
            results = []
            for i, element in enumerate(elements):
                x, y, w, h = element
                roi = image[y:y+h, x:x+w]
                
                # 提取ROI的特征
                roi_kp, roi_des = self._extract_features(roi)
                if roi_kp is None or roi_des is None:
                    self.logger.warning(f"无法提取元素 {i+1} 的特征")
                    continue
                
                self.logger.debug(f"元素 {i+1} 提取到 {len(roi_kp)} 个特征点")
                
                # 对每个模板进行匹配
                max_score = 0
                best_match = None
                match_scores = []
                
                for template_name, template_data in self.templates.items():
                    template = template_data['image']
                    template_kp = template_data['keypoints']
                    template_des = template_data['descriptors']
                    
                    # 计算匹配得分
                    score = self._match_template(roi_kp, roi_des, template_kp, template_des)
                    match_scores.append((template_name, score))
                    
                    if score > max_score:
                        max_score = score
                        best_match = template_name
                
                # 排序并记录前3个最佳匹配
                match_scores.sort(key=lambda x: x[1], reverse=True)
                top_matches = match_scores[:3]
                self.logger.debug(f"元素 {i+1} 的前3个匹配: " + 
                                ", ".join([f"{name}({score:.2f})" for name, score in top_matches]))
                
                # 如果找到匹配，降低阈值以提高识别率
                if max_score > 25:  # 进一步降低阈值
                    result = {
                        'name': best_match,
                        'score': max_score,
                        'box': [x, y, x+w, y+h]
                    }
                    results.append(result)
                    self.logger.info(f"识别到元素 {i+1}: {best_match}, 得分: {max_score:.2f}")
                else:
                    self.logger.warning(f"元素 {i+1} 未找到匹配的模板 (最高得分: {max_score:.2f})")
            
            if not results:
                self.logger.warning("未识别到任何元素")
            else:
                self.logger.info(f"共识别到 {len(results)} 个元素")
            
            return results
            
        except Exception as e:
            self.logger.error(f"图片识别失败: {str(e)}")
            traceback.print_exc()
            return []

    def _save_debug_image(self, image, filename):
        """Save debug image to output/debug directory"""
        cv2.imwrite(f'output/debug/{filename}', image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Save original grayscale
            self._save_debug_image(gray, '1_gray.png')
            
            # Resize image
            height, width = gray.shape
            scale = 800 / height
            resized = cv2.resize(gray, None, fx=scale, fy=scale)
            self._save_debug_image(resized, '2_resized.png')
            
            # Split into regions
            height, width = resized.shape
            bottom_y = int(height * 0.7)  # 调整底部区域比例
            
            # Process top region
            top = resized[:bottom_y, :]
            top_enhanced = cv2.createCLAHE(
                clipLimit=2.0,  # 降低对比度限制
                tileGridSize=(8,8)  # 增大网格大小
            ).apply(top)
            top_blurred = cv2.GaussianBlur(top_enhanced, (5,5), 0)  # 增大模糊核
            _, top_binary = cv2.threshold(top_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Process bottom region with different parameters
            bottom = resized[bottom_y:, :]
            # Apply stronger contrast enhancement
            bottom_enhanced = cv2.createCLAHE(
                clipLimit=4.0,  # 降低对比度限制
                tileGridSize=(4,4)  # 增大网格大小
            ).apply(bottom)
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            bottom_sharpened = cv2.filter2D(bottom_enhanced, -1, kernel)
            # Apply bilateral filter to preserve edges
            bottom_filtered = cv2.bilateralFilter(bottom_sharpened, 9, 75, 75)
            # Apply adaptive thresholding
            bottom_binary = cv2.adaptiveThreshold(
                bottom_filtered,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                15,  # 增大块大小
                3    # 增大常数
            )
            # Apply morphological operations
            kernel = np.ones((3,3), np.uint8)  # 增大核大小
            bottom_binary = cv2.morphologyEx(bottom_binary, cv2.MORPH_CLOSE, kernel)
            
            # Save debug images
            self._save_debug_image(top_enhanced, '3_top_enhanced.png')
            self._save_debug_image(bottom_enhanced, '3_bottom_enhanced.png')
            self._save_debug_image(top_binary, '4_top_binary.png')
            self._save_debug_image(bottom_binary, '4_bottom_binary.png')
            
            # Combine binary images
            binary = np.vstack((top_binary, bottom_binary))
            
            # Save final binary image
            self._save_debug_image(binary, '5_binary.png')
            
            return binary
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            return image

    def _detect_elements(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测图像中的元素"""
        try:
            height, width = image.shape
            
            # Find contours - handle different OpenCV versions
            contours_output = cv2.findContours(
                image,
                cv2.RETR_TREE,  # 使用TREE模式获取更多轮廓
                cv2.CHAIN_APPROX_SIMPLE
            )
            # OpenCV 3.x returns (image, contours, hierarchy)
            # OpenCV 4.x returns (contours, hierarchy)
            contours = contours_output[-2]  # Get contours regardless of version
            
            if not contours:
                self.logger.warning("未找到任何轮廓")
                return []
            
            # Calculate reference area
            ref_area = (width * height) / 400  # Expected size for a card
            min_area = ref_area * 0.2  # 降低最小面积限制
            max_area = ref_area * 2.5  # 增大最大面积限制
            
            # Calculate reference aspect ratio
            ref_ratio = 1.4  # Expected aspect ratio for a card
            min_ratio = ref_ratio * 0.6  # 降低最小宽高比限制
            max_ratio = ref_ratio * 1.4  # 增大最大宽高比限制
            
            # Define regions
            bottom_y = int(height * 0.7)  # 调整底部区域比例
            top_y = int(height * 0.15)    # 调整顶部区域比例
            left_x = int(width * 0.15)    # 调整左侧区域比例
            right_x = int(width * 0.85)   # 调整右侧区域比例
            
            # Filter contours
            elements = []
            filtered_stats = {
                'region': 0,
                'area': 0,
                'ratio': 0,
                'overlap': 0
            }
            
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Skip if area is too small or too large
                if area < min_area or area > max_area:
                    filtered_stats['area'] += 1
                    continue
                    
                # Skip if aspect ratio is too extreme
                if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                    filtered_stats['ratio'] += 1
                    continue
                
                # Check region validity
                center_x = x + w/2
                center_y = y + h/2
                
                is_valid = False
                # Bottom region (more lenient)
                if y > bottom_y:
                    is_valid = True
                    min_area *= 0.7  # 进一步降低面积限制
                    max_area *= 1.3  # 进一步增大面积限制
                # Top region
                elif y < top_y:
                    is_valid = True
                # Left region
                elif x < left_x:
                    is_valid = True
                # Right region
                elif x > right_x:
                    is_valid = True
                # Center region
                elif (bottom_y > y > top_y and 
                      left_x < x < right_x and
                      left_x < center_x < right_x and
                      top_y < center_y < bottom_y):
                    is_valid = True
                
                if not is_valid:
                    filtered_stats['region'] += 1
                    continue
                
                # Check for overlap with existing elements
                has_overlap = False
                for ex, ey, ew, eh in elements:
                    # Calculate overlap
                    overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                    overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                    overlap_area = overlap_x * overlap_y
                    min_area = min(area, ew * eh)
                    if overlap_area > min_area * 0.4:  # 增大重叠阈值
                        has_overlap = True
                        break
                
                if has_overlap:
                    filtered_stats['overlap'] += 1
                    continue
                
                elements.append((x, y, w, h))
            
            # Log filtering stats
            self.logger.info(f"总轮廓数: {len(contours)}")
            self.logger.info(f"区域过滤: {filtered_stats['region']}")
            self.logger.info(f"面积过滤: {filtered_stats['area']}")
            self.logger.info(f"宽高比过滤: {filtered_stats['ratio']}")
            self.logger.info(f"重叠过滤: {filtered_stats['overlap']}")
            self.logger.info(f"剩余元素: {len(elements)}")
            
            return elements
        except Exception as e:
            self.logger.error(f"元素检测失败: {str(e)}")
            return []

    def _is_valid_region(self, x: int, y: int, w: int, h: int, img_width: int, img_height: int) -> bool:
        """检查元素是否在有效区域内"""
        center_x = x + w//2
        center_y = y + h//2
        
        # Define regions
        regions = {
            'bottom': {
                'y_min': int(img_height * 0.6),  # Expanded bottom region
                'y_max': img_height,
                'x_min': int(img_width * 0.1),
                'x_max': int(img_width * 0.9)
            },
            'left': {
                'y_min': int(img_height * 0.2),
                'y_max': int(img_height * 0.8),
                'x_min': 0,
                'x_max': int(img_width * 0.25)
            },
            'top': {
                'y_min': 0,
                'y_max': int(img_height * 0.3),
                'x_min': int(img_width * 0.1),
                'x_max': int(img_width * 0.9)
            },
            'right': {
                'y_min': int(img_height * 0.2),
                'y_max': int(img_height * 0.8),
                'x_min': int(img_width * 0.75),
                'x_max': img_width
            },
            'center': {
                'y_min': int(img_height * 0.3),
                'y_max': int(img_height * 0.7),
                'x_min': int(img_width * 0.2),
                'x_max': int(img_width * 0.8)
            }
        }
        
        # Check if element center is in any valid region
        for region in regions.values():
            if (region['x_min'] <= center_x <= region['x_max'] and
                region['y_min'] <= center_y <= region['y_max']):
                return True
        
        return False

    def _extract_features(self, image: np.ndarray, is_bottom: bool = False) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """提取图像特征"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Save original image
            if is_bottom:
                self._save_debug_image(gray, '6_original.png')
            
            # Resize to standard size
            target_size = (48, 64)  # Keep aspect ratio
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
            
            if is_bottom:
                self._save_debug_image(resized, '6_resized.png')
            
            # Simple preprocessing
            # 1. Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            enhanced = clahe.apply(resized)
            
            if is_bottom:
                self._save_debug_image(enhanced, '6_enhanced.png')
            
            # 2. Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
            
            if is_bottom:
                self._save_debug_image(blurred, '6_blurred.png')
            
            # 3. Adaptive threshold
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            if is_bottom:
                self._save_debug_image(binary, '6_binary.png')
            
            # Create ORB detector
            orb = cv2.ORB_create(
                nfeatures=100,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
            
            # Extract features from binary image
            keypoints, descriptors = orb.detectAndCompute(binary, None)
            
            # If not enough features, try enhanced image
            if len(keypoints) < 4:
                keypoints, descriptors = orb.detectAndCompute(enhanced, None)
            
            # Save debug image with keypoints
            if is_bottom:
                kp_image = cv2.drawKeypoints(binary, keypoints, None,
                                           color=(0, 255, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                self._save_debug_image(kp_image, '6_keypoints.png')
                self.logger.debug(f"提取到 {len(keypoints)} 个特征点")
            
            return keypoints, descriptors if descriptors is not None else np.array([])
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            return [], np.array([])

    def _match_template(self, element_kp: List[cv2.KeyPoint], element_des: np.ndarray,
                       template_kp: List[cv2.KeyPoint], template_des: np.ndarray,
                       is_bottom: bool = False) -> float:
        """模板匹配"""
        try:
            if len(element_kp) < 2 or len(template_kp) < 2:
                return 0.0
            
            # Create BFMatcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match descriptors
            matches = bf.match(element_des, template_des)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate score
            min_matches = 3
            if len(matches) < min_matches:
                return 0.0
            
            # Use top N matches
            top_matches = matches[:10]
            
            # Calculate average distance
            avg_distance = sum(m.distance for m in top_matches) / len(top_matches)
            
            # Calculate score
            base_score = 100 * (1 - avg_distance / 100)  # ORB distances are typically smaller
            match_bonus = min(20, len(matches))
            
            final_score = max(0, min(100, base_score + match_bonus))
            
            return final_score
        except Exception as e:
            self.logger.error(f"模板匹配失败: {str(e)}")
            return 0.0

    def _recognize_element(self, image: np.ndarray, x: int, y: int, w: int, h: int,
                          height: int) -> Tuple[str, float]:
        """识别单个元素"""
        try:
            # Extract element image
            element = image[y:y+h, x:x+w]
            
            # Check if in bottom region
            is_bottom = y > int(height * 0.65)
            
            # Extract features
            element_kp, element_des = self._extract_features(element, is_bottom)
            
            if len(element_kp) < 2:
                self.logger.warning("二值图提取到的特征点太少: %d", len(element_kp))
                return "", 0.0
            
            # Match with all templates
            best_match = ("", 0.0)
            for name, template in self.templates.items():
                template_kp, template_des = self._extract_features(template, is_bottom)
                score = self._match_template(element_kp, element_des,
                                          template_kp, template_des,
                                          is_bottom)
                if score > best_match[1]:
                    best_match = (name, score)
            
            return best_match
        except Exception as e:
            self.logger.error(f"元素识别失败: {str(e)}")
            return "", 0.0

    def _load_templates(self) -> None:
        """加载模板图片"""
        try:
            template_dir = self.config.get('template_dir', 'data/templates')
            if not os.path.exists(template_dir):
                self.logger.error(f"模板目录不存在: {template_dir}")
                return
            
            # 清空现有模板
            self.templates.clear()
            
            # 遍历模板目录
            for filename in os.listdir(template_dir):
                if filename.endswith('.png'):
                    template_path = os.path.join(template_dir, filename)
                    template_name = os.path.splitext(filename)[0]
                    
                    try:
                        # 读取模板图片
                        template = cv2.imread(template_path)
                        if template is None:
                            self.logger.error(f"无法读取模板: {template_path}")
                            continue
                        
                        # 调整模板大小
                        template_resized = cv2.resize(template, (48, 64))
                        
                        # 提取特征
                        keypoints, descriptors = self._extract_features(template_resized)
                        if keypoints is None or descriptors is None:
                            self.logger.error(f"无法从模板提取特征: {template_name}")
                            continue
                        
                        # 保存模板和特征
                        self.templates[template_name] = {
                            'image': template_resized,
                            'keypoints': keypoints,
                            'descriptors': descriptors
                        }
                        
                        self.logger.debug(f"已加载模板 {template_name}: {len(keypoints)} 个特征点")
                        
                    except Exception as e:
                        self.logger.error(f"处理模板 {template_name} 时出错: {str(e)}")
                        continue
            
            if not self.templates:
                self.logger.error("未能加载任何模板")
            else:
                self.logger.info(f"已加载 {len(self.templates)} 个模板")
            
        except Exception as e:
            self.logger.error(f"加载模板失败: {str(e)}")

    def batch_recognize_images(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """批量识别目录中的所有图片
        
        Args:
            input_dir: 输入图片目录路径
            output_dir: 输出结果目录路径，如果为None则使用配置文件中的output_dir
            
        Returns:
            Dict[str, Dict]: 包含所有图片识别结果的字典，key为图片文件名，value为识别结果
        """
        if output_dir is None:
            if "output_dir" not in self.config:
                raise ValueError("配置文件错误：缺少 output_dir 配置项")
            output_dir = self.config["output_dir"]
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = []
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        if "image_extensions" in self.config:
            image_extensions = self.config["image_extensions"]
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if not image_files:
            self.logger.warning(f"在目录 {input_dir} 中没有找到图片文件")
            return {}
            
        self.logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 清空结果文件
        summary_path = os.path.join(output_dir, "recognition_summary.json")
        self.logger.info(f"清空结果文件: {summary_path}")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('')
        
        # 处理每张图片
        results = []
        
        for image_path in image_files:
            try:
                self.logger.info(f"正在处理图片: {image_path}")
                result = self.recognize_file(image_path)
                
                # 构建识别结果
                recognition_data = {
                    "filename": os.path.basename(image_path),
                    "timestamp": datetime.now().isoformat(),
                    "result": result
                }
                
                # 将结果追加到文件（每个结果占一行）
                with open(summary_path, 'a', encoding='utf-8') as f:
                    json.dump(recognition_data, f, ensure_ascii=False, separators=(',', ':'))
                    f.write('\n')
                
                results.append(recognition_data)
                
            except Exception as e:
                self.logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
                continue
                
        self.logger.info(f"批量识别完成，共处理 {len(image_files)} 张图片")
        self.logger.info(f"汇总结果已保存到: {summary_path}")
        
        return results

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """识别图像中的元素"""
        try:
            # 预处理
            binary = self._preprocess_image(image)
            if binary is None:
                return []
            
            # 检测元素
            elements = self._detect_elements(binary)
            if not elements:
                self.logger.warning("未检测到任何元素")
                return []
            
            self.logger.info(f"检测到 {len(elements)} 个元素，开始识别")
            
            # 识别每个元素
            results = []
            for i, (x, y, w, h) in enumerate(elements, 1):
                name, score = self._recognize_element(binary, x, y, w, h, binary.shape[0])
                if score > 0:
                    self.logger.info(f"识别到元素 {i}: {name}, 得分: {score:.2f}")
                    results.append((name, score, (x, y, w, h)))
            
            self.logger.info(f"共识别到 {len(results)} 个元素")
            return results
        except Exception as e:
            self.logger.error(f"识别失败: {str(e)}")
            return []

def get_latest_screenshot() -> Optional[str]:
    """获取最新的截图"""
    logger = get_logger("ocr.screenshot")
    try:
        # 获取配置
        config = ConfigManager("config/app_config.yaml").get_config()
        if "screenshots_dir" not in config:
            raise ValueError("配置文件错误：缺少 screenshots_dir 配置项")
            
        screenshots_dir = config["screenshots_dir"]
        logger.info(f"正在搜索截图目录: {screenshots_dir}")
        
        # 获取所有PNG文件
        png_files = glob.glob(os.path.join(screenshots_dir, "**/*.png"), recursive=True)
        if not png_files:
            logger.warning(f"在目录 {screenshots_dir} 中未找到PNG文件")
            return None
            
        # 按修改时间排序
        latest_file = max(png_files, key=os.path.getctime)
        logger.info(f"找到最新的截图: {latest_file}")
        return latest_file
        
    except Exception as e:
        logger.error(f"获取最新截图失败: {str(e)}")
        traceback.print_exc()
        return None

def analyze_game_state(recognition_result: List[Dict]) -> Dict:
    """分析游戏状态"""
    logger = logging.getLogger("ocr.analyze")
    game_state = {
        'bottom': [],
        'left': [],
        'top': [],
        'right': [],
        'center': []
    }
    
    # 定义区域
    regions = {
        'bottom': {'y_min': 0.7, 'y_max': 1.0, 'x_min': 0.2, 'x_max': 0.8},
        'left': {'y_min': 0.2, 'y_max': 0.8, 'x_min': 0.0, 'x_max': 0.2},
        'top': {'y_min': 0.0, 'y_max': 0.3, 'x_min': 0.2, 'x_max': 0.8},
        'right': {'y_min': 0.2, 'y_max': 0.8, 'x_min': 0.8, 'x_max': 1.0},
        'center': {'y_min': 0.3, 'y_max': 0.7, 'x_min': 0.2, 'x_max': 0.8}
    }
    
    for result in recognition_result:
        # 获取边界框坐标
        x, y, x2, y2 = result['box']
        center_x = (x + x2) / 2
        center_y = (y + y2) / 2
        
        # 归一化坐标
        norm_x = center_x / IMAGE_WIDTH
        norm_y = center_y / IMAGE_HEIGHT
        
        # 确定元素所在区域
        for region_name, region_bounds in regions.items():
            if (region_bounds['x_min'] <= norm_x <= region_bounds['x_max'] and
                region_bounds['y_min'] <= norm_y <= region_bounds['y_max']):
                game_state[region_name].append(result['name'])
                break
    
    # 记录每个区域识别到的牌数
    for region_name, tiles in game_state.items():
        logger.info(f"{region_name}区域识别到{len(tiles)}张牌")
        logger.info(f"{region_name}区域识别结果: {', '.join(tiles)}")
    
    return game_state

def verify_recognition_results(game_state: Dict) -> bool:
    """验证识别结果"""
    success = True
    
    # 验证手牌区域（bottom）
    if len(game_state['bottom']) == 0:
        print("bottom: ✗ 没有识别到手牌")
        success = False
    else:
        print(f"bottom: ✓ 识别到{len(game_state['bottom'])}张牌")
    
    # 验证其他区域
    for region in ['left', 'top', 'right', 'center']:
        print(f"{region}: ✓ 识别到{len(game_state[region])}个元素")
    
    return success

def run_recognition_test():
    """运行识别测试"""
    logger = get_logger("ocr.test")
    
    try:
        logger.info("开始运行识别测试...")
        
        # 清空结果文件
        result_file = "output/recognition_summary.json"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        logger.info(f"清空结果文件: {result_file}")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        
        # 获取最新的截图
        logger.info("正在获取最新截图...")
        latest_image = get_latest_screenshot()
        if not latest_image:
            logger.error("未找到可用的截图")
            return False
        
        # 识别图片
        logger.info(f"开始识别图片: {latest_image}")
        ocr = OCR(debug_mode=True)  # 启用调试模式
        result = ocr.recognize_image(latest_image)
        
        if not result:
            logger.error("图片识别失败")
            return False
        
        # 分析游戏状态
        logger.info("开始分析游戏状态...")
        game_state = analyze_game_state(result)
        
        # 验证结果
        print("\n=== 识别结果验证 ===")
        success = verify_recognition_results(game_state)
        
        # 保存结果
        save_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': latest_image,
            'game_state': game_state,
            'success': success
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        logger.info("识别测试完成")
        logger.info(f"结果已保存到: {result_file}")
        
        return success
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logger = logging.getLogger("ocr.main")
    try:
        # 运行识别测试
        logger.info("开始运行识别测试...")
        test_result = run_recognition_test()
        
        if test_result:
            logger.info("识别测试完成")
            # 检查结果文件
            summary_file = "output/recognition_summary.json"
            if os.path.exists(summary_file):
                logger.info(f"结果已保存到: {summary_file}")
            else:
                logger.warning("结果文件未生成")
        else:
            logger.error("识别测试失败")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
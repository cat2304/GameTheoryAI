"""
麻将牌识别工具类
=============

提供基于OpenCV和OCR的麻将牌识别功能。
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging

from .config_utils import config
from .log_utils import setup_logger

class MahjongOCR:
    """麻将牌识别类
    
    提供麻将牌识别功能，包括：
    - 图片预处理
    - 麻将牌区域检测
    - 牌面文字识别
    """
    
    def __init__(self):
        """初始化麻将牌识别工具"""
        self.logger = setup_logger("mahjong_ocr", "logs/mahjong_ocr.log")
        self._load_config()
        self.logger.info("MahjongOCR 初始化完成")
    
    def _load_config(self) -> None:
        """加载配置"""
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
            # 麻将牌检测参数
            'detection': {
                'min_area': 3000,          # 最小面积
                'max_area': 50000,         # 最大面积
                'aspect_ratio_min': 0.6,   # 最小宽高比
                'aspect_ratio_max': 1.5,   # 最大宽高比
                'min_width': 30,           # 最小宽度
                'max_width': 300,          # 最大宽度
                'y_threshold': 150,        # 底部区域阈值
                'expected_tiles': 13,      # 期望的麻将牌数量
                'tile_gap': 5,            # 麻将牌间隔阈值
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
    
    def _save_debug_image(self, name: str, image: np.ndarray, directory: str = "debug") -> None:
        """保存调试图片
        
        Args:
            name: 图片名称
            image: 图片数据
            directory: 调试图片保存目录
        """
        if self.config['preprocessing']['debug_output']:
            try:
                debug_dir = Path(directory)
                debug_dir.mkdir(parents=True, exist_ok=True)
                output_path = debug_dir / f"{name}.png"
                cv2.imwrite(str(output_path), image)
                self.logger.debug(f"保存调试图片: {output_path}")
            except Exception as e:
                self.logger.error(f"保存调试图片失败: {str(e)}")
    
    def _merge_nearby_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """合并相近的区域
        
        Args:
            regions: 区域列表 [(x, y, w, h), ...]
            
        Returns:
            List[Tuple[int, int, int, int]]: 合并后的区域列表
        """
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
    
    def find_mahjong_tiles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """查找麻将牌区域"""
        try:
            height, width = image.shape[:2]
            self.logger.debug(f"图片尺寸: {width}x{height}")
            
            # 只处理图片底部区域
            y_threshold = self.config['detection']['y_threshold']
            bottom_region = image[height - y_threshold:, :]
            self._save_debug_image("01_bottom_region", bottom_region)
            
            # 查找轮廓
            contours, hierarchy = cv2.findContours(bottom_region,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制轮廓调试图
            debug_contours = cv2.cvtColor(bottom_region, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
            self._save_debug_image("02_contours", debug_contours)
            
            tiles = []
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
                
                tiles.append((x, y, w, h))
            
            # 合并相近的区域
            tiles = self._merge_nearby_regions(tiles)
            
            # 检查识别到的牌数是否合理
            if len(tiles) < 10:  # 最少应该有10张牌
                self.logger.warning(f"识别到的麻将牌数量过少: {len(tiles)}")
                # 尝试调整参数重新识别
                if len(tiles) < config['expected_tiles']:
                    self.logger.info("尝试调整参数重新识别...")
                    # 临时降低阈值
                    config['min_area'] *= 0.8
                    config['min_width'] *= 0.8
                    return self.find_mahjong_tiles(image)
            
            # 按x坐标排序
            tiles.sort(key=lambda x: x[0])
            
            # 检查牌的间距
            for i in range(len(tiles) - 1):
                gap = tiles[i+1][0] - (tiles[i][0] + tiles[i][2])
                if gap > config['tile_gap']:
                    self.logger.warning(f"检测到异常间距: {gap}像素 (位置: {i+1})")
            
            # 绘制检测结果调试图
            debug_result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for (x, y, w, h) in tiles:
                cv2.rectangle(debug_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self._save_debug_image("03_detection", debug_result)
            
            return tiles
            
        except Exception as e:
            self.logger.error(f"查找麻将牌区域失败: {str(e)}")
            raise
    
    def extract_tile_features(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """提取麻将牌特征
        
        Args:
            image: 输入图片
            region: 麻将牌区域 (x, y, w, h)
            
        Returns:
            Dict[str, Any]: 特征字典，包含颜色特征和形状特征
        """
        try:
            x, y, w, h = region
            tile = image[y:y+h, x:x+w]
            
            # 保存原始牌图调试图
            self._save_debug_image(f"tile_{x}_{y}", tile)
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            
            # 提取红色区域
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # 提取绿色区域
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 计算颜色特征
            total_pixels = w * h
            red_ratio = cv2.countNonZero(red_mask) / total_pixels
            green_ratio = cv2.countNonZero(green_mask) / total_pixels
            
            # 转换为灰度图并二值化
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 
                                   self.config['recognition']['text_threshold'],
                                   255, cv2.THRESH_BINARY)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 计算形状特征
            shape_features = {
                'contour_count': len(contours),
                'contour_areas': [],
                'contour_perimeters': [],
                'circularity': []
            }
            
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # 计算圆形度
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                shape_features['contour_areas'].append(area)
                shape_features['contour_perimeters'].append(perimeter)
                shape_features['circularity'].append(circularity)
            
            # 保存特征调试图
            if self.config['recognition']['debug_features']:
                debug_features = np.zeros_like(tile)
                cv2.drawContours(debug_features, contours, -1, (0, 255, 0), 2)
                self._save_debug_image(f"features_{x}_{y}", debug_features)
            
            return {
                'color_features': {
                    'red_ratio': red_ratio,
                    'green_ratio': green_ratio
                },
                'shape_features': shape_features
            }
            
        except Exception as e:
            self.logger.error(f"提取麻将牌特征失败: {str(e)}")
            raise
    
    def recognize_tile(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """识别单个麻将牌
        
        Args:
            image: 原始图片
            region: (x, y, w, h) 麻将牌区域
            
        Returns:
            str: 识别结果
        """
        try:
            features = self.extract_tile_features(image, region)
            config = self.config['recognition']
            
            # 检测WM
            if features['color_features']['green_ratio'] > config['green_threshold']:
                return "WM"
            
            # 检测万字牌
            if features['color_features']['red_ratio'] > config['red_threshold']:
                if features['shape_features']['contour_count'] == 1:
                    return "一万"
                elif features['shape_features']['contour_count'] == 2:
                    return "二万"
                elif features['shape_features']['contour_count'] == 3:
                    return "三万"
                elif features['shape_features']['contour_count'] == 8:
                    return "八万"
            
            # 检测筒子牌
            if features['shape_features']['circularity'][-1] >= 0.7:
                return "筒"
            
            # 检测字牌
            if features['shape_features']['contour_count'] == 1:
                if features['shape_features']['contour_areas'][0] > 1000:
                    return "南"
                else:
                    return "北"
            
            return "未知"
            
        except Exception as e:
            self.logger.error(f"识别麻将牌失败: {str(e)}")
            raise
    
    def recognize_image(self, image_path: str) -> Dict[str, Any]:
        """识别图片中的所有麻将牌"""
        try:
            self.logger.info(f"开始识别图片: {image_path}")
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 保存原始图片
            self._save_debug_image("00_original", image)
            
            # 预处理
            processed = self.preprocess_image(image)
            self._save_debug_image("01_processed", processed)
            
            # 查找麻将牌区域
            tiles = self.find_mahjong_tiles(processed)
            
            # 检查识别到的牌数
            expected_tiles = self.config['detection']['expected_tiles']
            if len(tiles) < 10 or len(tiles) > expected_tiles + 2:
                self.logger.warning(f"识别到的麻将牌数量异常: {len(tiles)} (期望: {expected_tiles})")
            
            # 识别每个麻将牌
            results = []
            for region in tiles:
                result = self.recognize_tile(image, region)
                results.append({
                    'region': region,
                    'result': result
                })
            
            # 统计识别结果
            result_stats = {}
            for r in results:
                tile_type = r['result']
                result_stats[tile_type] = result_stats.get(tile_type, 0) + 1
            
            self.logger.info(f"识别结果统计: {result_stats}")
            
            return {
                'success': True,
                'tiles': results,
                'preprocessed': processed,
                'total_tiles': len(tiles),
                'statistics': result_stats
            }
            
        except Exception as e:
            self.logger.error(f"识别图片失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def recognize_images_in_directory(self, directory: str) -> Dict[str, Any]:
        """识别指定目录下的所有图片
        
        Args:
            directory: 图片目录路径
            
        Returns:
            Dict[str, Any]: 识别结果统计
        """
        try:
            self.logger.info(f"开始识别目录: {directory}")
            
            # 支持的图片格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            # 统计结果
            results = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'details': []
            }
            
            # 遍历目录
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and Path(file_path).suffix.lower() in image_extensions:
                    results['total'] += 1
                    try:
                        result = self.recognize_image(file_path)
                        if result['success']:
                            results['success'] += 1
                        else:
                            results['failed'] += 1
                        results['details'].append({
                            'file': file,
                            'result': result
                        })
                    except Exception as e:
                        results['failed'] += 1
                        results['details'].append({
                            'file': file,
                            'error': str(e)
                        })
            
            # 记录统计结果
            self.logger.info(f"目录识别完成: {directory}")
            self.logger.info(f"总计: {results['total']} 张图片")
            self.logger.info(f"成功: {results['success']} 张")
            self.logger.info(f"失败: {results['failed']} 张")
            
            return results
            
        except Exception as e:
            self.logger.error(f"目录识别失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def visualize_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """可视化识别结果
        
        Args:
            image: 原始图片
            results: 识别结果列表
            
        Returns:
            np.ndarray: 可视化后的图片
        """
        try:
            # 创建副本
            vis_image = image.copy()
            
            # 绘制每个麻将牌的区域和识别结果
            for result in results:
                x, y, w, h = result['region']
                text = result['result']
                
                # 绘制矩形
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 绘制文本
                cv2.putText(vis_image, text, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return vis_image
            
        except Exception as e:
            self.logger.error(f"可视化结果失败: {str(e)}")
            return image

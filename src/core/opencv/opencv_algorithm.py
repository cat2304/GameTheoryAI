import cv2
import numpy as np
from pathlib import Path
from src.utils.log_utils import get_logger

class OpenCVAlgorithm:
    def __init__(self):
        self.logger = get_logger("opencv_algorithm")
        
        # 麻将牌检测参数
        self.min_area = 3000  # 最小区域面积
        self.max_area = 15000  # 最大区域面积
        self.aspect_ratio_min = 0.6  # 最小宽高比
        self.aspect_ratio_max = 1.5  # 最大宽高比
        self.bottom_height = 150  # 底部区域高度
    
    def preprocess_image(self, img_path):
        """图像预处理"""
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值二值化
            thresh = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 4)
            
            # 降噪处理
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return processed, img
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def find_mahjong_tiles(self, processed_img, original_img):
        """查找麻将牌区域"""
        try:
            # 查找轮廓
            contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选合适的轮廓
            tiles = []
            height, width = processed_img.shape
            bottom_region = processed_img[height-self.bottom_height:height, :]
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_area <= area <= self.max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / h
                    
                    # 检查宽高比和位置
                    if (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max and 
                        y + h >= height - self.bottom_height):
                        tiles.append({
                            'region': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            # 按x坐标排序
            tiles.sort(key=lambda t: t['region'][0])
            
            self.logger.info(f"找到 {len(tiles)} 个可能的麻将牌区域")
            return tiles
        except Exception as e:
            self.logger.error(f"查找麻将牌失败: {str(e)}")
            raise
    
    def recognize_tile(self, original_img, tile):
        """识别单个麻将牌"""
        try:
            x, y, w, h = tile['region']
            tile_img = original_img[y:y+h, x:x+w]
            
            # TODO: 实现具体的麻将牌识别逻辑
            # 这里可以添加特征提取和模式匹配
            
            return "未知"
        except Exception as e:
            self.logger.error(f"识别麻将牌失败: {str(e)}")
            raise
    
    def visualize_results(self, image, tiles):
        """可视化识别结果"""
        try:
            # 创建图像副本
            vis_image = image.copy()
            
            # 为每个识别到的麻将牌绘制边框和标签
            for tile in tiles:
                # 获取区域坐标
                x, y, w, h = tile['region']
                
                # 绘制矩形框
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加识别结果标签
                label = tile.get('result', '未知')
                cv2.putText(vis_image, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return vis_image
        except Exception as e:
            self.logger.error(f"可视化结果失败: {str(e)}")
            raise 
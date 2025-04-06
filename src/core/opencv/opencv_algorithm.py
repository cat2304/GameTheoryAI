import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

class OpenCVAlgorithm:
    """OpenCV算法类，提供图像处理和游戏元素检测功能"""
    
    def __init__(self):
        """初始化OpenCV算法类"""
        self.logger = logging.getLogger(__name__)
        
        # 游戏元素检测参数
        self.params = {
            'min_area': 500,         # 最小区域面积
            'max_area': 10000,       # 最大区域面积
            'aspect_ratio_min': 0.5, # 最小宽高比
            'aspect_ratio_max': 2.0, # 最大宽高比
            'y_threshold': 200,      # 底部区域高度阈值
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 处理后的图像
        """
        try:
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            
            # 膨胀操作，连接边缘
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            return dilated
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def find_game_elements(self, processed_img, original_img):
        """查找游戏元素区域
        
        Args:
            processed_img: 处理后的图像
            original_img: 原始图像
            
        Returns:
            List[Dict]: 检测到的游戏元素列表
        """
        try:
            height, width = processed_img.shape[:2]
            
            # 只处理底部区域
            y_start = max(0, height - self.params['y_threshold'])
            bottom_region = processed_img[y_start:height, :]
            
            # 查找轮廓
            contours, _ = cv2.findContours(bottom_region, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选轮廓
            elements = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 根据面积筛选
                if area < self.params['min_area'] or area > self.params['max_area']:
                    continue
                
                # 获取外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                y += y_start  # 调整y坐标到原图
                
                # 根据宽高比筛选
                aspect_ratio = w / float(h)
                if (aspect_ratio < self.params['aspect_ratio_min'] or 
                    aspect_ratio > self.params['aspect_ratio_max']):
                    continue
                
                # 保存元素信息
                elements.append({
                    'region': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (x + w//2, y + h//2)
                })
            
            # 按从左到右排序
            elements.sort(key=lambda e: e['region'][0])
            
            # 绘制检测结果
            result_img = original_img.copy()
            for element in elements:
                x, y, w, h = element['region']
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            self.logger.info(f"找到 {len(elements)} 个游戏元素区域")
            return elements, result_img
            
        except Exception as e:
            self.logger.error(f"查找游戏元素失败: {str(e)}")
            return [], original_img.copy()
    
    def recognize_game_element(self, image, region):
        """识别单个游戏元素
        
        Args:
            image: 原始图像
            region: 游戏元素区域 (x, y, w, h)
            
        Returns:
            str: 识别结果
        """
        try:
            x, y, w, h = region
            element_img = image[y:y+h, x:x+w]
            
            # TODO: 实现具体的游戏元素识别逻辑
            
            # 简单的颜色特征提取示例
            hsv = cv2.cvtColor(element_img, cv2.COLOR_BGR2HSV)
            
            # 计算颜色分布
            h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [3], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [3], [0, 256])
            
            # 根据颜色特征判断元素类型
            h_dominant = np.argmax(h_hist)
            s_avg = np.mean(hsv[:,:,1])
            v_avg = np.mean(hsv[:,:,2])
            
            if h_dominant < 3 or h_dominant > 15:  # 红色区域
                return "红色元素"
            elif 3 <= h_dominant <= 6:  # 黄色区域
                return "黄色元素"
            elif 6 < h_dominant <= 10:  # 绿色区域
                return "绿色元素"
            elif 10 < h_dominant <= 13:  # 蓝色区域
                return "蓝色元素"
            else:
                return "其他元素"
                
        except Exception as e:
            self.logger.error(f"识别游戏元素失败: {str(e)}")
            return "未知"
    
    def draw_visualization(self, image, elements, predictions):
        """绘制可视化结果
        
        Args:
            image: 原始图像
            elements: 检测到的游戏元素
            predictions: 识别结果
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        try:
            # 创建图像副本
            vis_img = image.copy()
            
            # 为每个识别到的游戏元素绘制边框和标签
            for element, prediction in zip(elements, predictions):
                x, y, w, h = element['region']
                
                # 绘制矩形
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 绘制标签背景
                cv2.rectangle(vis_img, (x, y-20), (x+w, y), (0, 255, 0), -1)
                
                # 绘制文本
                cv2.putText(vis_img, prediction, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return vis_img
            
        except Exception as e:
            self.logger.error(f"绘制可视化结果失败: {str(e)}")
            return image 
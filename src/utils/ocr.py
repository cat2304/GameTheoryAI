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

# 导入配置管理器
from src.utils.utils import ConfigManager

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
        self.logger = get_logger("ocr.core")
        self.config = {
            'preprocessing': {
                'blur_kernel': (5,5),        # 增大模糊核减少噪声
                'min_threshold': 100,        # 降低二值化阈值
                'max_threshold': 255,
                'canny_min': 30,  # 原60 → 降低以保留更多边缘
                'canny_max': 150, # 原180 → 缩小范围
                'dilate_kernel': (3, 3),
                'dilate_iterations': 1  # 原0 → 添加膨胀操作连接断裂区域
            },
            'detection': {
                'min_area': 5,        # 原1 → 防止噪声
                'max_area': 999999,
                'min_aspect_ratio': 0.01,
                'max_aspect_ratio': 8.0,  # 原100 → 更合理范围
                'min_width': 1,
                'max_width': 999,
                'min_height': 1,
                'max_height': 999,
                'y_threshold': 50     # 新增Y坐标过滤
            },
            'recognition': {
                'template_matching': {
                    'threshold': 0.25,      # 降低匹配阈值
                    'scale_range': [0.5, 2.0],  # 缩小范围提升密度
                    'scale_steps': 15,           # 增加缩放采样点
                    'rotation_range': [-45, 45],  # 更大旋转角度
                    'rotation_steps': 10         # 增加旋转采样点
                },
                'ocr': {
                    'config': '--psm 6 --oem 3',
                    'whitelist': '0123456789万条筒东南西北中发白'
                }
            }
        }
        
        # 加载模板图片
        self.templates = self._load_templates()
        self.logger.info("游戏状态识别工具初始化完成")
        
        # 新增类属性
        self.debug_dir = Path("debug")
        self.debug_dir.mkdir(exist_ok=True)
    
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """加载模板图片（增加尺寸校验）"""
        self.logger.info("开始加载模板图片...")
        template_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, "data", "tencent")
        self.logger.debug(f"模板目录: {template_dir} 存在性校验: {os.path.exists(template_dir)}")
        
        try:
            file_list = os.listdir(template_dir)
            self.logger.info(f"发现{len(file_list)}个候选文件，开始过滤PNG文件...")
            
            templates = {}
            for filename in file_list:
                if not filename.endswith('.png'):
                    self.logger.debug(f"跳过非PNG文件: {filename}")
                    continue
                    
                filepath = os.path.join(template_dir, filename)
                self.logger.debug(f"正在处理模板: {filename} 路径: {filepath}")
                
                img = cv2.imread(filepath)
                if img is None:
                    self.logger.error(f"无法读取模板文件: {filepath}")
                    continue
                
                # 记录原始模板信息
                self.logger.debug(f"模板[{filename}] 原始尺寸: {img.shape} 通道数: {img.shape[2] if len(img.shape)==3 else 1}")
                
                # 增强模板预处理日志
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.logger.debug(f"模板[{filename}] 灰度化后直方图统计: 均值={np.mean(gray):.1f} 方差={np.var(gray):.1f}")
                
                blurred = cv2.GaussianBlur(gray, (5,5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.logger.debug(f"模板[{filename}] OTSU阈值: {_} 二值化后白像素占比: {np.sum(thresh==255)/thresh.size:.1%}")
                
                kernel = np.ones((3,3), np.uint8)
                opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                self.logger.debug(f"模板[{filename}] 开运算后连通域数量: {len(cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}")
                
                templates[filename] = opened
                self.logger.info(f"成功加载模板: {filename} 最终尺寸: {opened.shape}")

            self.logger.info(f"共加载{len(templates)}个有效模板")
            return templates
            
        except Exception as e:
            self.logger.error("模板加载过程异常", exc_info=True)
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理（增加直方图日志）"""
        try:
            # 新增调试目录定义
            self.debug_dir.mkdir(exist_ok=True)
            
            # 记录输入图像状态
            self.logger.debug(f"输入图像 尺寸: {image.shape} 类型: {image.dtype} 通道数: {image.shape[2] if len(image.shape)==3 else 1}")
            self.logger.debug(f"输入图像像素统计 最小值: {np.min(image)} 最大值: {np.max(image)} 均值: {np.mean(image):.1f}")
            
            # 确保输入始终为BGR三通道
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # HSV颜色处理
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 50, 50])    # 降低饱和度和亮度要求
            upper_red = np.array([30, 255, 255]) # 扩大色相范围
            hsv_mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # 修正LAB颜色空间处理（添加归一化）
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab = cv2.normalize(lab.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)  # 新增归一化
            lab = lab.astype('uint8')
            
            # 优化LAB空间处理
            lab_mask = cv2.inRange(lab[:,:,1], 120, 200) # 扩大阈值范围
            
            # 添加HSV通道直方图均衡化
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])  # 增强亮度通道
            
            # 修改合并策略为动态阈值融合
            combined_mask = cv2.addWeighted(hsv_mask, 0.6, lab_mask, 0.4, 30)  # 增加偏移量
            _, combined_mask = cv2.threshold(combined_mask, 160, 255, cv2.THRESH_BINARY)  # 降低阈值
            
            # 添加形态学优化（开运算去除小噪点）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 添加自适应区域增强
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:  # 只处理大区域
                    x,y,w,h = cv2.boundingRect(cnt)
                    combined_mask[y:y+h, x:x+w] = 255  # 填充有效区域
            
            # 在颜色处理部分添加调试输出
            cv2.imwrite(str(self.debug_dir / "hsv_mask_debug.png"), hsv_mask)
            cv2.imwrite(str(self.debug_dir / "lab_mask_debug.png"), lab_mask)
            
            # 添加掩码有效性检查
            mask_valid_ratio = np.sum(combined_mask == 255) / combined_mask.size
            self.logger.critical(f"颜色掩码有效性检查: HSV占比{np.sum(hsv_mask==255)/hsv_mask.size:.1%} "
                                f"LAB占比{np.sum(lab_mask==255)/lab_mask.size:.1%} "
                                f"合并后占比{mask_valid_ratio:.1%}")
            
            # 在Canny检测后添加边缘可视化
            cv2.imwrite(str(self.debug_dir / "canny_edges.png"), combined_mask)
            self.logger.debug(f"边缘像素统计: 有效边缘{np.sum(combined_mask == 255)}像素")
            
            # 在preprocess_image末尾添加
            cv2.imwrite(str(self.debug_dir / "final_mask.png"), combined_mask)
            self.logger.critical(f"最终掩码有效性: {np.sum(combined_mask==255)/combined_mask.size:.1%}")
            
            self.logger.debug(f"LAB通道直方图: {np.histogram(lab[:,:,1], bins=10)[0]}")
            
            # 添加形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            processed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # 添加预处理保护机制
            if processed.sum() == 0:
                # 使用备用处理方案
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            
            return processed
            
        except Exception as e:
            self.logger.error(f"图片预处理失败: {str(e)}", exc_info=True)
            raise
    
    def find_game_elements(self, image: np.ndarray) -> List[Dict]:
        """查找游戏元素区域（修复空数组问题）"""
        try:
            # 输入验证
            if len(image.shape) not in [2, 3]:
                raise ValueError(f"非法图像格式，通道数异常: {image.shape}")
            
            # 统一转换为三通道处理
            if len(image.shape) == 2:
                working_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                working_image = image.copy()

            # 预处理流程
            processed = self.preprocess_image(working_image)
            
            if processed.sum() == 0:
                self.logger.error("预处理结果全黑，启用紧急模式")
                return self.emergency_detection(image)
            
            # 调试信息记录
            self.logger.debug(f"预处理后图像形状: {processed.shape}")
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / "processed.png"), processed)

            # 轮廓检测
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.logger.info(f"发现原始轮廓数量: {len(contours)}")

            # 增强轮廓统计（添加空数组保护）
            if len(contours) > 0:
                contour_areas = [cv2.contourArea(c) for c in contours]
                self.logger.info(
                    f"轮廓面积统计: 中位数={np.median(contour_areas):.0f} "
                    f"最大={np.max(contour_areas):.0f} 最小={np.min(contour_areas):.0f} "
                    f"Q1={np.quantile(contour_areas, 0.25):.0f} Q3={np.quantile(contour_areas, 0.75):.0f}"
                )
            else:
                self.logger.warning("未检测到任何轮廓，请检查预处理结果")
                cv2.imwrite(str(self.debug_dir / "empty_contours.jpg"), processed)
                self.logger.error("预处理结果图像已保存至 empty_contours.jpg")
                return []  # 提前返回空列表

            # 过滤逻辑
            valid_elements = []
            filter_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h != 0 else 0

                # 过滤条件检查
                filters = [
                    (y < self.config['detection']['y_threshold'], 
                     f"Y坐标{y} < {self.config['detection']['y_threshold']}"),
                    (area < self.config['detection']['min_area'], 
                     f"面积{area:.0f} < {self.config['detection']['min_area']}"),
                    (area > self.config['detection']['max_area'], 
                     f"面积{area:.0f} > {self.config['detection']['max_area']}"),
                    (aspect_ratio < self.config['detection']['min_aspect_ratio'], 
                     f"宽高比{aspect_ratio:.2f} < {self.config['detection']['min_aspect_ratio']}"),
                    (aspect_ratio > self.config['detection']['max_aspect_ratio'], 
                     f"宽高比{aspect_ratio:.2f} > {self.config['detection']['max_aspect_ratio']}"),
                    (w < self.config['detection']['min_width'], 
                     f"宽度{w} < {self.config['detection']['min_width']}"),
                    (w > self.config['detection']['max_width'], 
                     f"宽度{w} > {self.config['detection']['max_width']}"),
                    (h < self.config['detection']['min_height'], 
                     f"高度{h} < {self.config['detection']['min_height']}"),
                    (h > self.config['detection']['max_height'], 
                     f"高度{h} > {self.config['detection']['max_height']}")
                ]

                # 检查过滤条件
                filtered = False
                for condition, msg in filters:
                    if condition:
                        self.logger.debug(f"轮廓{idx}被过滤 - {msg}")
                        filtered = True
                        filter_counts[0] += 1 if condition else 0
                        filter_counts[1] += 1 if condition else 0
                        filter_counts[2] += 1 if condition else 0
                        filter_counts[3] += 1 if condition else 0
                        filter_counts[4] += 1 if condition else 0
                        filter_counts[5] += 1 if condition else 0
                        filter_counts[6] += 1 if condition else 0
                        filter_counts[7] += 1 if condition else 0
                        filter_counts[8] += 1 if condition else 0
                        break
                    
                if not filtered:
                    valid_elements.append({
                        'contour': contour,
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
                    self.logger.debug(f"轮廓{idx}通过检查: X={x}, Y={y}, W={w}, H={h}, 面积={area:.0f}, 宽高比={aspect_ratio:.2f}")

            # 按Y坐标排序
            valid_elements.sort(key=lambda e: e['y'])
            
            # 保存调试图像
            debug_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR) if len(processed.shape) == 2 else processed.copy()
            for i, elem in enumerate(valid_elements):
                x, y, w, h = elem['x'], elem['y'], elem['width'], elem['height']
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(debug_dir / "detected_elements.png"), debug_image)
            
            # 在过滤检查后添加统计
            self.logger.info(
                f"过滤统计: 总轮廓{len(contours)} -> "
                f"因Y坐标过滤{filter_counts[0]}, "
                f"面积过滤{filter_counts[1]+filter_counts[2]}, "
                f"宽高比过滤{filter_counts[3]+filter_counts[4]}, "
                f"尺寸过滤{filter_counts[5]+filter_counts[6]+filter_counts[7]+filter_counts[8]}"
            )
            
            # 在保存调试图像后添加
            cv2.imwrite(str(debug_dir / "final_elements.jpg"), debug_image)
            
            # 候选元素分析
            if valid_elements:
                areas = [e['area'] for e in valid_elements]
                self.logger.info(
                    f"有效元素面积统计: 中位数={np.median(areas):.0f} "
                    f"最大={np.max(areas):.0f} 最小={np.min(areas):.0f}"
                )
            else:
                self.logger.warning("未发现任何有效元素")
            
            # 在find_game_elements开头添加
            cv2.imwrite(str(self.debug_dir / "original_input.jpg"), working_image)
            
            return valid_elements
            
        except Exception as e:
            self.logger.error(f"元素检测失败: {str(e)}", exc_info=True)
            raise
    
    def _template_match(self, roi: np.ndarray) -> Optional[str]:
        """模板匹配（增加匹配过程日志）"""
        try:
            self.logger.debug(f"开始模板匹配 ROI尺寸: {roi.shape} 通道数: {roi.shape[2] if len(roi.shape)==3 else 1}")
            
            # 新增ROI预处理流程
            roi = cv2.resize(roi, (64, 64))  # 标准化尺寸
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
            _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 新增二值化
            
            # 记录ROI预处理状态
            cv2.imwrite("debug/current_roi.png", roi_bin)  # 保存二值化结果
            self.logger.debug(f"ROI预处理后 白像素占比: {np.sum(roi_bin==255)/roi_bin.size:.1%}")
            
            # 模板循环增加统计
            total_templates = len(self.templates)
            self.logger.info(f"开始匹配 {total_templates} 个模板")
            
            # 修改匹配策略
            best_score = 0
            best_name = None
            for name, template in self.templates.items():
                # 使用多方法组合评分
                res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, _, _ = cv2.minMaxLoc(res)
                current_score = max_val
                
                if current_score > best_score:
                    best_score = current_score
                    best_name = name
            
            # 动态调整阈值
            dynamic_threshold = max(0.2, best_score - 0.1)
            return best_name if best_score > dynamic_threshold else None
            
        except Exception as e:
            self.logger.error(f"模板匹配失败: {str(e)}", exc_info=True)
            return None
    
    def _ocr_recognize(self, roi: np.ndarray) -> str:
        """OCR识别（增加图像状态日志）"""
        try:
            # 记录输入ROI状态
            self.logger.debug(f"OCR输入 ROI尺寸: {roi.shape} 平均亮度: {np.mean(roi):.1f}")
            cv2.imwrite("debug/ocr_input.png", roi)
            
            # 输入验证
            if roi.size == 0:
                self.logger.warning("空ROI区域")
                return ""
                
            # 统一通道处理
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                
            # 图像增强
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            blurred = cv2.medianBlur(denoised, 3)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # OCR识别
            text = pytesseract.image_to_string(
                binary,
                config=self.config['recognition']['ocr']['config'],
                lang='chi_sim'
            )
            
            # 结果过滤
            clean_text = ''.join([c for c in text.strip() if c in self.config['recognition']['ocr']['whitelist']])
            self.logger.debug(f"原始OCR结果: {text} -> 过滤后: {clean_text}")
            
            self.logger.debug(f"去噪后图像质量: PSNR={cv2.PSNR(gray, denoised):.2f}dB")
            self.logger.debug(f"二值化阈值: {_} 白像素占比: {np.sum(binary==255)/binary.size:.1%}")
            
            # OCR结果分析
            self.logger.info(f"原始OCR结果: {text} 过滤后: {clean_text}")
            if len(clean_text) != len(text.strip()):
                removed_chars = set(text.strip()) - set(clean_text)
                self.logger.warning(f"过滤掉非常用字符: {removed_chars}")
            
            return clean_text
            
        except Exception as e:
            self.logger.error(f"OCR识别失败: {str(e)}", exc_info=True)
            return ""
    
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
            
            # 对每个元素进行识别
            results = []
            for element in elements:
                # 提取元素区域
                x, y, w, h = element['x'], element['y'], element['width'], element['height']
                roi = image[y:y+h, x:x+w]
                
                # 先尝试模板匹配
                match_result = self._template_match(roi)
                if match_result:
                    results.append({
                        'region': (x, y, w, h),
                        'result': match_result,
                        'method': 'template'
                    })
                else:
                    # 如果模板匹配失败，使用OCR识别
                    text = self._ocr_recognize(roi)
                    if text:
                        results.append({
                            'region': (x, y, w, h),
                            'result': text,
                            'method': 'ocr'
                        })
            
            self.logger.debug(f"识别到{len(results)}个元素")
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
    
    def draw_visualization(self, image: np.ndarray, elements: List[Dict[str, Any]], predictions: List[str]) -> np.ndarray:
        """绘制可视化结果"""
        try:
            # 创建图像副本
            vis_img = image.copy()
            
            # 为每个识别到的游戏元素绘制边框和标签
            for element, prediction in zip(elements, predictions):
                x, y, w, h = element['x'], element['y'], element['width'], element['height']
                
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

    def _feature_match(self, roi: np.ndarray) -> Optional[str]:
        # 初始化SIFT检测器
        sift = cv2.SIFT_create()
        
        # 检测关键点和描述符
        kp1, des1 = sift.detectAndCompute(roi, None)
        
        best_match = None
        best_score = 0
        
        for template_name, template in self.templates.items():
            kp2, des2 = sift.detectAndCompute(template, None)
            
            # FLANN匹配器
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # 筛选优质匹配
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            
            # 计算匹配分数
            score = len(good) / len(matches) if len(matches)>0 else 0
            if score > best_score:
                best_score = score
                best_match = template_name
        
        return best_match if best_score > 0.3 else None

    def enhanced_ocr_processing(self, image_path: str) -> Dict[str, Any]:
        """增强版OpenCV处理流程"""
        self.logger.info("启动增强版OCR处理流程")
        try:
            # 读取图像并校验
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("无法读取图像文件")

            # 多阶段预处理
            processed = self.multi_stage_preprocessing(img)
            
            # 基于连通域分析的元素定位
            elements = self.connected_component_analysis(processed)
            
            # 多特征融合识别
            results = self.multi_feature_recognition(img, elements)
            
            return {
                'success': True,
                'elements': results,
                'debug_info': {
                    'preprocessed': processed,
                    'elements_img': self.draw_elements(img.copy(), elements)
                }
            }
        except Exception as e:  # 添加缺失的异常处理
            self.logger.error(f"增强处理流程异常: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def multi_stage_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """多阶段图像预处理"""
        # 颜色空间转换
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 60, 60])
        upper_red = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 边缘保留滤波
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return cv2.bitwise_and(thresh, thresh, mask=mask)

    def connected_component_analysis(self, img: np.ndarray) -> List[Dict]:
        """连通域分析"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        
        elements = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if self._is_valid_element(w, h, area):
                elements.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'aspect_ratio': w/h
                })
        return elements

    def multi_feature_recognition(self, img: np.ndarray, elements: List[Dict]) -> List[Dict]:
        """多特征融合识别"""
        # 初始化ORB检测器
        orb = cv2.ORB_create()
        
        results = []
        for elem in elements:
            x, y, w, h = elem['x'], elem['y'], elem['width'], elem['height']
            roi = img[y:y+h, x:x+w]
            
            # 特征匹配流程
            match_result = self.orb_feature_match(roi, orb)
            if match_result:
                results.append({
                    'region': (x, y, w, h),
                    'result': match_result,
                    'method': 'feature'
                })
            else:
                # 备用OCR识别
                text = self._ocr_recognize(roi)
                if text:
                    results.append({
                        'region': (x, y, w, h),
                        'result': text,
                        'method': 'ocr'
                    })
        
        return results

    def orb_feature_match(self, roi: np.ndarray, orb: cv2.ORB) -> Optional[str]:
        """ORB特征匹配"""
        try:
            # 检测ROI特征
            kp1, des1 = orb.detectAndCompute(roi, None)
            if des1 is None:
                return None
            
            best_match = None
            best_score = 0
            
            # 遍历模板
            for name, template in self.templates.items():
                kp2, des2 = orb.detectAndCompute(template, None)
                if des2 is None:
                    continue
                
                # 特征匹配
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                # 计算匹配分数
                if len(matches) > 10:
                    score = sum([m.distance for m in matches]) / len(matches)
                    if score > best_score:
                        best_score = score
                        best_match = name
            
            return best_match if best_score > 30 else None  # 根据实际情况调整阈值
            
        except Exception as e:
            self.logger.error(f"特征匹配失败: {str(e)}")
            return None

    def draw_elements(self, img: np.ndarray, elements: List[Dict]) -> np.ndarray:
        """绘制检测元素"""
        for elem in elements:
            x, y, w, h = elem['x'], elem['y'], elem['width'], elem['height']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img

    def emergency_detection(self, image: np.ndarray) -> List[Dict]:
        """基于边缘特征的紧急检测"""
        # 确保输入为三通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                              minLineLength=50, maxLineGap=10)
        
        # 将直线转换为伪元素区域
        return [{
            'x': x1, 
            'y': y1, 
            'width': max(1, x2-x1),  # 防止0宽度
            'height': max(1, y2-y1)  # 防止0高度
        } for x1,y1,x2,y2 in lines.reshape(-1,4)] if lines is not None else []

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

def parse_tile_name(filename: str) -> dict:
    """解析牌名称，返回类型和数字"""
    if not filename.endswith('.png'):
        return {"type": "unknown", "number": 0, "display_name": "未知"}
    
    name = filename.replace('.png', '')
    
    # 定义所有牌类型的中文名称
    type_names = {
        # 数字牌
        "tong": "筒",
        "tiao": "条",
        "wan": "万",
        # 字牌
        "dong": "东",
        "nan": "南",
        "xi": "西",
        "bei": "北",
        "zhong": "中",
        "fa": "发",
        "bai": "白"
    }
    
    # 处理字牌（没有数字的情况）
    if name in type_names:
        return {
            "type": name,
            "number": "",
            "display_name": type_names[name]
        }
    
    # 处理数字牌（带下划线的情况）
    parts = name.split('_')
    if len(parts) == 2:
        tile_type = parts[0]
        number = parts[1]
        
        if tile_type in ["tong", "tiao", "wan"]:
            return {
                "type": tile_type,
                "number": number,
                "display_name": f"{number}{type_names[tile_type]}"
            }
    
    return {"type": "unknown", "number": "", "display_name": "未知"}

def calculate_orientation(x: int, y: int, image_width: int, image_height: int) -> str:
    """计算牌的位置方位，返回：上、下、左、右、中"""
    # 计算图片中心点
    center_x = image_width / 2
    center_y = image_height / 2
    
    # 计算元素中心点
    element_center_x = x + (image_width / 2)
    element_center_y = y + (image_height / 2)
    
    # 计算元素中心点到图片中心的距离
    dx = element_center_x - center_x
    dy = element_center_y - center_y
    
    # 设置阈值（图片尺寸的1/4）
    threshold = min(image_width, image_height) / 4
    
    # 判断方位
    if abs(dx) < threshold and abs(dy) < threshold:
        return "中"
    elif abs(dx) > abs(dy):  # 水平方向更明显
        return "左" if dx < 0 else "右"
    else:  # 垂直方向更明显
        return "上" if dy < 0 else "下"

def group_elements_by_orientation(elements: List[Dict]) -> Dict[str, List[Dict]]:
    """按照方位对元素进行分组"""
    grouped = {
        "上": [],
        "下": [],
        "左": [],
        "右": [],
        "中": []
    }
    
    for element in elements:
        orientation = element["orientation"]
        if orientation in grouped:
            grouped[orientation].append(element)
    
    # 移除空的分组
    return {k: v for k, v in grouped.items() if v}

def handle_ocr_test(game_ocr, config_manager):
    """单张OCR图片识别功能"""
    logger = get_logger(__name__)
    logger.info("启动OCR识别流程")

    try:
        # 从配置获取截图目录
        screenshot_dir = config_manager.get('adb.screenshot.local_temp_dir')
        
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

        # 读取图片尺寸
        img = cv2.imread(latest_img)
        if img is None:
            raise ValueError("无法读取图片")
        image_height, image_width = img.shape[:2]

        # 获取文件名
        filename = os.path.basename(latest_img)

        # 构建基础JSON结构
        json_result = {
            "image_info": {
                "path": filename,
                "processing_time": round(elapsed_time, 2),
                "elements_count": len(result['elements']) if result['success'] else 0,
                "image_size": {
                    "width": image_width,
                    "height": image_height
                }
            }
        }

        if result['success']:
            # 处理所有元素
            elements = []
            for idx, element in enumerate(result['elements'], 1):
                x, y, w, h = element['region']
                tile_info = parse_tile_name(element['result'])
                
                elements.append({
                    "id": idx,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "orientation": calculate_orientation(x, y, image_width, image_height),
                    "tile": {
                        "filename": element['result'],
                        "type": tile_info["type"],
                        "number": tile_info["number"],
                        "display_name": tile_info["display_name"]
                    }
                })
            
            # 按方位分组
            json_result["elements_by_orientation"] = group_elements_by_orientation(elements)
        else:
            json_result["error"] = result.get('error', '未知错误')

        # 生成结果文件路径
        json_path = os.path.splitext(latest_img)[0] + ".json"
        
        # 写入JSON文件
        try:
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)
            print(f"\n识别结果已保存至: {json_path}")
            logger.info(f"OCR结果已保存到 {json_path}")
            
            # 打印JSON结果
            print("\n识别结果:")
            print(json.dumps(json_result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"\n警告：结果文件保存失败 - {str(e)}")
            logger.error(f"结果文件保存失败: {str(e)}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        logger.error(f"OCR识别异常: {str(e)}", exc_info=True)

def analyze_template_images(template_dir: str) -> None:
    """分析模板图片尺寸"""
    logger = get_logger("ocr.template")
    logger.info(f"开始分析模板图片: {template_dir}")
    
    try:
        # 获取所有PNG文件
        template_files = [f for f in os.listdir(template_dir) if f.endswith('.png')]
        template_files.sort()
        
        print("\n模板图片分析结果:")
        print("-" * 50)
        print(f"{'文件名':<15} {'宽度':>8} {'高度':>8} {'比例':>8}")
        print("-" * 50)
        
        # 分析每个文件
        sizes = []
        for filename in template_files:
            filepath = os.path.join(template_dir, filename)
            img = cv2.imread(filepath)
            if img is not None:
                h, w = img.shape[:2]
                ratio = w / h
                sizes.append((w, h))
                print(f"{filename:<15} {w:>8} {h:>8} {ratio:>8.2f}")
        
        # 计算统计信息
        if sizes:
            avg_w = sum(w for w, _ in sizes) / len(sizes)
            avg_h = sum(h for _, h in sizes) / len(sizes)
            print("-" * 50)
            print(f"平均尺寸: {avg_w:.1f} x {avg_h:.1f}")
            print(f"文件数量: {len(sizes)}")
            
    except Exception as e:
        logger.error(f"模板分析失败: {str(e)}")
        print(f"分析过程出错: {str(e)}")

def main():
    """OCR测试主函数"""
    print("OCR模块测试")
    print("-" * 50)
    
    try:
        # 初始化配置管理器
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = os.path.join(base_dir, "config", "app_config.yaml")
        print(f"使用配置文件: {config_path}")
        
        config_manager = ConfigManager(config_path)
        
        # 分析模板图片
        template_dir = os.path.join(base_dir, "data", "tencent")
        analyze_template_images(template_dir)
        
        # 初始化OCR工具
        game_ocr = GameOCR(config_path)
        
        # 执行OCR测试
        handle_ocr_test(game_ocr, config_manager)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
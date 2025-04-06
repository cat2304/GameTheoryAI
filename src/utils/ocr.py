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
import time

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

class OCR:
    """OCR核心类"""
    def __init__(self, config_path: str = "config/app_config.yaml", debug_mode: bool = False):
        self.logger = get_logger("ocr.core")
        self.config = ConfigManager(config_path).get_config()
        self.templates = {}
        self.debug_mode = debug_mode
        
        if self.debug_mode:
            self.debug_dir = Path("debug")
            self.debug_dir.mkdir(exist_ok=True)
        
        # 预处理参数
        self.preprocess_params = {
            'blur_kernel': (5, 5),
            'threshold': 0,
            'max_value': 255,
            'threshold_type': cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            'morph_kernel': np.ones((3, 3), np.uint8),
            'morph_iterations': 1
        }
        
        # 检测参数
        self.detection_params = {
            'min_area': 100,
            'max_area': 10000,
            'min_aspect_ratio': 0.1,
            'max_aspect_ratio': 10.0
        }
        
        # OCR参数
        self.ocr_params = {
            'lang': 'chi_sim',
            'config': '--psm 6 --oem 3',
            'whitelist': '0123456789万条筒东南西北中发白'
        }
        
        # 名称映射
        self.name_map = {
            'tong': '筒',
            'tiao': '条',
            'w': '万',
            'dong': '东',
            'nan': '南',
            'xi': '西',
            'bei': '北',
            'zhong': '中',
            'fa': '发',
            'bai': '白'
        }
        
        # 数字映射
        self.number_map = {
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九'
        }
        
        self.logger.info("OCR初始化完成")

    def recognize_file(self, image_path: str) -> Dict:
        """识别图片文件"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            return self.recognize_image(image)
        except Exception as e:
            self.logger.error(f"识别图片文件失败: {e}", exc_info=True)
            return {"error": str(e)}

    def recognize_image(self, image: np.ndarray) -> Dict:
        """识别图片内容"""
        try:
            start_time = time.time()
            
            # 获取图像基本信息
            if isinstance(image, str):
                # 如果输入是路径，读取图像
                image_path = image
                image = cv2.imread(image)
                if image is None:
                    raise ValueError("无法读取图片")
            else:
                image_path = None
                
            height, width = image.shape[:2]
            
            # 预处理
            processed = self.preprocess_image(image)
            
            # 检测元素
            elements = self.detect_elements(processed)
            
            # 识别文字
            for element in elements:
                x1, y1, x2, y2 = element['box']
                roi = image[y1:y2, x1:x2]
                text = self.recognize_text(roi)
                element['name'] = text
            
            # 计算处理时间
            processing_time = round(time.time() - start_time, 2)
            
            return {
                "elements": elements,
                "timestamp": datetime.now().isoformat(),
                "image_info": {
                    "path": image_path,
                    "processing_time": processing_time,
                    "elements_count": len(elements),
                    "image_size": {
                        "width": width,
                        "height": height
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"识别图片内容失败: {e}", exc_info=True)
            return {"error": str(e)}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        try:
            # 确保输入始终为BGR三通道
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, self.preprocess_params['blur_kernel'], 0)
            
            # 自适应阈值
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 形态学操作
            kernel = self.preprocess_params['morph_kernel']
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 
                                    iterations=self.preprocess_params['morph_iterations'])
            
            # 保存预处理结果用于调试
            cv2.imwrite(str(self.debug_dir / "preprocessed.png"), binary)
            
            return binary
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}", exc_info=True)
            raise

    def detect_elements(self, image: np.ndarray) -> List[Dict]:
        """检测元素"""
        try:
            elements = []
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h != 0 else 0
                
                # 应用过滤条件
                if (area < self.detection_params['min_area'] or 
                    area > self.detection_params['max_area'] or
                    aspect_ratio < self.detection_params['min_aspect_ratio'] or
                    aspect_ratio > self.detection_params['max_aspect_ratio']):
                    continue
                
                elements.append({
                    "box": [x, y, x+w, y+h],
                    "area": area,
                    "center": [x + w//2, y + h//2]
                })
            
            # 保存检测结果用于调试
            debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for elem in elements:
                x1, y1, x2, y2 = elem["box"]
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(self.debug_dir / "detected_elements.png"), debug_image)
            
            self.logger.info(f"检测到 {len(elements)} 个元素")
            return elements
        except Exception as e:
            self.logger.error(f"检测元素失败: {e}", exc_info=True)
            return []

    def recognize_text(self, image: np.ndarray) -> str:
        """识别文字"""
        try:
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 增强预处理
            # 1. 直方图均衡化
            gray = cv2.equalizeHist(gray)
            
            # 2. 自适应阈值
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 3. 形态学操作
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 先尝试模板匹配
            text = self.template_match(binary)
            if text:
                return text
            
            # 如果模板匹配失败，尝试OCR识别
            text = pytesseract.image_to_string(
                binary,
                lang=self.ocr_params['lang'],
                config=self.ocr_params['config']
            )
            
            # 过滤结果
            text = ''.join(c for c in text.strip() if c in self.ocr_params['whitelist'])
            
            return text
        except Exception as e:
            self.logger.error(f"识别文字失败: {e}", exc_info=True)
            return ""

    def template_match(self, image: np.ndarray) -> str:
        """模板匹配"""
        try:
            # 如果没有模板，加载模板
            if not self.templates:
                self._load_templates()
            
            # 标准化尺寸
            image = cv2.resize(image, (48, 64))
            
            # 转换为灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 增强预处理
            # 1. 直方图均衡化
            image = cv2.equalizeHist(image)
            
            # 2. 自适应阈值
            binary = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 3. 形态学操作
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            best_match = None
            best_score = 0
            
            # 遍历所有模板进行匹配
            for name, template in self.templates.items():
                # 确保模板也是二值化的
                if len(template.shape) == 3:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # 对模板进行同样的预处理
                template = cv2.equalizeHist(template)
                template = cv2.adaptiveThreshold(
                    template, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)
                
                # 尝试不同的匹配方法
                methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                for method in methods:
                    result = cv2.matchTemplate(binary, template, method)
                    _, score, _, _ = cv2.minMaxLoc(result)
                    
                    if score > best_score:
                        best_score = score
                        best_match = name
                        
                        # 只在调试模式下保存匹配结果
                        if self.debug_mode and score > 0.3:
                            debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                            h, w = template.shape
                            cv2.rectangle(debug_image, (0, 0), (w, h), (0, 255, 0), 2)
                            cv2.putText(debug_image, f"{name}:{score:.2f}", (0, h+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.imwrite(str(self.debug_dir / f"match_{name}.png"), debug_image)
            
            # 降低匹配阈值
            threshold = 0.3
            if best_score > threshold:
                self.logger.debug(f"模板匹配成功: {best_match} (分数: {best_score:.2f})")
                # 转换为中文显示名称
                if best_match:
                    parts = best_match.split('_')
                    if len(parts) == 2:
                        tile_type = parts[0]
                        number = parts[1]
                        if tile_type in self.name_map:
                            if number in self.number_map:
                                return f"{self.number_map[number]}{self.name_map[tile_type]}"
                            return f"{number}{self.name_map[tile_type]}"
                    elif best_match in self.name_map:
                        return self.name_map[best_match]
                return best_match
            else:
                self.logger.debug(f"模板匹配失败: 最佳分数 {best_score:.2f} < {threshold}")
                return ""
            
        except Exception as e:
            self.logger.error(f"模板匹配失败: {e}", exc_info=True)
            return ""

    def _load_templates(self) -> None:
        """加载模板图片"""
        try:
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "tencent")
            if not os.path.exists(template_dir):
                self.logger.warning(f"模板目录不存在: {template_dir}")
                return
            
            for filename in os.listdir(template_dir):
                if not filename.endswith('.png'):
                    continue
                
                filepath = os.path.join(template_dir, filename)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    # 标准化尺寸
                    image = cv2.resize(image, (48, 64))
                    # 增强预处理
                    image = cv2.equalizeHist(image)
                    image = cv2.adaptiveThreshold(
                        image, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, 11, 2
                    )
                    kernel = np.ones((3,3), np.uint8)
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                    # 保存模板
                    name = filename.replace('.png', '')
                    self.templates[name] = image
                    # 只在调试模式下保存模板
                    if self.debug_mode:
                        cv2.imwrite(str(self.debug_dir / f"template_{name}.png"), image)
            
            self.logger.info(f"加载了 {len(self.templates)} 个模板")
            
        except Exception as e:
            self.logger.error(f"加载模板失败: {e}", exc_info=True)

def get_latest_image(screenshots_dir: str) -> str:
    """获取最新的截图文件"""
    try:
        # 确保目录存在
        if not os.path.exists(screenshots_dir):
            raise FileNotFoundError(f"截图目录不存在: {screenshots_dir}")
            
        # 获取所有PNG文件
        png_files = [f for f in os.listdir(screenshots_dir) if f.endswith('.png')]
        if not png_files:
            raise FileNotFoundError(f"截图目录中没有PNG文件: {screenshots_dir}")
            
        # 按修改时间排序
        png_files.sort(key=lambda x: os.path.getmtime(os.path.join(screenshots_dir, x)), reverse=True)
        
        # 返回最新的文件路径
        return os.path.join(screenshots_dir, png_files[0])
    except Exception as e:
        logging.error(f"获取最新截图失败: {e}", exc_info=True)
        raise

def handle_ocr_test():
    """处理OCR测试"""
    try:
        # 初始化OCR实例
        ocr = OCR()
        
        # 获取最新的截图文件
        screenshots_dir = "/Users/mac/ai/temp/screenshots/20250407"
        latest_img = get_latest_image(screenshots_dir)
        
        # 检查文件是否存在
        if not os.path.exists(latest_img):
            raise FileNotFoundError(f"图片文件不存在: {latest_img}")
            
        # 检查文件是否可读
        if not os.access(latest_img, os.R_OK):
            raise PermissionError(f"无法读取图片文件: {latest_img}")
        
        # 读取图片
        image = cv2.imread(latest_img)
        if image is None:
            raise ValueError(f"无法读取图片文件，可能文件已损坏: {latest_img}")
            
        # 获取图片尺寸
        height, width = image.shape[:2]
        
        # 执行OCR识别
        start_time = time.time()
        result = ocr.recognize_file(latest_img)
        processing_time = round(time.time() - start_time, 2)
        
        # 按位置分组
        tiles_by_position = group_tiles_by_position(result['elements'], width, height)
        
        # 构建输出JSON
        output = {
            "table_status": {
                "center_display": None,
                "timer": None,
                "tiles_by_position": tiles_by_position
            },
            "image_info": {
                "path": os.path.basename(latest_img),
                "processing_time": processing_time,
                "elements_count": len(result['elements']),
                "image_size": {
                    "width": width,
                    "height": height
                }
            }
        }
        
        # 保存结果
        output_path = latest_img.rsplit('.', 1)[0] + '.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        print(f"识别结果已保存到: {output_path}")
        
    except FileNotFoundError as e:
        logging.error(f"文件不存在: {e}")
        raise
    except PermissionError as e:
        logging.error(f"权限错误: {e}")
        raise
    except ValueError as e:
        logging.error(f"图片读取错误: {e}")
        raise
    except Exception as e:
        logging.error(f"OCR测试失败: {e}", exc_info=True)
        raise

def group_tiles_by_position(elements, width, height):
    """按位置分组元素"""
    result = {
        "bottom": [],
        "right": [],
        "top": [],
        "left": [],
        "center": []
    }
    
    for element in elements:
        if 'box' not in element or 'name' not in element:
            continue
            
        x = (element['box'][0] + element['box'][2]) / 2
        y = (element['box'][1] + element['box'][3]) / 2
        
        # 计算相对位置
        rel_x = x / width
        rel_y = y / height
        
        # 根据相对位置分类
        if rel_y > 0.7:
            result["bottom"].append(element["name"])
        elif rel_x > 0.7:
            result["right"].append(element["name"])
        elif rel_y < 0.3:
            result["top"].append(element["name"])
        elif rel_x < 0.3:
            result["left"].append(element["name"])
        else:
            result["center"].append(element["name"])
    
    return result

if __name__ == "__main__":
    handle_ocr_test() 
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
        
        # 初始化debug目录
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
            'whitelist': '0123456789万条东南西北中发白'
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
            
            # 只在调试模式下保存预处理结果
            if self.debug_mode:
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
            
            # 只在调试模式下保存检测结果
            if self.debug_mode:
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

    def batch_recognize_images(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """批量识别目录中的所有图片
        
        Args:
            input_dir: 输入图片目录路径
            output_dir: 输出结果目录路径，如果为None则使用输入目录
            
        Returns:
            Dict[str, Dict]: 包含所有图片识别结果的字典，key为图片文件名，value为识别结果
        """
        if output_dir is None:
            output_dir = input_dir
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            
        if not image_files:
            self.logger.warning(f"在目录 {input_dir} 中没有找到图片文件")
            return {}
            
        self.logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 处理每张图片
        results = []
        summary_path = os.path.join(output_dir, "recognition_summary.json")
        
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
                    f.write('\n')  # 添加换行符
                
                results.append(recognition_data)
                
            except Exception as e:
                self.logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
                continue
                
        self.logger.info(f"批量识别完成，共处理 {len(image_files)} 张图片")
        self.logger.info(f"汇总结果已保存到: {summary_path}")
        
        return results

def get_latest_image(screenshots_dir: str) -> Optional[str]:
    """获取最新的截图文件
    
    Args:
        screenshots_dir: 截图目录路径
        
    Returns:
        Optional[str]: 最新的截图文件路径，如果没有找到则返回None
    """
    try:
        # 获取所有PNG文件
        png_files = glob.glob(os.path.join(screenshots_dir, "*.png"))
        if not png_files:
            logging.warning(f"截图目录中没有PNG文件: {screenshots_dir}")
            return None
            
        # 按修改时间排序
        latest_file = max(png_files, key=os.path.getmtime)
        return latest_file
        
    except Exception as e:
        logging.error(f"获取最新截图失败: {e}")
        return None

def handle_ocr_test():
    """处理OCR测试"""
    # 获取配置
    config_manager = ConfigManager("config/app_config.yaml")
    config = config_manager.get_config()
    screenshots_dir = config.get("screenshots_dir", "/Users/mac/ai/temp/screenshots/20250407")
    
    # 创建OCR实例
    ocr = OCR(debug_mode=True)
    
    while True:
        print("\n=== OCR测试菜单 ===")
        print("1. 识别最新一张图片")
        print("2. 批量识别所有图片")
        print("0. 退出")
        
        choice = input("请选择操作 (0-2): ")
        
        if choice == "1":
            # 获取最新的截图
            latest_image = get_latest_image(screenshots_dir)
            if not latest_image:
                print("没有找到截图文件")
                continue
                
            print(f"\n正在识别最新图片: {latest_image}")
            # 识别图片
            result = ocr.recognize_file(latest_image)
            
            # 构建识别结果
            recognition_data = {
                "filename": os.path.basename(latest_image),
                "timestamp": datetime.now().isoformat(),
                "result": result
            }
            
            # 将结果追加到文件（每个结果占一行）
            summary_path = os.path.join(screenshots_dir, "recognition_summary.json")
            with open(summary_path, 'a', encoding='utf-8') as f:
                json.dump(recognition_data, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')  # 添加换行符
                
            print(f"识别结果已保存到: {summary_path}")
            
        elif choice == "2":
            print("\n开始批量识别测试...")
            batch_results = ocr.batch_recognize_images(screenshots_dir)
            print(f"批量识别完成，共处理 {len(batch_results)} 张图片")
            print(f"汇总结果已保存到: {os.path.join(screenshots_dir, 'recognition_summary.json')}")
            
        elif choice == "0":
            print("退出程序")
            break
            
        else:
            print("无效的选择，请重新输入")

def save_recognition_result(image_path, result, summary_file):
    """保存识别结果到JSON文件"""
    try:
        # 准备要保存的数据
        data = {
            "filename": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "result": {
                "elements": result["elements"],
                "image_info": {
                    "path": result["image_info"]["path"],
                    "processing_time": result["image_info"]["processing_time"],
                    "elements_count": result["image_info"]["elements_count"],
                    "image_size": result["image_info"]["image_size"]
                }
            }
        }
        
        # 将结果追加到文件
        with open(summary_file, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')  # 添加换行符，使每个结果占一行
            
        logger.info(f"识别结果已保存到: {summary_file}")
    except Exception as e:
        logger.error(f"保存识别结果失败: {str(e)}")
        raise

if __name__ == "__main__":
    handle_ocr_test() 
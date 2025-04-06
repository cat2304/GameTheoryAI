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




        #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR测试脚本
用于测试OCR模块的功能
"""

import os
import sys
from pathlib import Path

# 导入必要模块（直接导入，避免通过__init__.py导入）
from src.utils.ocr import GameOCR, handle_ocr_test
from src.utils.utils import ConfigManager

def main():
    """OCR测试主函数"""
    print("OCR模块测试")
    print("-" * 50)
    
    try:
        # 初始化配置管理器
        config_path = os.path.join(Path(__file__).parent, "config", "app_config.yaml")
        config_manager = ConfigManager(config_path)
        
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

    #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建测试用的截图文件
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def create_test_image(output_dir=None):
    """创建测试用的截图文件"""
    # 设置默认输出目录
    if output_dir is None:
        # 使用配置中的目录结构
        temp_dir = os.path.join(Path(__file__).parent, "data")
        output_dir = os.path.join(temp_dir, "screenshots")
    
    # 创建日期子目录
    date_str = datetime.now().strftime('%Y%m%d')
    target_dir = os.path.join(output_dir, date_str)
    os.makedirs(target_dir, exist_ok=True)
    
    # 创建测试图片
    # 创建一个空白图片
    width, height = 800, 600
    test_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 在底部添加一些模拟的游戏元素
    y_start = height - 150  # 底部区域开始位置
    
    # 添加一些彩色方块，模拟游戏元素
    elements = [
        {"text": "元素1", "color": (255, 0, 0)},    # 红色
        {"text": "元素2", "color": (0, 255, 0)},    # 绿色
        {"text": "元素3", "color": (0, 0, 255)},    # 蓝色
        {"text": "元素4", "color": (255, 255, 0)},  # 黄色
        {"text": "元素5", "color": (255, 0, 255)},  # 紫色
        {"text": "元素6", "color": (0, 255, 255)},  # 青色
        {"text": "元素7", "color": (128, 0, 0)},    # 深红色
        {"text": "元素8", "color": (0, 128, 0)},    # 深绿色
        {"text": "元素9", "color": (0, 0, 128)},    # 深蓝色
        {"text": "特殊元素A", "color": (128, 128, 0)},  # 橄榄色
        {"text": "特殊元素B", "color": (128, 0, 128)},  # 深紫色
        {"text": "特殊元素C", "color": (0, 128, 128)}   # 深青色
    ]
    
    # 计算每个元素的宽度
    element_width = width // len(elements)
    
    # 绘制元素
    for i, element in enumerate(elements):
        # 计算元素位置
        x1 = i * element_width
        x2 = (i + 1) * element_width
        y1 = y_start
        y2 = height
        
        # 绘制有颜色的矩形
        cv2.rectangle(test_image, (x1, y1), (x2, y2), element["color"], -1)
        
        # 绘制文字
        cv2.putText(
            test_image, 
            element["text"], 
            (x1 + 5, y1 + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0),  # 黑色文字
            1
        )
    
    # 在顶部添加一些说明文字
    cv2.putText(
        test_image,
        "测试用OCR识别图片",
        (width // 4, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    
    # 保存图片
    filename = f"1.png"  # 使用数字作为文件名，模拟截图管理器的行为
    filepath = os.path.join(target_dir, filename)
    cv2.imwrite(filepath, test_image)
    
    print(f"测试图片已保存至: {filepath}")
    return filepath

if __name__ == "__main__":
    create_test_image()
    print("测试图片创建完成！") 
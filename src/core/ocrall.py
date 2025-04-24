import os
import cv2
import json
import logging
import numpy as np
import sys
from paddleocr import PaddleOCR
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============ 配置常量 ============
DEBUG_DIR = "data/debug"
RESULT_IMG = os.path.join(DEBUG_DIR, "full_ocr_result.png")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============ 登录状态关键词（按优先级排序） ============
LOGIN_KEYWORDS = ["重新登录", "登录", "游戏大厅"]

# ============ OCR 配置 ============
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="ch",
    det_db_thresh=0.1,
    det_db_box_thresh=0.1,
    det_db_unclip_ratio=2.0,
    det_limit_side_len=2000,
    show_log=False
)

def find_element_position(result: List, target_text: str) -> Optional[Dict]:
    """查找指定文本的位置"""
    if not result or not result[0]:
        return None
        
    for box, (text, conf) in result[0]:
        text = text.strip()
        if text == target_text and conf > 0.5:
            # 计算中心点坐标
            points = np.array(box)
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))
            
            position = {
                "gameStatus": text,
                "position": {
                    "x": center_x,
                    "y": center_y
                }
            }
            logger.info(f"找到元素: {text} 位置: ({center_x}, {center_y})")
            return position
            
    return None

def analyze_screen(img: np.ndarray) -> Dict:
    """分析游戏屏幕内容"""
    if img is None:
        logger.error("无法读取图像")
        return {
            "success": False,
            "error": "无法读取图像"
        }

    try:
        # 第一步：OCR识别
        logger.info("开始OCR识别...")
        result = ocr.ocr(img, cls=False)
        
        # 第二步：按优先级查找元素
        for keyword in LOGIN_KEYWORDS:
            element = find_element_position(result, keyword)
            if element:
                # 一旦找到元素就返回结果
                return {
                    "success": True,
                    **element
                }
        
        # 如果没有找到任何匹配的元素
        return {
            "success": False,
            "error": "未找到匹配的游戏状态"
        }
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def analyze_screenshot(image_path: str) -> Dict:
    """分析游戏截图"""
    logger.info(f"开始分析截图: {image_path}")
    
    # 第一步：读取图像
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"无法读取图像: {image_path}")
        return {
            "success": False,
            "error": f"无法读取图像: {image_path}"
        }
    
    # 第二步：分析图像
    return analyze_screen(img)

class ScreenCapture:
    def __init__(self, screenshot_dir: str = "data/screenshots"):
        self.screenshot_dir = screenshot_dir

    def capture(self) -> Tuple[bool, str]:
        """获取屏幕截图"""
        # 执行截图命令

    def capture_region(self, x: int, y: int, width: int, height: int) -> Tuple[bool, str]:
        """获取指定区域截图"""
        # 执行区域截图命令

    def save_screenshot(self, image_path: str) -> bool:
        """保存截图到指定路径"""
        # 保存截图文件

class OCRProcessor:
    def __init__(self):
        # 初始化OCR模型
        pass

    def recognize(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """OCR识别"""
        # 执行文字识别

    def recognize_region(self, image_path: str, region: Tuple[int, int, int, int]) -> Tuple[bool, Dict[str, Any]]:
        """识别指定区域"""
        # 执行区域识别

    def recognize_text(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """识别文字"""
        # 执行文字识别

    def recognize_numbers(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """识别数字"""
        # 执行数字识别

class ImageProcessor:
    def __init__(self):
        pass

    def preprocess(self, image_path: str) -> Tuple[bool, str]:
        """图像预处理"""
        # 执行预处理

    def find_template(self, image_path: str, template_path: str) -> Tuple[bool, Tuple[int, int]]:
        """模板匹配"""
        # 执行模板匹配

    def find_color(self, image_path: str, color: Tuple[int, int, int]) -> Tuple[bool, List[Tuple[int, int]]]:
        """颜色识别"""
        # 执行颜色识别

class Utils:
    @staticmethod
    def ensure_dir(directory: str) -> None:
        """确保目录存在"""
        # 创建目录

    @staticmethod
    def get_timestamp() -> int:
        """获取时间戳"""
        # 返回时间戳

    @staticmethod
    def format_path(path: str) -> str:
        """格式化路径"""
        # 处理路径

class Config:
    def __init__(self):
        self.adb_path = "adb"
        self.screenshot_dir = "data/screenshots"
        self.ocr_model_path = "models/ocr"
        self.template_dir = "templates"

    def load_config(self, config_path: str) -> bool:
        """加载配置"""
        # 加载配置文件

    def save_config(self, config_path: str) -> bool:
        """保存配置"""
        # 保存配置文件

if __name__ == "__main__":
    # 测试代码
    test_image = "data/screenshots/latest.png"
    if os.path.exists(test_image):
        # 第一步：分析截图
        result = analyze_screenshot(test_image)
        print("分析结果：", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.error(f"测试图片不存在: {test_image}") 
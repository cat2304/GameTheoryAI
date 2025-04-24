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
        self.ocr = ocr
        self.logger = logging.getLogger(__name__)

    def recognize_all_text(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """识别图片中的所有文字
        
        Args:
            image_path: 图片路径
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否成功, 识别结果)
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
            
            # 执行OCR识别
            result = self.ocr.ocr(img, cls=False)
            
            # 处理识别结果
            if not result or not result[0]:
                return False, {"error": "未识别到任何文字"}
            
            # 提取所有文字及其位置信息
            texts = []
            for box, (text, conf) in result[0]:
                # 计算文本框的中心点
                points = np.array(box)
                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))
                
                # 计算文本框的宽度和高度
                width = int(np.max(points[:, 0]) - np.min(points[:, 0]))
                height = int(np.max(points[:, 1]) - np.min(points[:, 1]))
                
                # 将 numpy 数组转换为 Python 列表
                box_points = [[int(x), int(y)] for x, y in box]
                
                texts.append({
                    "text": text.strip(),
                    "confidence": float(conf),
                    "position": {
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height
                    },
                    "box": box_points  # 使用转换后的列表
                })
            
            # 在图片上标注识别结果
            vis_img = img.copy()
            for text_info in texts:
                box = np.array(text_info["box"])
                cv2.polylines(vis_img, [box], True, (0, 255, 0), 2)
                cv2.putText(vis_img, text_info["text"], 
                          (text_info["position"]["x"], text_info["position"]["y"]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 保存标注后的图片
            cv2.imwrite(RESULT_IMG, vis_img)
            
            return True, {
                "success": True,
                "texts": texts,
                "result_image": RESULT_IMG
            }
            
        except Exception as e:
            self.logger.error(f"OCR识别失败: {str(e)}")
            return False, {"error": str(e)}

    def recognize(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """OCR识别（兼容旧接口）"""
        return self.recognize_all_text(image_path)

    def recognize_region(self, image_path: str, region: Tuple[int, int, int, int]) -> Tuple[bool, Dict[str, Any]]:
        """识别指定区域"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
            
            # 裁剪指定区域
            x, y, w, h = region
            roi = img[y:y+h, x:x+w]
            
            # 执行OCR识别
            result = self.ocr.ocr(roi, cls=False)
            
            # 处理识别结果
            if not result or not result[0]:
                return False, {"error": "未识别到任何文字"}
            
            # 提取文字及其位置信息
            texts = []
            for box, (text, conf) in result[0]:
                # 调整坐标到原图坐标系
                adjusted_box = np.array(box) + np.array([x, y])
                center_x = int(np.mean(adjusted_box[:, 0]))
                center_y = int(np.mean(adjusted_box[:, 1]))
                
                texts.append({
                    "text": text.strip(),
                    "confidence": float(conf),
                    "position": {
                        "x": center_x,
                        "y": center_y
                    }
                })
            
            return True, {
                "success": True,
                "texts": texts
            }
            
        except Exception as e:
            self.logger.error(f"区域OCR识别失败: {str(e)}")
            return False, {"error": str(e)}

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
        # 创建OCR处理器
        processor = OCRProcessor()
        
        # 测试全图识别
        success, result = processor.recognize_all_text(test_image)
        if success:
            print("全图识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("全图识别失败：", result.get("error", "未知错误"))
            
        # 测试区域识别
        success, result = processor.recognize_region(test_image, (100, 100, 200, 200))
        if success:
            print("区域识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("区域识别失败：", result.get("error", "未知错误"))
    else:
        logger.error(f"测试图片不存在: {test_image}") 
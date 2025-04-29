import os
import sys
import cv2
import json
import logging
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.ocr_color import ColorProcessor

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# OCR配置
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="ch",
    det_db_thresh=0.1,
    det_db_box_thresh=0.1,
    det_db_unclip_ratio=2.0,
    det_limit_side_len=2000,
    show_log=False
)

def recognize_cards(image_path: str) -> Dict[str, Any]:
    """识别图片中的扑克牌"""
    processor = OCRProcessor()
    success, result = processor.recognize_all_text(image_path)
    
    if not success:
        return {
            "success": False,
            "error": result.get("error", "未知错误")
        }
    
    # 提取识别到的卡牌信息
    cards = []
    
    for text_info in result["texts"]:
        card_info = {
            "text": text_info["text"].upper(),
            "confidence": text_info["confidence"],
            "position": text_info["position"],
            "color": text_info["color"]
        }
        cards.append(card_info)
    
    return {
        "success": True,
        "data": {
            "cards": cards,
            "image_path": image_path
        }
    }

class OCRProcessor:
    def __init__(self):
        self.ocr = ocr
        self.logger = logging.getLogger(__name__)
        self.color_processor = ColorProcessor()
        # 扑克牌数字的正则表达式
        self.card_patterns = {
            'numbers': ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'a', 'j', 'q', 'k']
        }

    def _is_poker_card_text(self, text: str) -> bool:
        """判断文本是否为扑克牌信息"""
        text = text.strip()
        # 移除所有空格
        text = text.replace(' ', '')
        
        # 检查是否完全匹配扑克牌数字
        return text in self.card_patterns['numbers']

    def _calculate_text_box(self, points: List[List[int]]) -> Dict[str, int]:
        """计算文本框的位置和大小"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }

    def _process_ocr_result(self, result: List[List[Any]]) -> List[Dict[str, Any]]:
        """处理OCR识别结果"""
        texts = []
        for line in result:
            if not line:
                continue
            text = line[1][0]  # 文本内容
            confidence = float(line[1][1])  # 置信度
            points = line[0]  # 位置信息
            
            # 只处理扑克牌文本
            if not self._is_poker_card_text(text):
                continue
            
            # 计算文本框位置和大小
            position = self._calculate_text_box(points)
            
            # 获取文本颜色
            x, y = int(position["x"]), int(position["y"])
            w, h = int(position["width"]), int(position["height"])
            roi = self.current_image[y:y+h, x:x+w]
            if roi.size > 0:
                color = self._get_text_color(roi)
            else:
                color = {"r": 0, "g": 0, "b": 0, "hex": "#000000"}
            
            texts.append({
                "text": text.upper(),  # 统一转换为大写
                "confidence": confidence,
                "position": position,
                "color": color
            })
        return texts

    def _get_text_color(self, roi: np.ndarray) -> Dict[str, Any]:
        """获取文本颜色"""
        if roi.size == 0:
            return {"r": 0, "g": 0, "b": 0, "hex": "#000000"}
            
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 找到非零像素
        non_zero = cv2.findNonZero(binary)
        if non_zero is None:
            return {"r": 0, "g": 0, "b": 0, "hex": "#000000"}
            
        # 计算非零像素的平均颜色
        mean_color = cv2.mean(roi, mask=binary)[:3]
        
        # 转换为整数
        b, g, r = map(int, mean_color)
        
        # 转换为十六进制
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        return {
            "r": r,
            "g": g,
            "b": b,
            "hex": hex_color
        }

    def recognize_all_text(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """识别图片中的所有文字"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"图片不存在: {image_path}")
                return False, {"error": f"图片不存在: {image_path}"}

            # 读取图片
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                self.logger.error(f"无法读取图片: {image_path}")
                return False, {"error": f"无法读取图片: {image_path}"}
            
            # OCR识别
            result = self.ocr.ocr(self.current_image, cls=False)
            if not result or not result[0]:
                self.logger.warning("未识别到任何文字")
                return False, {"error": "未识别到任何文字"}
            
            # 处理识别结果
            texts = self._process_ocr_result(result[0])
            
            # 记录识别结果
            self.logger.info(f"识别到 {len(texts)} 个文本")
            for text in texts:
                self.logger.info(f"文本: {text['text']}, 置信度: {text['confidence']:.4f}, 位置: {text['position']}, 颜色: {text['color']['hex']}")
            
            return True, {
                "success": True,
                "texts": texts
            }
            
        except Exception as e:
            self.logger.error(f"OCR识别失败: {str(e)}")
            return False, {"error": str(e)}

    def recognize(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """OCR识别（兼容旧接口）"""
        return self.recognize_all_text(image_path)

if __name__ == "__main__":
    test_image = "data/screenshots/hand/2.png"
    
    # 测试recognize_all_text
    processor = OCRProcessor()
    success, result = processor.recognize_all_text(test_image)
    if success:
        print("\n全图识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n全图识别失败：", result.get("error", "未知错误"))
        
    # 测试recognize_cards
    card_result = recognize_cards(test_image)
    print("\n卡牌识别结果：", json.dumps(card_result, ensure_ascii=False, indent=2)) 
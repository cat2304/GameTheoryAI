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
    """识别图片中的扑克牌（兼容旧接口）"""
    processor = OCRProcessor()
    success, result = processor.recognize_all_text(image_path)
    
    if not success:
        return {
            "success": False,
            "error": result.get("error", "未知错误")
        }
    
    # 提取识别到的卡牌文本
    cards = [text["text"].upper() for text in result["texts"]]
    
    return {
        "success": True,
        "hand_cards": cards,  # 暂时将所有识别到的卡牌放在手牌中
        "public_cards": []    # 公共牌暂时返回空列表
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

    def _calculate_text_box(self, points: np.ndarray) -> Dict[str, int]:
        """计算文本框的位置和大小"""
        return {
            "x": int(np.mean(points[:, 0])),
            "y": int(np.mean(points[:, 1])),
            "width": int(np.max(points[:, 0]) - np.min(points[:, 0])),
            "height": int(np.max(points[:, 1]) - np.min(points[:, 1]))
        }

    def _process_ocr_result(self, result: List, image_path: str) -> List[Dict[str, Any]]:
        """处理OCR识别结果"""
        texts = []
        for box, (text, conf) in result[0]:
            # 只处理扑克牌相关的文本
            if not self._is_poker_card_text(text):
                continue
                
            points = np.array(box)
            position = self._calculate_text_box(points)
            
            # 获取区域颜色
            _, color_result = self.color_processor.get_region_color(image_path, position)
            
            texts.append({
                "text": text.strip(),
                "confidence": float(conf),
                "position": position,
                "color": color_result.get("color", {}) if color_result.get("success") else {}
            })
        return texts

    def recognize_all_text(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """识别图片中的所有文字"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
            
            result = self.ocr.ocr(img, cls=False)
            if not result or not result[0]:
                return False, {"error": "未识别到任何文字"}
            
            texts = self._process_ocr_result(result, image_path)
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
    test_image = "data/templates/test.png"
    processor = OCRProcessor()
    success, result = processor.recognize_all_text(test_image)
    if success:
        print("全图识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("全图识别失败：", result.get("error", "未知错误")) 
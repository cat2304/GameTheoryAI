import os
import cv2
import json
import logging
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, List, Tuple, Optional, Any

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
                
                texts.append({
                    "text": text.strip(),
                    "confidence": float(conf),
                    "position": {
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height
                    }
                })
            
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
    else:
        logger.error(f"测试图片不存在: {test_image}") 
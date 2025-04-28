import os
import cv2
import json
import logging
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, List, Tuple, Optional, Any
from src.core.enums import RegionType

# ============ 配置常量 ============
DEBUG_DIR = "data/debug"
RESULT_IMG = os.path.join(DEBUG_DIR, "region_ocr_result.png")
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

class OCRRegionProcessor:
    def __init__(self):
        self.ocr = ocr
        self.logger = logging.getLogger(__name__)

    def get_image_size(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """获取图片尺寸"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
            
            height, width = img.shape[:2]
            return True, {
                "success": True,
                "width": width,
                "height": height
            }
        except Exception as e:
            self.logger.error(f"获取图片尺寸失败: {str(e)}")
            return False, {"error": str(e)}

    def recognize_region(self, image_path: str, region: Tuple[int, int, int, int], type: RegionType = RegionType.PUBLIC) -> Tuple[bool, Dict[str, Any]]:
        """识别指定区域的文字"""
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
                
                # 计算文本框的宽度和高度
                width = int(np.max(adjusted_box[:, 0]) - np.min(adjusted_box[:, 0]))
                height = int(np.max(adjusted_box[:, 1]) - np.min(adjusted_box[:, 1]))
                
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
                "texts": texts,
                "type": type.value
            }
            
        except Exception as e:
            self.logger.error(f"区域OCR识别失败: {str(e)}")
            return False, {"error": str(e)}

if __name__ == "__main__":
    # 测试代码
    test_image = "data/screenshots/latest.png"
    if os.path.exists(test_image):
        # 创建OCR处理器
        processor = OCRRegionProcessor()
        
        # 获取图片尺寸
        success, size_info = processor.get_image_size(test_image)
        if success:
            print("图片尺寸信息：", json.dumps(size_info, ensure_ascii=False, indent=2))
            
            # 测试区域识别
            success, result = processor.recognize_region(test_image, (100, 100, 200, 200))
            if success:
                print("区域识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("区域识别失败：", result.get("error", "未知错误"))
        else:
            print("获取图片尺寸失败：", size_info.get("error", "未知错误"))
    else:
        logger.error(f"测试图片不存在: {test_image}") 
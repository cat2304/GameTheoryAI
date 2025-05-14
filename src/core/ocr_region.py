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

    def _calculate_text_box(self, points: np.ndarray) -> Dict[str, int]:
        """计算文本框的位置和大小"""
        return {
            "x": int(np.mean(points[:, 0])),
            "y": int(np.mean(points[:, 1])),
            "width": int(np.max(points[:, 0]) - np.min(points[:, 0])),
            "height": int(np.max(points[:, 1]) - np.min(points[:, 1]))
        }

    def _process_ocr_result(self, result: List) -> List[Dict[str, Any]]:
        """处理OCR识别结果"""
        texts = []
        for box, (text, conf) in result[0]:
            points = np.array(box)
            texts.append({
                "text": text.strip(),
                "confidence": float(conf),
                "position": self._calculate_text_box(points)
            })
        return texts

    def _validate_input(self, image_path: str, region: Tuple[int, int, int, int], region_type: int) -> Tuple[bool, str]:
        """验证输入参数的有效性
        
        Args:
            image_path: 图片路径
            region: 区域坐标 (x, y, width, height)
            region_type: 区域类型
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not isinstance(region, tuple) or len(region) != 4:
            return False, "区域坐标必须是包含4个元素的元组"
            
        x, y, w, h = region
        if not all(isinstance(v, int) and v >= 0 for v in [x, y, w, h]):
            return False, "区域坐标必须是非负整数"
            
        if not isinstance(region_type, int) or region_type not in [1, 2, 3]:
            return False, "区域类型必须是1(公牌区域)、2(手牌区域)或3(操作区域)"
            
        return True, ""

    def _check_region_bounds(self, img: np.ndarray, region: Tuple[int, int, int, int]) -> Tuple[bool, str]:
        """检查区域是否超出图片边界
        
        Args:
            img: 图片数组
            region: 区域坐标
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        height, width = img.shape[:2]
        x, y, w, h = region
        
        if x + w > width or y + h > height:
            return False, f"区域超出图片边界 (图片大小: {width}x{height}, 区域: {region})"
            
        if w < 10 or h < 10:
            return False, f"区域太小 (最小 10x10, 当前: {w}x{h})"
            
        return True, ""

    def _post_process_results(self, texts: List[Dict[str, Any]], region_type: int) -> List[Dict[str, Any]]:
        """根据区域类型对识别结果进行后处理
        
        Args:
            texts: OCR识别结果
            region_type: 区域类型
            
        Returns:
            List[Dict[str, Any]]: 处理后的结果
        """
        processed = []
        for text in texts:
            # 移除空白文本
            if not text["text"].strip():
                continue
                
            # 根据区域类型进行特殊处理
            if region_type == RegionType.PUBLIC.value:  # 公牌区域
                # 移除置信度过低的结果
                if text["confidence"] < 0.6:
                    continue
            elif region_type == RegionType.HAND.value:  # 手牌区域
                # 对于手牌区域，我们只保留数字和特定字符
                if not any(char.isdigit() or char in "东南西北中发白" for char in text["text"]):
                    continue
            elif region_type == RegionType.ACTION.value:  # 操作区域
                # 对于操作区域，移除过小的文本框
                if text["position"]["width"] < 20 or text["position"]["height"] < 20:
                    continue
                    
            processed.append(text)
            
        return processed

    def recognize_region(self, image_path: str, region: Tuple[int, int, int, int], region_type: int) -> Tuple[bool, Dict[str, Any]]:
        """识别指定区域的文字
        
        Args:
            image_path: 图片路径
            region: 区域坐标 (x, y, width, height)
            region_type: 区域类型 1=公牌区域, 2=手牌区域, 3=操作区域
            
        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - 第一个元素表示是否成功
                - 第二个元素包含识别结果或错误信息
        """
        try:
            # 验证输入参数
            valid, error_msg = self._validate_input(image_path, region, region_type)
            if not valid:
                return False, {"error": error_msg}

            if not os.path.exists(image_path):
                return False, {"error": f"图片不存在: {image_path}"}

            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
                
            # 检查区域边界
            valid, error_msg = self._check_region_bounds(img, region)
            if not valid:
                return False, {"error": error_msg}
            
            # 裁剪指定区域
            x, y, w, h = region
            roi = img[y:y+h, x:x+w]
            
            # OCR识别
            result = self.ocr.ocr(roi, cls=False)
            if not result or not result[0]:
                return False, {"error": "未识别到任何文字"}
            
            # 处理识别结果
            texts = self._process_ocr_result(result)
            
            # 调整坐标（相对于原图）
            for text in texts:
                text["position"]["x"] += x
                text["position"]["y"] += y
            
            # 后处理结果
            texts = self._post_process_results(texts, region_type)
            
            if not texts:
                return False, {"error": "后处理后没有有效的识别结果"}
            
            return True, {
                "success": True,
                "texts": texts,
                "type": region_type
            }
            
        except Exception as e:
            self.logger.error(f"区域OCR识别失败: {str(e)}")
            return False, {"error": str(e)}

if __name__ == "__main__":
    test_image = "data/screenshots/public/1.png"
    processor = OCRRegionProcessor()
    
    # 测试区域识别
    region = (100, 100, 200, 200)  # x, y, width, height
    success, result = processor.recognize_region(test_image, region, 1)
    
    if success:
        print("区域识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("区域识别失败：", result.get("error", "未知错误")) 
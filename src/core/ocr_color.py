import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ColorProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_region_color(self, image_path: str, region: Dict[str, int]) -> Tuple[bool, Dict[str, Any]]:
        """获取指定区域的颜色"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
            
            x, y = region["x"], region["y"]
            w, h = region["width"], region["height"]
            
            roi = img[y:y+h, x:x+w]
            b, g, r = map(int, np.mean(roi, axis=(0, 1)))
            
            return True, {
                "success": True,
                "color": {
                    "b": b,
                    "g": g,
                    "r": r,
                    "hex": f"#{r:02x}{g:02x}{b:02x}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取颜色失败: {str(e)}")
            return False, {"error": str(e)}

if __name__ == "__main__":
    test_image = "data/screenshots/public/1.png"
    processor = ColorProcessor()
    
    # 测试区域1
    success, result = processor.get_region_color(
        test_image,
        {"x": 151, "y": 97, "width": 49, "height": 40}
    )
    print("颜色识别结果：", result)

    # 测试区域2
    success, result = processor.get_region_color(
        test_image,
        {"x": 65, "y": 98, "width": 20, "height": 22}
    )
    print("颜色识别结果：", result)

    # 测试区域3
    success, result = processor.get_region_color(
        test_image,
        {"x": 234, "y": 97, "width": 19, "height": 24}
    )
    print("颜色识别结果：", result)

    # 测试区域4
    success, result = processor.get_region_color(
        test_image,
        {"x": 318, "y": 96, "width": 16, "height": 18}
    )
    print("颜色识别结果：", result)


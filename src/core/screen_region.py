import os
import sys
import cv2
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.screen_all import ScreenCapture
from src.core.enums import RegionType

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ScreenRegionProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.screen_capture = ScreenCapture()

    def validate_region(self, x: int, y: int, w: int, h: int, img_shape: Tuple[int, int, int]) -> bool:
        """验证区域坐标是否有效"""
        img_h, img_w = img_shape[:2]
        return (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= img_w and y + h <= img_h)

    def check_image_quality(self, img: np.ndarray) -> bool:
        """检查图片质量"""
        if img is None or img.size == 0:
            return False
        mean_value = np.mean(img)
        return 5 <= mean_value <= 250

    def get_save_path(self, region_type: RegionType, custom_name: Optional[str] = None) -> str:
        """获取保存路径"""
        save_dir = os.path.join("data/screenshots", region_type.name.lower())
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{custom_name}.png" if custom_name else f"{region_type.value}.png"
        return os.path.join(save_dir, filename)

    def capture_region(self, device_id: str, region_type: RegionType, 
                      region: Tuple[int, int, int, int],
                      custom_name: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """截取指定区域的图片"""
        try:
            # 截取全屏
            success, result = self.screen_capture.capture(device_id)
            if not success:
                return False, {"error": f"截屏失败: {result}"}

            # 读取图片
            img = cv2.imread(result)
            if img is None:
                return False, {"error": f"无法读取图片: {result}"}

            # 检查图片质量
            if not self.check_image_quality(img):
                return False, {"error": "图片质量检查失败"}

            # 裁剪指定区域
            x, y, w, h = region
            if not self.validate_region(x, y, w, h, img.shape):
                return False, {"error": "区域坐标无效"}

            # 保存图片
            roi = img[y:y+h, x:x+w]
            save_path = self.get_save_path(region_type, custom_name)
            cv2.imwrite(save_path, roi)

            return True, {
                "success": True,
                "path": save_path,
                "region": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "type": region_type.value
            }

        except Exception as e:
            return False, {"error": str(e)}

if __name__ == "__main__":
    device_id = "127.0.0.1:16384"
    processor = ScreenRegionProcessor()
    
    test_regions = [
        (RegionType.PUBLIC, (200, 300, 400, 500)),
        (RegionType.HAND, (200, 300, 400, 500)),
        (RegionType.OP, (200, 300, 400, 500))
    ]
    
    for region_type, region in test_regions:
        success, result = processor.capture_region(
            device_id, 
            region_type,
            region=region
        )
        print(f"{region_type.name}区域截图{'成功' if success else '失败'}：", result) 
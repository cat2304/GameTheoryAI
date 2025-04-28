import os
import cv2
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from src.core.screen_all import ScreenCapture
from src.core.enums import RegionType

# ============ 日志配置 ============
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
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return False
        if x + w > img_w or y + h > img_h:
            return False
        return True

    def check_image_quality(self, img: np.ndarray) -> bool:
        """检查图片质量"""
        if img is None or img.size == 0:
            return False
        # 检查图片是否全黑或全白
        if np.mean(img) < 5 or np.mean(img) > 250:
            return False
        return True

    def get_save_path(self, region_type: RegionType, custom_name: Optional[str] = None) -> str:
        """获取保存路径"""
        save_dir = os.path.join("data/screenshots", str(region_type.value))
        os.makedirs(save_dir, exist_ok=True)
        
        if custom_name:
            filename = f"{custom_name}.png"
        else:
            filename = f"{region_type.value}.png"
            
        return os.path.join(save_dir, filename)

    def capture_region(self, device_id: str, region_type: RegionType, 
                      region: Tuple[int, int, int, int],
                      custom_name: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        根据区域类型截取指定区域的图片
        
        Args:
            device_id: 设备ID
            region_type: 区域类型
            region: 区域坐标 (x, y, width, height)
            custom_name: 自定义文件名（不包含扩展名）
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否成功, 结果信息)
        """
        try:
            # 截取全屏
            success, result = self.screen_capture.capture(device_id)
            if not success:
                self.logger.error(f"截屏失败: {result}")
                return False, {"error": f"截屏失败: {result}"}

            # 读取图片
            img = cv2.imread(result)
            if img is None:
                self.logger.error(f"无法读取图片: {result}")
                return False, {"error": f"无法读取图片: {result}"}

            # 检查图片质量
            if not self.check_image_quality(img):
                self.logger.error("图片质量检查失败")
                return False, {"error": "图片质量检查失败"}

            # 裁剪指定区域
            x, y, w, h = region
            if not self.validate_region(x, y, w, h, img.shape):
                self.logger.error(f"区域坐标无效: x={x}, y={y}, w={w}, h={h}")
                return False, {"error": "区域坐标无效"}

            roi = img[y:y+h, x:x+w]

            # 获取保存路径
            save_path = self.get_save_path(region_type, custom_name)

            # 保存图片
            cv2.imwrite(save_path, roi)

            self.logger.info(f"成功保存区域截图: {save_path}")
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
            self.logger.error(f"区域截图失败: {str(e)}")
            return False, {"error": str(e)}

if __name__ == "__main__":
    # 测试代码
    device_id = "127.0.0.1:16384"
    processor = ScreenRegionProcessor()
    
    # 测试公牌区域截图
    region = (200, 300, 400, 500)  # (x, y, width, height)
    success, result = processor.capture_region(
        device_id, 
        RegionType.PUBLIC,
        region=region
    )
    if success:
        print("公牌区域截图结果：", result)
    else:
        print("公牌区域截图失败：", result.get("error", "未知错误"))

    # 测试手牌区域截图
    region = (200, 300, 400, 500)  # (x, y, width, height)
    success, result = processor.capture_region(
        device_id, 
        RegionType.HAND,
        region=region
    )
    if success:
        print("手牌区域截图结果：", result)
    else:
        print("手牌区域截图失败：", result.get("error", "未知错误"))

    # 测试操作区域截图
    region = (200, 300, 400, 500)  # (x, y, width, height)
    success, result = processor.capture_region(
        device_id, 
        RegionType.OP,
        region=region
    )
    if success:
        print("操作区域截图结果：", result)
    else:
        print("操作区域截图失败：", result.get("error", "未知错误")) 
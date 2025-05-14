import os
import cv2
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime

# ============ 配置常量 ============
DEBUG_DIR = "data/debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ============ 区域类型 ============
class RegionType(Enum):
    PUBLIC = 1  # 公牌
    HAND = 2    # 手牌
    OP = 3      # 操作

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RegionProcessor:
    def __init__(self):
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

    def copy_region(self, image_path: str, region: Tuple[int, int, int, int], type: RegionType = RegionType.PUBLIC) -> Tuple[bool, Dict[str, Any]]:
        """复制指定区域的图片"""
        try:
            # 检查输入参数
            if not os.path.exists(image_path):
                return False, {"error": f"源图片不存在: {image_path}"}
            
            if len(region) != 4:
                return False, {"error": "区域参数格式错误，应为 (x, y, width, height)"}
            
            # 获取图片尺寸
            success, size_info = self.get_image_size(image_path)
            if not success:
                return False, size_info
            
            # 检查区域是否有效
            x, y, w, h = region
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return False, {"error": "无效的区域参数"}
            
            if x + w > size_info["width"] or y + h > size_info["height"]:
                return False, {"error": "区域超出图片范围"}
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return False, {"error": f"无法读取图片: {image_path}"}
            
            # 裁剪指定区域
            roi = img[y:y+h, x:x+w]
            
            # 保存裁剪后的图片
            save_dir = os.path.join("data/screenshots", type.name.lower())
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            filename = f"{type.value}.png"
            save_path = os.path.join(save_dir, filename)
            
            # 保存图片
            cv2.imwrite(save_path, roi)
            
            return True, {
                "success": True,
                "path": save_path
            }
            
        except Exception as e:
            self.logger.error(f"区域复制失败: {str(e)}")
            return False, {"error": str(e)}

if __name__ == "__main__":
    test_image = "data/screenshots/latest.png"
    if os.path.exists(test_image):
        processor = RegionProcessor()
        # 获取图片尺寸
        success, size_info = processor.get_image_size(test_image)
        if success:
            print("图片尺寸信息：", size_info)
            
            # 测试区域复制
            success, result = processor.copy_region(
                image_path=test_image,
                region=(100, 100, 200, 200),
                type=RegionType.PUBLIC  # 指定区域类型
            )
            if success:
                print("裁剪成功：", result)
            else:
                print("裁剪失败：", result.get("error", "未知错误"))
           
        else:
            print("获取图片尺寸失败：", size_info.get("error", "未知错误"))
    else:
        logger.error(f"测试图片不存在: {test_image}")


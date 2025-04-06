from pathlib import Path
from src.utils.log_utils import get_logger
from .opencv_algorithm import OpenCVAlgorithm

class OpenCVProcessor:
    def __init__(self):
        # 初始化OpenCV算法引擎
        self.opencv_engine = OpenCVAlgorithm()
        
        # 获取日志记录器
        self.logger = get_logger("opencv_processor")
    
    def process(self, img_path):
        """处理图片并识别麻将牌"""
        try:
            # 检查文件是否存在
            if not Path(img_path).exists():
                raise FileNotFoundError(f"文件不存在: {img_path}")
            
            self.logger.info(f"开始处理图片: {img_path}")
            
            # 预处理图像
            processed_img, original_img = self.opencv_engine.preprocess_image(img_path)
            
            # 查找麻将牌区域
            tiles = self.opencv_engine.find_mahjong_tiles(processed_img, original_img)
            
            # 识别每个麻将牌
            for tile in tiles:
                tile['result'] = self.opencv_engine.recognize_tile(original_img, tile)
            
            # 准备返回结果
            result = {
                'success': True,
                'tiles': tiles,
                'preprocessed': True,
                'total_tiles': len(tiles)
            }
            
            # 记录识别结果
            self.logger.info(f"识别完成，共找到 {len(tiles)} 张麻将牌")
            
            return result
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'tiles': []
            }

# 使用示例
if __name__ == "__main__":
    # 初始化OpenCV引擎
    processor = OpenCVProcessor()
    
    # 处理截图
    result = processor.process('mahjong_screenshot.png')
    
    # 打印结果
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result) 
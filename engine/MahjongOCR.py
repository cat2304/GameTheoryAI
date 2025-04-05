import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import logging
from pathlib import Path

class MahjongOCR:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        # 自定义麻将识别配置
        self.config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789V.红中麻将库侠信店方大集底分新技能机器人'
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, img_path):
        """专业级图像预处理"""
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值二值化
            thresh = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 4)
            
            # 降噪处理
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return processed
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def extract_text(self, img_path):
        """高级OCR文本提取"""
        try:
            # 预处理图像
            processed_img = self.preprocess_image(img_path)
            
            # 使用Tesseract进行OCR
            text = pytesseract.image_to_string(
                Image.fromarray(processed_img),
                lang='chi_sim+eng',
                config=self.config
            )
            
            self.logger.info(f"OCR识别结果: {text}")
            return text
        except Exception as e:
            self.logger.error(f"OCR识别失败: {str(e)}")
            raise
    
    def parse_mahjong_data(self, text):
        """麻将数据专业解析"""
        result = {
            'version': None,
            'robot_id': None,
            'game_type': None,
            'scores': [],
            'base_score': None,
            'duration': None
        }
        
        try:
            # 版本号识别
            version_match = re.search(r'V(\d+\.\d+\.\d+)', text)
            if version_match:
                result['version'] = version_match.group(1)
            
            # 机器人ID识别
            robot_match = re.search(r'robot_(\w+)', text)
            if robot_match:
                result['robot_id'] = robot_match.group(1)
            
            # 游戏类型识别
            game_match = re.search(r'(红中麻将\w+)', text)
            if game_match:
                result['game_type'] = game_match.group(1)
            
            # 分数识别
            score_matches = re.findall(r'([A-Z]{2}\d+[a-z]*)\s*(\d+\.\d+)', text)
            result['scores'] = [{'player': m[0], 'score': float(m[1])} for m in score_matches]
            
            # 底分识别
            base_match = re.search(r'底分[:：]\s*(\d+)', text)
            if base_match:
                result['base_score'] = int(base_match.group(1))
            
            # 游戏时长识别
            time_match = re.search(r'(\d{2}[:：]\d{2}[:：]\d{2})', text)
            if time_match:
                result['duration'] = time_match.group(1)
            
            self.logger.info(f"解析结果: {result}")
            return result
        except Exception as e:
            self.logger.error(f"数据解析失败: {str(e)}")
            raise
    
    def process(self, img_path):
        """端到端处理流程"""
        try:
            # 检查文件是否存在
            if not Path(img_path).exists():
                raise FileNotFoundError(f"文件不存在: {img_path}")
            
            # 提取文本
            text = self.extract_text(img_path)
            
            # 解析数据
            result = self.parse_mahjong_data(text)
            
            return result
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            raise

# 使用示例
if __name__ == "__main__":
    # 初始化OCR引擎
    ocr = MahjongOCR(tesseract_path='/usr/bin/tesseract')  # 修改为您的tesseract路径
    
    # 处理截图
    result = ocr.process('mahjong_screenshot.png')
    
    # 打印结果
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)
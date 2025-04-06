import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from MahjongOCR import MahjongOCR
import re
import cv2
import numpy as np

class MahjongAnalyzer:
    def __init__(self):
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化OCR
        self.ocr = MahjongOCR()
        self.img_dir = "/Users/mac/ai/img"
        self.date_format = "%Y%m%d"
        
        # 麻将牌映射
        self.tile_mapping = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9'
        }
        
    def get_latest_image(self):
        """获取最新的图片"""
        try:
            today = datetime.now().strftime(self.date_format)
            pattern = f"{today}*.png"
            
            # 获取所有匹配的文件
            files = list(Path(self.img_dir).glob(pattern))
            if not files:
                self.logger.warning(f"没有找到今天({today})的图片")
                return None
                
            # 按修改时间排序，返回最新的
            latest = max(files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"找到最新图片: {latest}")
            return str(latest)
        except Exception as e:
            self.logger.error(f"获取图片失败: {str(e)}")
            return None
    
    def enhance_image(self, img):
        """增强图像质量"""
        try:
            # 调整对比度和亮度
            alpha = 1.5  # 对比度
            beta = 10    # 亮度
            enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # 降噪
            denoised = cv2.fastNlMeansDenoisingColored(enhanced)
            
            # 锐化
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            return img
    
    def normalize_tile_text(self, text):
        """标准化麻将牌文本"""
        # 替换中文数字
        for cn, num in self.tile_mapping.items():
            text = text.replace(cn, num)
        return text
    
    def analyze_game_state(self, img_path):
        """分析游戏状态"""
        try:
            # 读取并增强图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图片: {img_path}")
            
            enhanced_img = self.enhance_image(img)
            
            # 转换为灰度图
            gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            
            # 分析不同区域
            height, width = gray.shape[:2]
            
            # 1. 分析手牌区域（底部）
            hand_region = gray[int(height*0.8):height, :]
            hand_text = pytesseract.image_to_string(hand_region, lang='chi_sim')
            hand_text = self.normalize_tile_text(hand_text)
            print("\n=== 手牌区域 ===")
            print(hand_text.strip())
            
            # 2. 分析场面信息（左上角）
            info_region = gray[:int(height*0.2), :int(width*0.3)]
            info_text = pytesseract.image_to_string(info_region, lang='chi_sim')
            print("\n=== 场面信息 ===")
            print(info_text.strip())
            
            # 3. 分析当前局数（中间）
            round_region = gray[int(height*0.4):int(height*0.6), 
                              int(width*0.4):int(width*0.6)]
            round_text = pytesseract.image_to_string(round_region, lang='chi_sim')
            print("\n=== 当前局数 ===")
            print(round_text.strip())
            
            # 4. 分析玩家分数
            score_region = gray[int(height*0.8):height, :int(width*0.2)]
            score_text = pytesseract.image_to_string(score_region, lang='chi_sim')
            print("\n=== 玩家分数 ===")
            print(score_text.strip())
            
            # 解析具体的牌型
            tiles = []
            patterns = [
                r'[1-9一二三四五六七八九][万条筒]',  # 数字+花色
                r'[东南西北中发白]',  # 字牌
                r'[1-9][mps]'  # 数字+字母简写
            ]
            
            for line in hand_text.split('\n'):
                if line.strip():
                    for pattern in patterns:
                        matches = re.findall(pattern, line)
                        tiles.extend(matches)
            
            if tiles:
                print("\n=== 识别到的牌 ===")
                print(" ".join(tiles))
                
                # 分析牌型
                print("\n=== 牌型分析 ===")
                self.analyze_tiles(tiles)
            
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            raise
    
    def analyze_tiles(self, tiles):
        """分析牌型"""
        try:
            # 统计每种牌的数量
            tile_count = {}
            for tile in tiles:
                tile_count[tile] = tile_count.get(tile, 0) + 1
            
            # 输出统计信息
            print("牌型统计:")
            for tile, count in tile_count.items():
                print(f"{tile}: {count}张")
            
            # 分析可能的组合
            self.analyze_combinations(tiles)
            
        except Exception as e:
            self.logger.error(f"牌型分析失败: {str(e)}")
    
    def analyze_combinations(self, tiles):
        """分析可能的组合"""
        try:
            # 按花色分类
            wan = [t for t in tiles if '万' in t]
            tiao = [t for t in tiles if '条' in t]
            tong = [t for t in tiles if '筒' in t]
            zipai = [t for t in tiles if t in ['东', '南', '西', '北', '中', '发', '白']]
            
            print("\n可能的组合:")
            if wan:
                print(f"万子: {' '.join(wan)}")
            if tiao:
                print(f"条子: {' '.join(tiao)}")
            if tong:
                print(f"筒子: {' '.join(tong)}")
            if zipai:
                print(f"字牌: {' '.join(zipai)}")
            
        except Exception as e:
            self.logger.error(f"组合分析失败: {str(e)}")

def main():
    try:
        analyzer = MahjongAnalyzer()
        latest_img = analyzer.get_latest_image()
        
        if latest_img:
            print(f"正在分析图片: {latest_img}")
            analyzer.analyze_game_state(latest_img)
        else:
            print("未找到可分析的图片")
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")

if __name__ == "__main__":
    main() 
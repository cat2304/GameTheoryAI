"""
麻将AI核心模块

整合了OCR识别、配置管理和工具函数。
"""

import os
import cv2
import numpy as np
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pytesseract
import time

class MahjongAI:
    """麻将AI主类"""
    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config = self._load_config(config_path)
        self.templates = self._load_templates()
        self.logger = self._setup_logger()
        self.adb_path = self.config.get('adb', {}).get('path', 'adb')

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'mahjong.log'),
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        return logger

    def _load_templates(self) -> dict:
        """加载模板图片"""
        templates = {}
        template_dir = self.config.get('template_dir', 'data/templates')
        for tile_type in ['characters', 'bamboos', 'dots', 'winds', 'dragons']:
            templates[tile_type] = {}
            for value in range(1, 10):
                path = f"{template_dir}/{tile_type}_{value}.png"
                if os.path.exists(path):
                    templates[tile_type][value] = cv2.imread(path, 0)
        return templates

    def _run_adb_command(self, command: List[str]) -> Tuple[int, str, str]:
        """执行ADB命令"""
        full_command = [self.adb_path] + command
        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True
        )
        return process.returncode, process.stdout, process.stderr

    def take_screenshot(self) -> Optional[str]:
        """获取设备截图"""
        try:
            # 生成远程文件名
            remote_path = f"/sdcard/screenshot_{int(time.time())}.png"
            
            # 执行截图命令
            code, _, stderr = self._run_adb_command(["shell", "screencap", "-p", remote_path])
            if code != 0:
                self.logger.error(f"截图失败: {stderr}")
                return None
            
            # 设置本地保存路径
            save_dir = self.config.get('screenshot_dir', 'screenshots')
            os.makedirs(save_dir, exist_ok=True)
            local_path = os.path.join(save_dir, os.path.basename(remote_path))
            
            # 拉取截图到本地
            code, _, stderr = self._run_adb_command(["pull", remote_path, local_path])
            if code != 0:
                self.logger.error(f"拉取截图失败: {stderr}")
                return None
            
            # 删除远程文件
            self._run_adb_command(["shell", "rm", remote_path])
            
            return local_path
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 降噪
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    def detect_tiles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测麻将牌位置"""
        processed = self.preprocess_image(image)
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        tiles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_valid_tile(w, h):
                tiles.append((x, y, w, h))
        
        return sorted(tiles, key=lambda t: (t[1], t[0]))

    def _is_valid_tile(self, width: int, height: int) -> bool:
        """判断是否为有效的麻将牌"""
        min_size = self.config.get('min_tile_size', 30)
        max_size = self.config.get('max_tile_size', 100)
        aspect_ratio = width / height
        
        return (min_size <= width <= max_size and
                min_size <= height <= max_size and
                0.8 <= aspect_ratio <= 1.2)

    def recognize_tile(self, image: np.ndarray, tile_roi: Tuple[int, int, int, int]) -> Optional[str]:
        """识别单个麻将牌"""
        x, y, w, h = tile_roi
        tile_img = image[y:y+h, x:x+w]
        processed = self.preprocess_image(tile_img)
        
        # 模板匹配
        best_match = None
        best_score = 0
        
        for tile_type, templates in self.templates.items():
            for value, template in templates.items():
                if template.shape != processed.shape:
                    template = cv2.resize(template, (processed.shape[1], processed.shape[0]))
                
                result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = f"{tile_type}_{value}"
        
        if best_score > self.config.get('min_confidence', 0.7):
            return best_match
        
        # 如果模板匹配失败，尝试OCR
        try:
            text = pytesseract.image_to_string(
                processed,
                config='--psm 10 --oem 3'
            ).strip()
            
            if text:
                return self._parse_ocr_text(text)
        except Exception as e:
            self.logger.error(f"OCR识别失败: {str(e)}")
        
        return None

    def _parse_ocr_text(self, text: str) -> Optional[str]:
        """解析OCR识别的文本"""
        # 实现文本解析逻辑
        pass

    def recognize_all_tiles(self, image: np.ndarray) -> List[Optional[str]]:
        """识别图像中的所有麻将牌"""
        tiles = self.detect_tiles(image)
        return [self.recognize_tile(image, tile) for tile in tiles]

    def run(self):
        """运行麻将AI"""
        self.logger.info("麻将AI启动")
        
        while True:
            try:
                # 获取截图
                screenshot_path = self.take_screenshot()
                if not screenshot_path:
                    self.logger.error("获取截图失败")
                    continue
                
                # 读取图像
                image = cv2.imread(screenshot_path)
                if image is None:
                    self.logger.error("读取截图失败")
                    continue
                
                # 识别麻将牌
                tiles = self.recognize_all_tiles(image)
                self.logger.info(f"识别结果: {tiles}")
                
                # 等待一段时间
                time.sleep(self.config.get('interval', 1))
                
            except KeyboardInterrupt:
                self.logger.info("收到退出信号，正在退出...")
                break
            except Exception as e:
                self.logger.error(f"运行出错: {str(e)}")
                continue

if __name__ == "__main__":
    ai = MahjongAI()
    ai.run() 
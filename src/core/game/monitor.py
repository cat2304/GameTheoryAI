import subprocess
import threading
import time
import re
from datetime import datetime
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

class MahjongGameMonitor:
    def __init__(self):
        self.process = None
        self.is_running = False
        self.thread = None
        self.players = {}  # 用于存储玩家信息
        self.current_game = []  # 存储当前游戏的所有动作
        self.screenshot_dir = "screenshots"  # 截图保存目录
        
        # 创建截图目录
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def _parse_game_action(self, log_line):
        """解析游戏动作"""
        try:
            # 打印原始日志用于调试
            print(f"原始日志: {log_line.strip()}")
            
            # 尝试多种匹配模式
            patterns = [
                # 模式1: 标准出牌格式
                (r'出牌\[(\d+)\]\s*([东南西北中发白一二三四五六七八九万条筒]+)', 'play'),
                # 模式2: 吃碰杠格式
                (r'(吃|碰|杠)\[(\d+)\]\s*([东南西北中发白一二三四五六七八九万条筒]+)', 'action'),
                # 模式3: 简化的出牌格式
                (r'Player\[(\d+)\].*?played\[([东南西北中发白一二三四五六七八九万条筒]+)\]', 'play'),
                # 模式4: 带时间戳的格式
                (r'\[(\d+)\]\s*([东南西北中发白一二三四五六七八九万条筒]+)', 'play'),
                # 模式5: 中文描述格式
                (r'玩家(\d+).*?(打出|打|出)([东南西北中发白一二三四五六七八九万条筒]+)', 'play'),
                # 模式6: 数字格式
                (r'(\d+)\s*([东南西北中发白一二三四五六七八九万条筒]+)', 'play'),
                # 模式7: 九筒格式
                (r'(\d+)筒', 'play'),
                # 模式8: 九万格式
                (r'(\d+)万', 'play'),
                # 模式9: 九条格式
                (r'(\d+)条', 'play'),
                # 模式10: 简写格式
                (r'(\d+)([万条筒])', 'play'),
                # 模式11: 带方向的格式
                (r'([东南西北]家).*?([东南西北中发白一二三四五六七八九万条筒]+)', 'play'),
                # 模式12: 带动作的格式
                (r'(打出|打|出).*?([东南西北中发白一二三四五六七八九万条筒]+)', 'play'),
            ]
            
            for pattern, action_type in patterns:
                match = re.search(pattern, log_line)
                if match:
                    if action_type == 'play':
                        if len(match.groups()) == 2:
                            seat_id, tile = match.groups()
                            # 处理数字格式的牌
                            if tile.isdigit():
                                if '筒' in log_line:
                                    tile = f"{tile}筒"
                                elif '万' in log_line:
                                    tile = f"{tile}万"
                                elif '条' in log_line:
                                    tile = f"{tile}条"
                            return 'play', seat_id, tile
                        elif len(match.groups()) == 3:
                            seat_id, _, tile = match.groups()
                            return 'play', seat_id, tile
                    elif action_type == 'action':
                        action, seat_id, tiles = match.groups()
                        return action, seat_id, tiles
                        
            return None
        except Exception as e:
            print(f"解析游戏动作失败: {str(e)}")
            return None

    def capture_screenshot(self):
        """捕获游戏截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
            
            # 使用ADB截图
            subprocess.run(['adb', 'shell', 'screencap', '-p', '/sdcard/screenshot.png'])
            subprocess.run(['adb', 'pull', '/sdcard/screenshot.png', filename])
            
            return filename
        except Exception as e:
            print(f"截图失败: {str(e)}")
            return None

    def analyze_screenshot(self, image_path):
        """分析游戏截图"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print("无法读取图片")
                return
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 识别手牌区域（底部区域）
            height, width = img.shape[:2]
            hand_tiles_region = img[int(height*0.8):height, :]
            
            # 使用OCR识别文字
            text = pytesseract.image_to_string(hand_tiles_region, lang='chi_sim')
            
            # 解析识别到的文字
            tiles = []
            for line in text.split('\n'):
                if line.strip():
                    tiles.append(line.strip())
            
            # 输出结果
            print("\n=== 场面分析 ===")
            print("手牌:", " ".join(tiles))
            
            # 识别分数
            score_region = img[int(height*0.8):height, :int(width*0.2)]
            score_text = pytesseract.image_to_string(score_region, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            if score_text.strip():
                print(f"分数: {score_text.strip()}")
            
            # 识别当前局数
            round_region = img[int(height*0.4):int(height*0.6), int(width*0.4):int(width*0.6)]
            round_text = pytesseract.image_to_string(round_region, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            if round_text.strip():
                print(f"当前局数: {round_text.strip()}")
            
        except Exception as e:
            print(f"分析截图失败: {str(e)}")

    def start_monitoring(self):
        """开始监控游戏"""
        try:
            print("=== 麻将游戏监控已启动 ===")
            print("每5秒进行一次截图分析...")
            print("按 Ctrl+C 停止监控")
            
            while True:
                # 捕获并分析截图
                screenshot_path = self.capture_screenshot()
                if screenshot_path:
                    self.analyze_screenshot(screenshot_path)
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n停止监控")
        except Exception as e:
            print(f"监控失败: {str(e)}")

    def stop_monitoring(self):
        """停止监控"""
        if self.process:
            self.is_running = False
            self.process.terminate()
            self.process.wait()
            print("\n=== 监控已停止 ===")
            self._print_game_summary()

    def _print_game_summary(self):
        """打印游戏总结"""
        print("\n=== 游戏记录总结 ===")
        for seat_id, actions in self.players.items():
            print(f"\n玩家 {seat_id} 的行动记录:")
            print("出牌:", " -> ".join(actions['play']) if actions['play'] else "无")
            print("吃:", " | ".join(actions['chi']) if actions['chi'] else "无")
            print("碰:", " | ".join(actions['peng']) if actions['peng'] else "无")
            print("杠:", " | ".join(actions['gang']) if actions['gang'] else "无")
        
        print("\n=== 完整游戏记录 ===")
        for timestamp, action_type, seat_id, tiles in self.current_game:
            print(f"[{timestamp}] 玩家{seat_id} {action_type}: {tiles}")

def main():
    monitor = MahjongGameMonitor()
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n程序已退出")

if __name__ == "__main__":
    main()
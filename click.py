import subprocess
import time
import logging
from typing import Optional, Tuple
import random

class PokerClicker:
    def __init__(self, device_id: str = "127.0.0.1:16384"):
        self.device_id = device_id
        self.logger = logging.getLogger("PokerClicker")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler('poker_click.log'))
        
    def _run_adb_command(self, command: str) -> bool:
        """执行ADB命令"""
        try:
            result = subprocess.run(['adb', '-s', self.device_id, 'shell', command], 
                                  check=True, capture_output=True, text=True)
            self.logger.debug(f"ADB命令输出: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ADB命令执行失败: {command}")
            self.logger.error(f"错误信息: {e.stderr.decode()}")
            return False

    def _get_screen_info(self) -> Tuple[int, int]:
        """获取屏幕信息"""
        try:
            result = subprocess.run(['adb', '-s', self.device_id, 'shell', 'wm', 'size'], 
                                  check=True, capture_output=True, text=True)
            size = result.stdout.strip().split(': ')[1]
            width, height = map(int, size.split('x'))
            return width, height
        except Exception as e:
            self.logger.error(f"获取屏幕信息失败: {e}")
            return 1920, 1080  # 默认分辨率

    def start_motor(self) -> bool:
        """启动电机"""
        self.logger.info("正在启动电机...")
        # 获取屏幕信息
        width, height = self._get_screen_info()
        # 计算相对位置
        x = int(width * 0.5)  # 屏幕中间
        y = int(height * 0.3)  # 屏幕上方30%处
        
        # 添加随机偏移
        x += random.randint(-10, 10)
        y += random.randint(-10, 10)
        
        self.logger.info(f"点击位置: ({x}, {y})")
        time.sleep(1)  # 点击前等待
        success = self._run_adb_command(f'input tap {x} {y}')
        if success:
            self.logger.info("电机启动成功")
            time.sleep(2)  # 等待电机启动
        return success

    def click_tile(self, tile: Optional[str], position: Optional[Tuple[int, int]] = None, retry: int = 3) -> bool:
        """点击指定的牌"""
        if tile is None:
            self.logger.info("无需点击")
            return True
            
        # 获取屏幕信息
        width, height = self._get_screen_info()
            
        if position is None:
            # 默认点击位置（屏幕中间）
            x = int(width * 0.5)
            y = int(height * 0.5)
        else:
            x, y = position
            
        # 添加随机偏移
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)
        
        for attempt in range(retry):
            self.logger.info(f"尝试点击 {tile}，坐标: ({x}, {y})，第{attempt + 1}次")
            time.sleep(0.5)  # 点击前等待
            
            # 使用长按来确保点击生效
            success = self._run_adb_command(f'input swipe {x} {y} {x} {y} 100')
            if success:
                self.logger.info(f"点击 {tile} 成功")
                time.sleep(0.5)  # 点击后等待
                return True
                
            self.logger.warning(f"点击 {tile} 失败，重试中...")
            time.sleep(1)  # 失败后等待
            
        self.logger.error(f"点击 {tile} 失败，已达到最大重试次数")
        return False

    def click_public_cards(self, cards: list) -> bool:
        """点击公共牌"""
        if not cards:
            self.logger.info("没有公共牌需要点击")
            return True
            
        self.logger.info(f"正在点击公共牌: {cards}")
        # 获取屏幕信息
        width, height = self._get_screen_info()
        
        # 公共牌的位置坐标（相对位置）
        positions = {
            'A': (0.3, 0.4),
            'K': (0.4, 0.4),
            'Q': (0.5, 0.4),
            'J': (0.6, 0.4),
            '10': (0.7, 0.4)
        }
        
        for card in cards:
            if card in positions:
                rel_x, rel_y = positions[card]
                x = int(width * rel_x)
                y = int(height * rel_y)
                if not self.click_tile(card, (x, y)):
                    return False
        return True

    def click_hand_cards(self, cards: list) -> bool:
        """点击手牌"""
        if not cards:
            self.logger.info("没有手牌需要点击")
            return True
            
        self.logger.info(f"正在点击手牌: {cards}")
        # 获取屏幕信息
        width, height = self._get_screen_info()
        
        # 手牌的位置坐标（相对位置）
        positions = {
            'A': (0.3, 0.8),
            'K': (0.4, 0.8),
            'Q': (0.5, 0.8),
            'J': (0.6, 0.8),
            '10': (0.7, 0.8)
        }
        
        for card in cards:
            if card in positions:
                rel_x, rel_y = positions[card]
                x = int(width * rel_x)
                y = int(height * rel_y)
                if not self.click_tile(card, (x, y)):
                    return False
        return True

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建点击器实例
    clicker = PokerClicker()
    
    # 启动电机
    if not clicker.start_motor():
        return
        
    # 示例：点击公共牌和手牌
    public_cards = ['A', 'K', 'Q']
    hand_cards = ['J', '10']
    
    clicker.click_public_cards(public_cards)
    clicker.click_hand_cards(hand_cards)

if __name__ == "__main__":
    main()
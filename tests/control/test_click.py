import unittest
from src.control.click import PokerClicker

class TestPokerClicker(unittest.TestCase):
    def setUp(self):
        self.clicker = PokerClicker()
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.clicker.device_id)
        self.assertIsNotNone(self.clicker.logger)
    
    def test_get_screen_info(self):
        """测试获取屏幕信息"""
        width, height = self.clicker._get_screen_info()
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
    
    def test_click_position(self):
        """测试点击位置"""
        # 测试点击屏幕中心
        width, height = self.clicker._get_screen_info()
        x, y = width // 2, height // 2
        success = self.clicker.click_position(x, y)
        self.assertIsInstance(success, bool)
    
    def test_click_card(self):
        """测试点击牌"""
        # 测试点击 A 牌
        success = self.clicker.click_card("A")
        self.assertIsInstance(success, bool)
    
    def test_click_public_cards(self):
        """测试点击公共牌"""
        # 测试点击公共牌
        success = self.clicker.click_public_cards(["A", "K", "Q"])
        self.assertIsInstance(success, bool)
    
    def test_click_hand_cards(self):
        """测试点击手牌"""
        # 测试点击手牌
        success = self.clicker.click_hand_cards(["A", "K"])
        self.assertIsInstance(success, bool)
    
    def test_click_action(self):
        """测试点击操作按钮"""
        # 测试点击各个操作按钮
        actions = ["弃牌", "加注", "让牌", "跟注"]
        for action in actions:
            success = self.clicker.click_action(action)
            self.assertIsInstance(success, bool)

if __name__ == '__main__':
    unittest.main() 
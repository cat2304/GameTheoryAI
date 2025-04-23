import time
import logging
from typing import Dict, Any, Optional, Tuple
from src.vision.screen import ScreenCapture
from src.vision.ocr import recognize_cards
from src.core.game_state import GameState, GameRound
from src.core.game_maker import GameMaker
from src.core.game_executor import GameExecutor
from src.core.game_click import GameClicker

class GameController:
    # 常量定义
    SCREENSHOT_RETRY_DELAY = 5  # 截图失败后的重试延迟（秒）
    GAME_LOOP_DELAY = 5  # 游戏主循环的延迟（秒）

    def __init__(self, screen_capture: Optional[ScreenCapture] = None):
        """初始化游戏控制器 """
        self._setup_logging()
        self._initialize_components(screen_capture)
        self._reset_game_state()

    def _setup_logging(self) -> None:
        """设置日志记录器"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化游戏控制器...")

    def _initialize_components(self, screen_capture: Optional[ScreenCapture]) -> None:
        """初始化所有组件"""
        self.screen_capture = screen_capture if screen_capture is not None else ScreenCapture()
        self.game_state = GameState()
        self.game_maker = GameMaker()
        self.game_executor = GameExecutor()
        self.game_clicker = GameClicker()
        self.logger.info("所有组件初始化完成")

    def _reset_game_state(self) -> None:
        """重置游戏状态"""
        self.last_hand_cards = []
        self.last_public_cards = []
        self.last_action = None

    def _take_screenshot(self) -> Tuple[bool, str]:
        """获取屏幕截图 """
        self.logger.info("开始获取截图...")
        success, image_path = self.screen_capture.take_screenshot()
        if not success:
            self.logger.error(f"截图失败: {image_path}")
        else:
            self.logger.info(f"截图成功: {image_path}")
        return success, image_path

    def _detect_card_changes(self, result: Dict[str, Any]) -> Tuple[bool, bool]:
        """检测牌面变化 """
        hand_changed = result["handCards"] != self.last_hand_cards
        public_changed = result["publicCards"] != self.last_public_cards
        
        if hand_changed or public_changed:
            self.logger.info(f"检测到牌面变化:")
            self.logger.info(f"手牌变化: {self.last_hand_cards} -> {result['handCards']}")
            self.logger.info(f"公牌变化: {self.last_public_cards} -> {result['publicCards']}")
        
        return hand_changed, public_changed

    def _update_game_state(self, result: Dict[str, Any]) -> None:
        """更新游戏状态"""
        self.last_hand_cards = result["handCards"]
        self.last_public_cards = result["publicCards"]
        
        self.game_state.update_cards(
            [card["action"] for card in result["handCards"]],
            [card["action"] for card in result["publicCards"]]
        )

    def _make_and_execute_decision(self) -> None:
        """进行决策并执行"""
        self.logger.info("开始决策过程...")
        decision = self.game_maker.make_decision(self.game_state)
        self.logger.info(f"决策结果: {decision}")
        
        self.game_clicker.execute_decision(decision)

    def process_screenshot(self, image_path: str) -> Dict[str, Any]:
        """处理截图并返回识别结果"""
        try:
            self.logger.info("开始OCR识别...")
            result = recognize_cards(image_path)
            
            if not result["success"]:
                self.logger.error(f"识别失败: {result['error']}")
                return result
            
            hand_changed, public_changed = self._detect_card_changes(result)
            
            if hand_changed or public_changed:
                self._update_game_state(result)
                self._make_and_execute_decision()
            else:
                self.logger.info("牌面未发生变化")
            
            return result
            
        except Exception as e:
            error_msg = f"处理截图失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def run(self) -> None:
        """运行游戏监控主循环"""
        try:
            self.logger.info("开始游戏监控...")
            while True:
                success, image_path = self._take_screenshot()
                if not success:
                    time.sleep(self.SCREENSHOT_RETRY_DELAY)
                    continue
                
                result = self.process_screenshot(image_path)
                if not result["success"]:
                    time.sleep(self.SCREENSHOT_RETRY_DELAY)
                    continue
                
                self.logger.info(f"识别结果: {result}")
                time.sleep(self.GAME_LOOP_DELAY)
                
        except Exception as e:
            self.logger.error(f"游戏监控发生错误: {str(e)}")
        finally:
            self.logger.info("清理资源...") 
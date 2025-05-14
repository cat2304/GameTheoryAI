# decision_module.py

"""
德州扑克 AI 决策模块

该模块负责:
1.  接收来自视觉识别模块的结构化牌局状态。
2.  将牌局状态转换为 `pokerllm/pokerbench` 模型所需的输入格式。
3.  调用 `pokerllm/pokerbench` 模型进行决策推理。
4.  解析模型输出，并将其转换为标准化的动作指令。
"""

import json
# import <pokerllm_model_library> # 假设的 pokerllm 模型库，后续根据实际情况替换

class DecisionModule:
    def __init__(self, model_path=None):
        """
        初始化决策模块，加载 pokerllm/pokerbench 模型。
        model_path: 模型文件或目录的路径 (如果需要本地加载)
        """
        self.model = None
        # self.tokenizer = None # 如果模型需要 tokenizer
        # if model_path:
        #     self.load_model(model_path)
        print("决策模块初始化完成。模型加载逻辑待实现。")

    def load_model(self, model_path):
        """
        加载 pokerllm/pokerbench 模型。
        具体的加载方式取决于 pokerllm/pokerbench 提供的 API 或库。
        """
        # 示例: (需要根据实际模型库调整)
        # self.model = <pokerllm_model_library>.load(model_path)
        # self.tokenizer = <pokerllm_model_library>.tokenizer(model_path)
        print(f"模型加载中 (路径: {model_path})... 此处为占位符，后续实现具体加载逻辑。")
        # 假设模型加载成功
        self.model = "pokerllm_placeholder_model"
        print("模型加载成功 (占位符)。")

    def _format_input_for_model(self, game_state):
        """
        将结构化的牌局状态 (来自视觉识别模块) 转换为模型所需的输入格式。
        pokerllm/pokerbench 的输入是自然语言指令。

        game_state (dict): 结构化的牌局信息，示例如下:
        {
            "my_hand": ["As", "Kd"], // 自己的手牌
            "community_cards": ["Qh", "Js", "Td"], // 公共牌
            "pot_size": 150, // 底池大小
            "current_player": "Hero", // 当前行动玩家
            "hero_position": "BB", // Hero的位置
            "players": [
                {"name": "Player1", "stack": 850, "position": "SB", "last_action": "BET", "bet_amount": 50},
                {"name": "Hero", "stack": 950, "position": "BB", "last_action": null, "bet_amount": 0}
            ],
            "available_actions": ["CALL", "RAISE", "FOLD"], // Hero可执行的动作
            "game_stage": "FLOP", // 游戏阶段 (PREFLOP, FLOP, TURN, RIVER)
            "betting_history": "UTG/RAISE/2BB/BTN/CALL/SB/FOLD" // 之前的下注历史
        }
        """
        instruction = "You are a specialist in playing 6-handed No Limit Texas Holdem. "
        instruction += "The following is a game scenario and you need to make the optimal decision. "

        if not game_state or not isinstance(game_state, dict):
            return "Error: Invalid game state provided."

        my_hand_str = " and ".join(game_state.get("my_hand", []))
        if my_hand_str:
            instruction += f"Your hand is {my_hand_str}. "
        else:
            instruction += "Your hand is unknown. "

        community_cards_str = ", ".join(game_state.get("community_cards", []))
        if community_cards_str:
            instruction += f"The community cards are {community_cards_str}. "
        else:
            instruction += "There are no community cards on the board yet. "

        pot_size = game_state.get("pot_size", 0)
        instruction += f"The current pot size is {pot_size}. "

        betting_history = game_state.get("betting_history", "No previous actions.")
        instruction += f"The betting history is: {betting_history}. "

        hero_position = game_state.get("hero_position", "Unknown")
        instruction += f"You are in position {hero_position}. "

        available_actions_str = ", ".join(game_state.get("available_actions", []))
        if available_actions_str:
            instruction += f"Your available actions are: {available_actions_str}. "

        instruction += "What is your optimal decision?"

        return instruction

    def get_decision(self, game_state):
        """
        根据当前牌局状态获取 AI 决策。

        game_state (dict): 结构化的牌局信息。
        Returns:
            str: AI 的决策指令 (例如 "CALL", "RAISE 100", "FOLD")
        """
        if self.model is None:
            print("错误：模型未加载。")
            available_actions = game_state.get("available_actions", [])
            if not available_actions: return "FOLD"
            if "CHECK" in available_actions:
                return "CHECK"
            if "FOLD" in available_actions:
                return "FOLD"
            return available_actions[0] # Fallback to first available action if CHECK/FOLD not options

        model_input_prompt = self._format_input_for_model(game_state)
        print(f"模型输入提示: {model_input_prompt}")

        # 模拟模型输出 (实际应调用 self.model.predict or similar)
        # This simulation logic should be improved or replaced by actual model call
        simulated_output = "FOLD" # Default simulated output
        available_actions_upper = [a.upper() for a in game_state.get("available_actions", [])]

        if "RAISE" in available_actions_upper and game_state.get("pot_size", 0) < 200:
            simulated_output = "RAISE 20"
        elif "BET" in available_actions_upper and game_state.get("pot_size", 0) < 50: # Example for BET
             simulated_output = "BET 10"
        elif "CALL" in available_actions_upper:
            simulated_output = "CALL"
        elif "CHECK" in available_actions_upper:
            simulated_output = "CHECK"
        
        print(f"模拟模型原始输出: {simulated_output}")

        decision = self._parse_model_output(simulated_output, game_state.get("available_actions", []))
        print(f"最终决策: {decision}")
        return decision

    def _parse_model_output(self, model_output_str, available_actions):
        output_upper = model_output_str.strip().upper()
        
        if not available_actions:
            return "FOLD"

        available_actions_upper = [a.upper() for a in available_actions]
        original_available_actions = available_actions # Keep original casing

        parsed_verb = ""
        parsed_amount = None
        parts = output_upper.split()
        if not parts:
            return "FOLD"

        parsed_verb = parts[0]
        if len(parts) > 1:
            try:
                amount_str = parts[1]
                cleaned_amount_str = "".join(filter(lambda x: x.isdigit() or x == '.', amount_str))
                if cleaned_amount_str:
                    parsed_amount = float(cleaned_amount_str)
            except ValueError:
                pass

        # 1. Exact match for the full model output (e.g., 

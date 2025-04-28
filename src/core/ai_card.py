import os
import torch
from ultralytics import YOLO
import cv2
import json
import uuid

class YOLOTester:
    def __init__(self, model_path):
        """
        初始化推理器
        :param model_path: 已训练好的 best.pt 权重文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")
        
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"🚀 使用设备: {'MPS加速' if self.device == 'mps' else 'CPU'}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"✅ 成功加载模型: {model_path}")

        # 类别名称映射
        self.class_names = ['10f', '10h', '10m', '10s', '2f', '2h', '2m', '2s', '3f', '3h', '3m', '3s', 
                          '4f', '4h', '4m', '4s', '5f', '5h', '5m', '5s', '6f', '6h', '6m', '6s', 
                          '7f', '7h', '7m', '7s', '8f', '8h', '8m', '8s', '9f', '9h', '9m', '9s', 
                          'Af', 'Ah', 'Am', 'As', 'Jf', 'Jh', 'Jm', 'Js', 'Kf', 'Kh', 'Km', 'Ks', 
                          'Qf', 'Qh', 'Qm', 'Qs']

    def predict_single_image(self, image_path):
        """
        对单张图片进行推理
        :param image_path: 要推理的图片路径
        :return: 识别结果字典
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 找不到图片: {image_path}")

        # 推理
        print(f"🔍 正在推理: {image_path}")
        results = self.model.predict(
            source=image_path,
            imgsz=640,
            conf=0.25,        # 置信度阈值
            iou=0.45,         # NMS的IoU阈值
            device=self.device,
            verbose=False
        )

        # 转换为预测结果
        predictions = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls_idx = box.tolist()
                cls_idx = int(cls_idx)
               
                # 计算中心点和宽高
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width/2
                y_center = y1 + height/2
                
                prediction = {
                    "x": round(x_center, 1),
                    "y": round(y_center, 1),
                    "width": round(width, 1),
                    "height": round(height, 1),
                    "confidence": round(conf, 3),
                    "class": self.class_names[cls_idx],
                    "class_id": cls_idx,
                    "detection_id": str(uuid.uuid4())
                }
                predictions.append(prediction)
        
        # 按置信度排序
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # 构建输出格式
        output = {
            "predictions": predictions
        }
        
        return output

def recognize_cards(image_path: str) -> dict:
    """识别卡牌
    
    Args:
        image_path: 图片路径
        
    Returns:
        dict: 识别结果，包含success、predictions、hand_cards、public_cards和error字段
    """
    try:
        model_path = "data/best.pt"  # 模型路径
        tester = YOLOTester(model_path)
        result = tester.predict_single_image(image_path)
        
        # 处理预测结果，将牌分类为手牌和公共牌
        hand_cards = []
        public_cards = []
        
        for pred in result["predictions"]:
            card = pred["class"]
            # 根据y坐标判断是手牌还是公共牌
            if pred["y"] > 0.7:  # 假设y坐标大于0.7的是手牌
                hand_cards.append(card)
            else:
                public_cards.append(card)
        
        return {
            "success": True,
            "predictions": result["predictions"],  # 添加完整的预测结果
            "hand_cards": hand_cards,
            "public_cards": public_cards
        }
    except Exception as e:
        return {
            "success": False,
            "predictions": [],
            "hand_cards": [],
            "public_cards": [],
            "error": str(e)
        }

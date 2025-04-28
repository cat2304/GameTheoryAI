import os
import torch
from ultralytics import YOLO
import cv2
import json
import uuid

class YOLOCard:
    def __init__(self, model_path):
        """初始化卡牌识别模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")
        
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"🚀 使用设备: {'MPS加速' if self.device == 'mps' else 'CPU'}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"✅ 成功加载模型: {model_path}")

        # 扑克牌类别名称
        self.class_names = ['10f', '10h', '10m', '10s', '2f', '2h', '2m', '2s', '3f', '3h', '3m', '3s', 
                          '4f', '4h', '4m', '4s', '5f', '5h', '5m', '5s', '6f', '6h', '6m', '6s', 
                          '7f', '7h', '7m', '7s', '8f', '8h', '8m', '8s', '9f', '9h', '9m', '9s', 
                          'Af', 'Ah', 'Am', 'As', 'Jf', 'Jh', 'Jm', 'Js', 'Kf', 'Kh', 'Km', 'Ks', 
                          'Qf', 'Qh', 'Qm', 'Qs']

    def predict_single_image(self, image_path):
        """识别单张图片中的扑克牌"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 找不到图片: {image_path}")

        print(f"🔍 正在推理: {image_path}")
        results = self.model.predict(
            source=image_path,
            imgsz=640,
            conf=0.39,    # 置信度阈值
            iou=0.45,     # NMS阈值
            device=self.device,
            verbose=False
        )

        # 处理识别结果
        predictions = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls_idx = box.tolist()
                cls_idx = int(cls_idx)
               
                # 计算中心点和大小
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width/2
                y_center = y1 + height/2
                
                predictions.append({
                    "x": round(x_center, 1),
                    "y": round(y_center, 1),
                    "width": round(width, 1),
                    "height": round(height, 1),
                    "confidence": round(conf, 3),
                    "class": self.class_names[cls_idx],
                    "class_id": cls_idx,
                    "detection_id": str(uuid.uuid4())
                })
        
        # 按位置排序（从上到下，从左到右）
        predictions.sort(key=lambda x: (x["y"], x["x"]))
        return {"predictions": predictions}

def recognize_cards(image_path: str) -> dict:
    """识别图片中的扑克牌，返回手牌和公共牌"""
    try:
        model_path = "data/best.pt"
        tester = YOLOCard(model_path)
        result = tester.predict_single_image(image_path)
        
        # 分类手牌和公共牌
        hand_cards = []
        public_cards = []
        for pred in result["predictions"]:
            card = pred["class"]
            if pred["y"] > 0.7:  # y坐标大于0.7为手牌
                hand_cards.append(card)
            else:
                public_cards.append(card)
        
        return {
            "success": True,
            "predictions": result["predictions"],
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

if __name__ == "__main__":
    # 配置
    model_path = "data/best.pt"
    image_path = "data/templates/test.png"
    save_dir = "data/debug/yolo"

    # 执行识别
    tester = YOLOCard(model_path)
    result = tester.predict_single_image(image_path)
    
    # 打印结果
    print("\n✨ 识别结果:")
    if result["predictions"]:
        for pred in result["predictions"]:
            print(f"- 牌面: {pred['class']}, 置信度: {pred['confidence']:.3f}, 位置: ({pred['x']:.1f}, {pred['y']:.1f})")
    else:
        print("没有识别到任何卡牌")

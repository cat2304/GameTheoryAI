import os
import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
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

    def predict_single_image(self, image_path, save_dir="./outputs"):
        """
        对单张图片进行推理并保存结果
        :param image_path: 要推理的图片路径
        :param save_dir: 保存推理结果的目录
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 找不到图片: {image_path}")

        os.makedirs(save_dir, exist_ok=True)

        # 推理
        print(f"🔍 正在推理: {image_path}")
        results = self.model.predict(
            source=image_path,
            save=True,
            project=save_dir,  # 使用project参数而不是save_dir
            name="predict",    # 指定输出目录名
            imgsz=640,
            conf=0.5,        # 置信度阈值，与 Roboflow 设置一致
            iou=0.45,         # NMS的IoU阈值
            device=self.device,
            verbose=False
        )

        # 转换为Roboflow格式的预测结果
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
        
        # 按位置排序：从上到下，从左到右
        predictions.sort(key=lambda x: (x["y"], x["x"]))
        
        # 构建完整的输出格式
        output = {
            "predictions": predictions
        }
        
        # 打印JSON格式的预测结果
        print(json.dumps(output, indent=2))
        
        # 保存完成后的路径 (注意：YOLOv8 输出jpg格式)
        base_name = os.path.splitext(os.path.basename(image_path))[0] + '.jpg'
        result_path = os.path.join(save_dir, "predict", base_name)
        print(f"✅ 推理完成，保存到: {result_path}")

        # 可选：展示推理后的图片
        if os.path.exists(result_path):
            img = cv2.imread(result_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.title('预测结果')
                plt.show()
            else:
                print(f"⚠️ 无法读取结果图片: {result_path}")
        else:
            print(f"⚠️ 结果图片不存在: {result_path}")

def main():
    # ==== 配置 ====
    model_path = "data/best.pt"  # 你的best.pt模型位置
    image_path = "data/screenshots/public/1.png"  # 测试图片路径
    save_dir = "data/debug/yolo"  # 结果保存目录

    # ==== 执行推理 ====
    tester = YOLOTester(model_path)
    tester.predict_single_image(image_path, save_dir)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import onnxruntime as ort
yaml = __import__("yaml")
from vision.utils import load_config, non_max_suppression, draw_results
from typing import List, Tuple

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # 对每个预测框的类别概率进行softmax
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class VisionPredictor:
    def __init__(self, config_path="configs/vision_config.yaml"):
        self.config = load_config(config_path)
        self.class_names = self.config["class_names"]
        self.session = ort.InferenceSession(self.config["model_path"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # 处理输入形状，使用默认值
        self.input_shape = [
            1 if dim == 'batch' else  # batch size = 1
            3 if dim == 'channels' else  # RGB channels = 3
            640 if dim in ['height', 'width'] else  # 默认图像大小 640x640
            dim for dim in input_shape
        ]
        self.img_size = (self.input_shape[2], self.input_shape[3])  # (height, width)
        print(f"✨ 模型输入: {self.input_name} - {self.input_shape}")
        print(f"✨ 模型输出: {[output.name for output in self.session.get_outputs()]}")

    def preprocess(self, image):
        # 保存原始图像尺寸
        self.orig_h, self.orig_w = image.shape[:2]
        
        # 直接缩放到640x640
        resized = cv2.resize(image, (640, 640))
        
        # 转换为float32并归一化
        img = resized.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        return img

    def predict(self, image):
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 模型推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return outputs

    def recognize_objects(self, frame):
        # 获取模型输出
        outputs = self.predict(frame)
        
        # 转置预测结果，使其形状为 (8400, 43)
        outputs = outputs[0].transpose(1, 0)
        
        # 处理预测结果
        boxes = outputs[:, :4]  # 边界框坐标 (x, y, w, h)
        scores = outputs[:, 4]  # 目标置信度
        class_probs = outputs[:, 5:39]  # 类别概率 (只取前34个类别)
        
        # 将边界框坐标归一化到 [0, 1] 范围
        boxes = boxes / 640.0  # 直接除以输入尺寸
        
        # 应用sigmoid到置信度和类别概率
        scores = sigmoid(scores)
        class_probs = sigmoid(class_probs)
        
        # 获取最高类别概率和对应的类别
        class_scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        
        # 计算最终置信度（使用乘积）
        scores = scores * class_scores
        
        # 打印调试信息
        print(f"\n预测结果分析:")
        print(f"置信度范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"类别概率范围: [{class_scores.min():.4f}, {class_scores.max():.4f}]")
        
        # 根据置信度过滤
        mask = scores > self.config["confidence_threshold"]
        if not mask.any():
            print("没有检测到任何目标")
            return []
        
        # 只保留高置信度的预测
        boxes = boxes[mask]
        scores = scores[mask, None]  # 添加一个维度以匹配后续操作
        class_probs = class_probs[mask]
        class_ids = class_ids[mask]
        
        # 将中心点坐标和宽高转换为左上角和右下角坐标
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        # 组合预测结果
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        predictions = np.concatenate([boxes, scores, class_probs], axis=1)
        
        # 应用非极大值抑制
        detections = non_max_suppression(predictions, self.class_names, self.config)
        
        # 还原到原始图像尺寸
        for det in detections:
            bbox = det['bbox']
            # 还原到原始图像尺寸
            bbox[0] = bbox[0] * self.orig_w
            bbox[1] = bbox[1] * self.orig_h
            bbox[2] = bbox[2] * self.orig_w
            bbox[3] = bbox[3] * self.orig_h
            
            # 确保坐标在图像范围内
            bbox[0] = max(0, min(bbox[0], self.orig_w))
            bbox[1] = max(0, min(bbox[1], self.orig_h))
            bbox[2] = max(0, min(bbox[2], self.orig_w))
            bbox[3] = max(0, min(bbox[3], self.orig_h))
            
            # 转换为整数
            bbox[:] = [int(x) for x in bbox]
        
        return detections

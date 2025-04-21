import os
import cv2
import sys
import signal
import argparse
import numpy as np
import onnxruntime
import yaml

def signal_handler(sig, frame):
    print("\n🛑 正在退出程序...")
    cv2.destroyAllWindows()
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description='麻将牌识别测试')
    parser.add_argument('--image', type=str, default='data/templates/1.png', help='要识别的图片路径')
    parser.add_argument('--model', type=str, default='data/yolov8/train/train/weights/best.onnx', help='ONNX模型路径')
    parser.add_argument('--config', type=str, default='config/config_train.yaml', help='配置文件路径（含类别名）')
    return parser.parse_args()

def load_class_names(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['class_names'] if 'class_names' in cfg else []

def draw_results(image, results):
    for obj in results:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        label = obj["label"]
        score = obj["score"]

        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} {score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image

def nms(boxes, scores, iou_threshold):
    # 按置信度降序排序
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # 保留置信度最高的框
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # 计算IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area1 + area2 - inter)
        
        # 保留IoU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

class VisionPredictor:
    def __init__(self, model_path, config_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # (1, 3, 640, 640)
        self.class_names = load_class_names(config_path)

    def preprocess(self, image):
        h0, w0 = image.shape[:2]
        input_h, input_w = self.input_shape[2], self.input_shape[3]

        image_resized = cv2.resize(image, (input_w, input_h))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_input = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        image_input = np.expand_dims(image_input, axis=0)
        return image_input, w0, h0

    def postprocess(self, outputs, orig_w, orig_h):
        # 获取模型输出并调整维度顺序
        predictions = outputs[0][0].transpose()  # (8400, 32)
        print(f"\n调试信息:")
        print(f"  原始预测形状: {outputs[0].shape}")
        print(f"  调整后预测形状: {predictions.shape}")
        print(f"  类别数量: {len(self.class_names)}")
        
        # 提取边界框和置信度
        boxes = predictions[:, :4]  # 前4个是边界框坐标
        scores = predictions[:, 4:5]  # 第5个是置信度
        class_probs = predictions[:, 5:32]  # 后面是类别概率
        
        print(f"  边界框形状: {boxes.shape}")
        print(f"  置信度形状: {scores.shape}")
        print(f"  类别概率形状: {class_probs.shape}")
        
        # 应用sigmoid激活函数
        scores = 1 / (1 + np.exp(-scores))
        class_probs = 1 / (1 + np.exp(-class_probs))
        
        # 获取最高类别概率
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = np.max(class_probs, axis=1)
        
        print(f"  类别ID范围: [{np.min(class_ids)}, {np.max(class_ids)}]")
        print(f"  类别概率范围: [{np.min(class_scores):.4f}, {np.max(class_scores):.4f}]")
        print(f"  置信度范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        
        # 计算最终置信度
        scores = scores.squeeze() * class_scores
        
        # 过滤低置信度的预测
        mask = scores > 0.3  # 降低置信度阈值
        if not mask.any():
            return []
            
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        print(f"  过滤后数量: {len(boxes)}")
        print(f"  过滤后类别ID: {class_ids.tolist()}")
        
        # 将xywh格式转换为xyxy格式
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # 将边界框坐标还原到原始图像尺寸
        scale_x = orig_w / self.input_shape[3]
        scale_y = orig_h / self.input_shape[2]
        
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        
        # 确保坐标在图像范围内
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        
        # 转换为整数坐标
        boxes = boxes.astype(np.int32)
        
        # 应用NMS
        keep = nms(boxes, scores, iou_threshold=0.45)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        
        print(f"  NMS后数量: {len(boxes)}")
        
        # 构建结果列表
        results = []
        for i in range(len(boxes)):
            if class_ids[i] < len(self.class_names):  # 确保类别ID在有效范围内
                results.append({
                    "bbox": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "label": self.class_names[class_ids[i]]
                })
            
        return results

    def recognize_objects(self, image):
        input_tensor, orig_w, orig_h = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        print("\n模型输出信息:")
        print(f"  输出数量: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  输出 {i} 形状: {output.shape}")
        results = self.postprocess(outputs, orig_w, orig_h)
        return results

def main():
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"❌ 测试图像不存在: {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return

    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法加载图像: {args.image}")
        return
    print(f"✅ 成功加载图像: {image.shape}")

    predictor = VisionPredictor(args.model, args.config)
    results = predictor.recognize_objects(image)

    print("\n🎯 识别结果:")
    if not results:
        print("  ⚠️ 没有检测到任何物体")
    else:
        for i, obj in enumerate(results):
            print(f"  {i+1}. 类别: {obj['label']}, 置信度: {obj['score']:.4f}, 位置: {obj['bbox']}")

    result_image = draw_results(image.copy(), results)
    out_path = f"result_{os.path.basename(args.image)}"
    cv2.imwrite(out_path, result_image)
    print(f"\n💾 已保存识别结果图像到: {out_path}")

    cv2.imshow("识别结果预览", result_image)
    print("📷 按任意键退出（5秒自动关闭）...")
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import time

# 扑克牌类别信息 - 从COCO标注文件获取
POKER_CLASSES = [
    "objects",  # 类别ID为0
    "10f", "10h", "10m", "10s", "2f", "2h", "2m", "2s", "3f", 
    "3h", "3m", "3s", "4f", "4h", "4m", "4s", "5f", "5h", "5m", 
    "5s", "6f", "6h", "6m", "6s", "7f", "7h", "7m", "7s", "8f", 
    "8h", "8m", "8s", "9f", "9h", "9m", "9s", "Af", "Ah", "Am", 
    "As", "Jf", "Jh", "Jm", "Js", "Kf", "Kh", "Km", "Ks", "Qf", 
    "Qh", "Qm", "Qs"
]

def softmax(x, axis=None):
    """
    计算softmax值
    
    Args:
        x: 输入数组
        axis: 计算softmax的轴
        
    Returns:
        输出数组，softmax结果
    """
    # 减去最大值以提高数值稳定性
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def load_model(model_path):
    """加载ONNX模型
    
    Args:
        model_path: ONNX模型路径
        
    Returns:
        onnxruntime会话对象
    """
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
def preprocess_image(image_input):
    """
    预处理图像
    
    Args:
        image_input: 图像路径或已加载的OpenCV图像对象
        
    Returns:
        tuple: (预处理后的图像数组, 原始图像, 原始图像尺寸)
    """
    # 检查输入类型
    if isinstance(image_input, str):
        # 如果是路径字符串，读取图像
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"无法读取图像: {image_input}")
    else:
        # 否则假设是已加载的OpenCV图像
        img = image_input
    
    # 保存原始图像用于可视化
    original_image = img.copy()
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 调整图像大小到模型输入大小 (通常是448x448)
    resized_img = cv2.resize(img, (448, 448))
    
    # 确保data/temp目录存在
    os.makedirs("data/temp", exist_ok=True)
    
    # 如果是从路径加载的图像，保存压缩后的图像
    resized_image_path = None
    if isinstance(image_input, str):
        # 提取原始图像名称
        base_name = os.path.basename(image_input)
        base_name = os.path.splitext(base_name)[0]
        timestamp = int(time.time())
        resized_image_path = f"data/temp/{base_name}_resized_{timestamp}.png"
        cv2.imwrite(resized_image_path, resized_img)
    
    # 转换为浮点数并归一化
    img = resized_img.astype(np.float32) / 255.0
    
    # 转换为RGB (从BGR)
    img = img[:, :, ::-1]
    
    # 转置为模型需要的格式 [1, 3, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img, original_image, resized_image_path

def run_inference(model, img_input):
    """
    运行模型推理
    
    Args:
        model: 加载的ONNX模型
        img_input: 预处理后的输入图像张量
        
    Returns:
        tuple: (检测框, logits)
    """
    # 获取输入名称
    input_name = model.get_inputs()[0].name
    print(f"模型输入名称: {input_name}, 输入形状: {img_input.shape}")
    
    # 执行推理
    print("开始执行推理...")
    outputs = model.run(None, {input_name: img_input})
    
    # 解析输出
    boxes = outputs[0]  # 检测框 [batch_size, num_boxes, 4]
    logits = outputs[1]  # 类别logits [batch_size, num_boxes, num_classes]
    
    print(f"输出形状 - boxes: {boxes.shape}, logits: {logits.shape}")
    
    return boxes, logits

def post_process(boxes, logits, image_shape, threshold=0.1):
    """
    后处理模型输出
    
    Args:
        boxes: 模型输出的边界框 [batch_size, num_boxes, 4]
        logits: 模型输出的类别logits [batch_size, num_boxes, num_classes]
        image_shape: 原始图像尺寸 (height, width)
        threshold: 置信度阈值
        
    Returns:
        tuple: (过滤后的框坐标, 类别ID, 置信度分数)
    """
    # 获取图像尺寸
    height, width = image_shape[:2]
    
    # 只处理第一个批次的结果
    boxes = boxes[0]  # [num_boxes, 4]
    logits = logits[0]  # [num_boxes, num_classes]
    
    # 应用softmax获取概率
    probs = softmax(logits, axis=1)
    
    # 获取每个框的最高置信度及其对应的类别
    max_probs = np.max(probs, axis=1)
    class_ids = np.argmax(probs, axis=1)
    
    # 过滤置信度低的检测结果
    keep_indices = np.where(max_probs > threshold)[0]
    filtered_boxes = boxes[keep_indices]
    filtered_classes = class_ids[keep_indices]
    filtered_probs = max_probs[keep_indices]
    
    # 将框转换为绝对坐标 [x1, y1, x2, y2]
    processed_boxes = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        # 确保坐标是有效的浮点数
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            continue
            
        # 转换为绝对坐标
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # 确保坐标有效，x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # 确保边界框有最小尺寸
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
            
        processed_boxes.append([x1, y1, x2, y2])
    
    return processed_boxes, filtered_classes, filtered_probs

def visualize_results(original_image, boxes, class_ids, scores, threshold=0.1, image_name=None):
    """
    可视化检测结果
    
    Args:
        original_image: 原始图像（OpenCV格式，BGR）
        boxes: 检测框坐标
        class_ids: 类别ID
        scores: 置信度分数
        threshold: 置信度阈值
        image_name: 原始图像名称，用于生成结果文件名
        
    Returns:
        PIL图像对象，检测到的目标数量，结果图像保存路径
    """
    # 转换BGR到RGB
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    detected_count = 0
    
    # 对每个检测结果绘制框和标签
    for box, class_id, score in zip(boxes, class_ids, scores):
        if score > threshold:
            try:
                detected_count += 1
                # 解析坐标
                x1, y1, x2, y2 = box
                
                # 确保坐标是整数
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 确保x1 < x2, y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # 绘制边界框
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                
                # 绘制标签
                label = f"{POKER_CLASSES[class_id]}: {score:.2f}"
                draw.text((x1, y1-10), label, fill="red")
            except Exception as e:
                print(f"绘制边界框错误: {e}, 坐标: {box}")
                continue
    
    # 确保data/temp目录存在
    os.makedirs("data/temp", exist_ok=True)
    
    # 保存结果图像
    result_path = None
    if detected_count > 0:
        # 使用时间戳和原始图像名称生成唯一文件名
        timestamp = int(time.time())
        
        # 提取原始图像名称
        if image_name:
            # 移除路径和扩展名
            base_name = os.path.basename(image_name)
            base_name = os.path.splitext(base_name)[0]
            result_path = f"data/temp/{base_name}_result_{timestamp}.png"
        else:
            result_path = f"data/temp/detection_{timestamp}.png"
            
        pil_image.save(result_path)
    
    return pil_image, detected_count, result_path

def print_top_predictions(class_ids, scores, boxes, top_k=5):
    """
    打印置信度最高的前k个预测结果
    
    Args:
        class_ids: 类别ID数组
        scores: 置信度分数数组
        boxes: 检测框坐标数组
        top_k: 显示前k个预测，默认5
    """
    # 打印前k个预测
    if len(scores) == 0:
        print("未检测到任何预测")
        return
    
    print(f"前 {min(top_k, len(scores))} 个预测:")
    # 按置信度排序
    indices = np.argsort(scores)[::-1][:top_k]
    
    for i, idx in enumerate(indices):
        cls_id = class_ids[idx]
        cls_name = POKER_CLASSES[cls_id] if cls_id < len(POKER_CLASSES) else f"class_{cls_id}"
        print(f"类别: {cls_name}, 置信度: {scores[idx]:.4f}, 坐标: {boxes[idx]}")

def detect_image(model, image_path, conf_threshold=0.1):
    """
    检测单张图像
    
    Args:
        model: 加载的ONNX模型
        image_path: 图像路径
        conf_threshold: 置信度阈值
        
    Returns:
        boxes: 检测框坐标
        class_ids: 类别ID
        scores: 置信度分数
        result_img: 结果图像（PIL格式）
    """
    # 读取和预处理图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return [], [], [], None, 0, None, None
    
    preprocessed_img, original_image, resized_image_path = preprocess_image(image_path)
    
    # 如果有压缩后的图像路径，打印出来
    if resized_image_path:
        print(f"压缩后的图像已保存至: {resized_image_path}")
    
    # 运行推理
    boxes, logits = run_inference(model, preprocessed_img)
    
    # 后处理结果
    processed_boxes, class_ids, scores = post_process(boxes, logits, original_image.shape, conf_threshold)
    
    # 打印预测结果
    print_top_predictions(class_ids, scores, processed_boxes)
    
    # 可视化结果
    result_img, detected_count, result_path = visualize_results(
        original_image, processed_boxes, class_ids, scores, 
        conf_threshold, image_name=image_path
    )
    
    return processed_boxes, class_ids, scores, result_img, detected_count, result_path, resized_image_path

def batch_detect(model, image_paths, conf_threshold=0.1):
    """
    批量检测多张图像
    
    Args:
        model: 加载的ONNX模型
        image_paths: 图像路径列表
        conf_threshold: 置信度阈值
    """
    results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] 处理图像: {img_path}")
        
        # 检测单张图像
        boxes, class_ids, scores, result_img, detected_count, result_path, resized_path = detect_image(model, img_path, conf_threshold)
        
        # 添加结果摘要
        results.append({
            "image_path": img_path,
            "detected_count": detected_count,
            "result_path": result_path,
            "resized_path": resized_path
        })
        
        # 打印结果
        if detected_count > 0:
            print(f"检测到 {detected_count} 个置信度 > {conf_threshold} 的目标")
            print(f"结果已保存至: {result_path}")
        else:
            print(f"检测到 {detected_count} 个置信度 > {conf_threshold} 的目标")
    
    return results

if __name__ == "__main__":
    print("正在执行批量检测...")
    print("正在加载模型...\n")
    
    # 加载模型
    model = load_model("data/inference_model.onnx")
    
    # 单图像检测
    # boxes, class_ids, scores, result_img, detected_count, result_path = detect_image(model, "data/templates/test.png")
    
    # 批量检测
    image_paths = [
        "data/templates/test.png"
    ]
    
    results = batch_detect(model, image_paths)
    
    # 打印结果摘要
    print("\n检测结果摘要:")
    for i, result in enumerate(results):
        if result["detected_count"] > 0:
            print(f"图像 {i+1}: 检测到 {result['detected_count']} 个目标，保存至 {result['result_path']}")
        else:
            print(f"图像 {i+1}: 未检测到目标")

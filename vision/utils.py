import cv2
import yaml
import numpy as np

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def non_max_suppression(predictions, class_names, config):
    results = []
    conf_thres = config["confidence_threshold"]
    nms_thres = config["nms_threshold"]
    
    print(f"\nNMS调试信息:")
    print(f"类别数量: {len(class_names)}")
    print(f"预测结果形状: {predictions.shape}")
    print(f"置信度阈值: {conf_thres}")
    print(f"NMS阈值: {nms_thres}")
    
    # 找出所有置信度大于阈值的预测
    valid_mask = predictions[:, 4] > conf_thres
    valid_preds = predictions[valid_mask]
    
    print(f"置信度大于阈值的预测数量: {len(valid_preds)}")
    if len(valid_preds) > 0:
        print(f"有效预测的置信度范围: [{valid_preds[:, 4].min()}, {valid_preds[:, 4].max()}]")
    
    if len(valid_preds) == 0:
        print("没有预测结果超过置信度阈值")
        return results
    
    # 对每个类别分别进行NMS
    for cls_idx in range(len(class_names)):
        # 获取当前类别的预测
        cls_mask = valid_preds[:, 5 + cls_idx] > conf_thres
        cls_preds = valid_preds[cls_mask]
        
        if len(cls_preds) == 0:
            continue
        
        print(f"\n处理类别 {class_names[cls_idx]}:")
        print(f"  该类别的预测数量: {len(cls_preds)}")
        
        # 按置信度排序
        scores = cls_preds[:, 4] * cls_preds[:, 5 + cls_idx]
        order = scores.argsort()[::-1]
        cls_preds = cls_preds[order]
        
        print(f"  最高置信度: {scores[order[0]]:.4f}")
        
        # 进行NMS
        while len(cls_preds) > 0:
            # 选择置信度最高的预测
            best_pred = cls_preds[0]
            best_box = best_pred[:4]
            best_score = best_pred[4] * best_pred[5 + cls_idx]
            
            # 添加到结果中
            results.append({
                "bbox": best_box.tolist(),
                "score": float(best_score),
                "label": class_names[cls_idx]
            })
            
            # 计算与剩余预测的IoU
            if len(cls_preds) == 1:
                break
                
            ious = np.array([compute_iou(best_box, pred[:4]) for pred in cls_preds[1:]])
            
            # 移除IoU大于阈值的预测
            keep = ious < nms_thres
            cls_preds = cls_preds[1:][keep]
    
    print(f"\n最终检测到的目标数量: {len(results)}")
    return results

def draw_results(image, results):
    """在图像上绘制检测结果"""
    debug_image = image.copy()
    
    # 定义颜色映射
    colors = {
        '一': (255, 0, 0),    # 红色
        '二': (0, 255, 0),    # 绿色
        '三': (0, 0, 255),    # 蓝色
        '四': (255, 255, 0),  # 黄色
        '五': (255, 0, 255),  # 紫色
        '六': (0, 255, 255),  # 青色
        '七': (128, 0, 0),    # 深红
        '八': (0, 128, 0),    # 深绿
        '九': (0, 0, 128),    # 深蓝
        '中': (255, 128, 0),  # 橙色
        '发': (128, 255, 0),  # 黄绿
        '白': (255, 255, 255),# 白色
        '东': (128, 0, 255),  # 紫色
        '南': (255, 0, 128),  # 粉色
        '西': (0, 128, 255),  # 天蓝
        '北': (128, 128, 0)   # 橄榄
    }
    
    # 默认颜色
    default_color = (200, 200, 200)  # 灰色
    
    for result in results:
        # 获取边界框坐标
        x1, y1, x2, y2 = result['bbox']
        label = result['label']
        score = result['score']
        
        # 获取颜色
        color = colors.get(label[0], default_color)  # 使用第一个字作为键
        
        # 绘制边界框
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label_text = f"{label}: {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(debug_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(debug_image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return debug_image

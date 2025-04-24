"""
扑克牌OCR识别模块
职责：识别屏幕中的扑克牌和操作按钮
"""

import os
import re
import cv2
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from paddleocr import PaddleOCR

# ============ 常量定义 ============
DEBUG_DIR = "data/debug"
PREVIEW_IMG = os.path.join(DEBUG_DIR, "regions_preview.png")
RESULT_IMG = os.path.join(DEBUG_DIR, "ocr_result.png")
os.makedirs(DEBUG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# 有效卡牌值
VALID_CARD = set(["A","2","3","4","5","6","7","8","9","10","J","Q","K"])

# 有效操作
VALID_ACTION = set(["弃牌","加注","让牌","跟注"])

# 识别区域比例
REGION_RATIOS = {
    "PUBLIC_REGION": (0.2, 0.5, 0.8, 0.6),  # 公共牌区域
    "HAND_REGION":   (0.3, 0.88, 0.7, 0.98), # 手牌区域
    "CLICK_REGION":  (0.3, 0.65, 0.7, 0.85)  # 操作按钮区域
}

# ============ OCR 模型 ============
# 英文模型：用于识别卡牌
ocr_en = PaddleOCR(use_angle_cls=False, lang="en",
                   det_db_thresh=0.15,  # 检测阈值
                   det_db_box_thresh=0.15,
                   det_db_unclip_ratio=1.5,  # 文本框扩张比例
                   det_limit_side_len=2000)

# 中文模型：用于识别操作按钮
ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch",
                   det_db_thresh=0.15,
                   det_db_box_thresh=0.15,
                   det_db_unclip_ratio=1.5,
                   det_limit_side_len=2000)

# ============ 工具函数 ============
def get_regions(w: int, h: int) -> Dict[str, Tuple[int,int,int,int]]:
    """获取识别区域"""
    pad = 10
    regs = {}
    for name, (l, t, r, b) in REGION_RATIOS.items():
        x1 = max(0, int(w * l) - pad)
        y1 = max(0, int(h * t) - pad)
        x2 = min(w, int(w * r) + pad)
        y2 = min(h, int(h * b) + pad)
        regs[name] = (x1, y1, x2, y2)
    return regs

def preprocess_card(roi: np.ndarray) -> np.ndarray:
    """预处理卡牌图像"""
    # 1. 添加边框
    h, w = roi.shape[:2]
    roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    
    # 2. 放大图像
    roi = cv2.resize(roi, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    
    # 3. 提取灰度图和红色通道
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    red  = roi[:, :, 2]
    
    # 4. 融合通道
    fused = cv2.max(gray, red)
    
    # 5. 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(fused)

def extract_cards(lines: List) -> List[Tuple[str,float,List]]:
    """提取卡牌信息"""
    if not lines:  # 如果识别结果为空
        return []
        
    cards = []
    for box, (txt, conf) in lines:
        t = txt.strip().upper()
        if t not in VALID_CARD:
            m = re.search(r'([A-Z]|\d+)', t)
            if m:
                t = m.group(1)
                if t == '0':
                    t = '10'
        if t in VALID_CARD:
            cards.append((t, conf, box))
    cards.sort(key=lambda c: np.mean([pt[0] for pt in c[2]]))
    return cards

def is_action(text: str) -> bool:
    """判断是否为有效操作"""
    return text in VALID_ACTION or "底池" in text

# ============ 主要功能 ============
def DualChannelPokerOCR(img: np.ndarray) -> Dict:
    """双通道扑克牌OCR识别"""
    # 1. 输入检查
    if img is None:
        logging.error("无法读取图像")
        return {"success": False, "error": "无法读取图像"}

    # 2. 计算识别区域
    h, w = img.shape[:2]
    regs = get_regions(w, h)

    # 3. 可视化ROIs
    vis = img.copy()
    for name, (x1, y1, x2, y2) in regs.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, name, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(PREVIEW_IMG, vis)

    # 4. 初始化结果
    result = {"success": True, "publicCards": [], "handCards": [], "actions": []}
    out = vis.copy()

    # 5. 识别公共牌
    x1, y1, x2, y2 = regs["PUBLIC_REGION"]
    roi_pub = preprocess_card(img[y1:y2, x1:x2])
    res_pub = ocr_en.ocr(roi_pub, cls=False)
    
    # 调试信息
    if res_pub and res_pub[0]:
        print("公牌原始识别结果:", [(txt, conf) for box, (txt, conf) in res_pub[0]])
    
    for t, conf, box in extract_cards(res_pub[0] if res_pub else []):
        abs_pts = [[int(pt[0]/2 + x1 - 10), int(pt[1]/2 + y1 - 10)] for pt in box]
        result["publicCards"].append(t)  # 只保存卡牌值
        cx, cy = int(np.mean([p[0] for p in abs_pts])), int(np.mean([p[1] for p in abs_pts]))
        cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 6. 识别手牌
    x1, y1, x2, y2 = regs["HAND_REGION"]
    roi_hand = preprocess_card(img[y1:y2, x1:x2])
    res_hand = ocr_en.ocr(roi_hand, cls=False)
    
    # 调试信息
    if res_hand and res_hand[0]:
        print("手牌原始识别结果:", [(txt, conf) for box, (txt, conf) in res_hand[0]])
    
    for t, conf, box in extract_cards(res_hand[0] if res_hand else []):
        abs_pts = [[int(pt[0]/2 + x1 - 10), int(pt[1]/2 + y1 - 10)] for pt in box]
        result["handCards"].append(t)  # 只保存卡牌值
        cx, cy = int(np.mean([p[0] for p in abs_pts])), int(np.mean([p[1] for p in abs_pts]))
        cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 7. 识别按钮
    x1, y1, x2, y2 = regs["CLICK_REGION"]
    roi_click = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_click = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(roi_click)
    res_click = ocr_ch.ocr(roi_click, cls=False)
    for box, (txt, conf) in (res_click[0] if res_click else []):
        t = txt.strip()
        if is_action(t):
            abs_pts = [[pt[0] + x1, pt[1] + y1] for pt in box]
            result["actions"].append({"action": t, "box": abs_pts})  # 保留坐标信息
            cx, cy = int(np.mean([p[0] for p in abs_pts])), int(np.mean([p[1] for p in abs_pts]))
            cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # 8. 保存结果
    cv2.imwrite(RESULT_IMG, out)
    return result

if __name__ == "__main__":
    # 读取测试图像
    img = cv2.imread("data/templates/test.png")
    if img is None:
        logging.error("无法读取测试图像")
        exit(1)
        
    # 执行OCR识别
    result = DualChannelPokerOCR(img)
    print("识别结果：", json.dumps(result, ensure_ascii=False))

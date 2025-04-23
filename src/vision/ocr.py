import os
import cv2
import json
import logging
import re
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, Tuple, List

# ============ 输出目录 ============
DEBUG_DIR = "data/debug"
PREVIEW_IMG = os.path.join(DEBUG_DIR, "regions_preview.png")
RESULT_IMG = os.path.join(DEBUG_DIR, "ocr_result.png")
os.makedirs(DEBUG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============ 区域比例 ============
REGION_RATIOS = {
    "PUBLIC_REGION": (0.2, 0.5, 0.8, 0.6),
    "HAND_REGION":   (0.3, 0.88, 0.7, 0.98),
    "CLICK_REGION":  (0.3, 0.65, 0.7, 0.85)
}

VALID_CARD   = set(["A","2","3","4","5","6","7","8","9","10","J","Q","K"])
VALID_ACTION = set(["弃牌","加注","让牌","跟注"])

def is_action(text: str) -> bool:
    return text in VALID_ACTION or "底池" in text

# ============ OCR 模型 ============
ocr_en = PaddleOCR(use_angle_cls=True, lang="en",
                   det_db_thresh=0.15, det_db_box_thresh=0.15,
                   det_db_unclip_ratio=1.5, det_limit_side_len=2000)
ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch",
                   det_db_thresh=0.15, det_db_box_thresh=0.15,
                   det_db_unclip_ratio=1.5, det_limit_side_len=2000)

def get_regions(w: int, h: int) -> Dict[str, Tuple[int,int,int,int]]:
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
    h, w = roi.shape[:2]
    roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    roi = cv2.resize(roi, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    red  = roi[:, :, 2]
    fused = cv2.max(gray, red)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(fused)

def extract_cards(lines: List) -> List[Tuple[str,float,List]]:
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

def DualChannelPokerOCR(img: np.ndarray) -> Dict:
    """
    对外接口：传入已读取的图像 img，返回识别结果 dict。
    内部不再打印，调用方可拿到返回值自行处理。
    """
    if img is None:
        logging.error("无法读取图像")
        return {"success": False, "error": "无法读取图像"}

    h, w = img.shape[:2]
    regs = get_regions(w, h)

    # 可视化 ROIs
    vis = img.copy()
    for name, (x1, y1, x2, y2) in regs.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, name, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(PREVIEW_IMG, vis)

    result = {"success": True, "publicCards": [], "handCards": [], "actions": []}
    out = vis.copy()

    # 公共牌区 OCR
    x1, y1, x2, y2 = regs["PUBLIC_REGION"]
    roi_pub = preprocess_card(img[y1:y2, x1:x2])
    res_pub = ocr_en.ocr(roi_pub, cls=True)
    for t, conf, box in extract_cards(res_pub[0] if res_pub else []):
        abs_pts = [[int(pt[0]/2 + x1 - 10), int(pt[1]/2 + y1 - 10)] for pt in box]
        result["publicCards"].append({"action": t, "box": abs_pts})
        cx, cy = int(np.mean([p[0] for p in abs_pts])), int(np.mean([p[1] for p in abs_pts]))
        cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 手牌区 OCR
    x1, y1, x2, y2 = regs["HAND_REGION"]
    roi_hand = preprocess_card(img[y1:y2, x1:x2])
    res_hand = ocr_en.ocr(roi_hand, cls=True)
    for t, conf, box in extract_cards(res_hand[0] if res_hand else []):
        abs_pts = [[int(pt[0]/2 + x1 - 10), int(pt[1]/2 + y1 - 10)] for pt in box]
        result["handCards"].append({"action": t, "box": abs_pts})
        cx, cy = int(np.mean([p[0] for p in abs_pts])), int(np.mean([p[1] for p in abs_pts]))
        cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 按钮区 OCR
    x1, y1, x2, y2 = regs["CLICK_REGION"]
    roi_click = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_click = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(roi_click)
    res_click = ocr_ch.ocr(roi_click, cls=True)
    for box, (txt, conf) in (res_click[0] if res_click else []):
        t = txt.strip()
        if is_action(t):
            abs_pts = [[pt[0] + x1, pt[1] + y1] for pt in box]
            result["actions"].append({"action": t, "box": abs_pts})
            cx, cy = int(np.mean([p[0] for p in abs_pts])), int(np.mean([p[1] for p in abs_pts]))
            cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imwrite(RESULT_IMG, out)
    return result

def recognize_cards(image_path: str) -> Dict:
    """
    对外接口：传入图像路径，返回识别结果 dict。
    内部调用 DualChannelPokerOCR 处理图像。
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": f"无法读取图像: {image_path}"}
    return DualChannelPokerOCR(img)

if __name__ == "__main__":
    # 示例：既打印结果，又可在脚本外拿到返回值
    img = cv2.imread("data/templates/5.png")
    result = DualChannelPokerOCR(img)
    # 打印 JSON
    print("识别结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

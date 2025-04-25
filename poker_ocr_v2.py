# poker_ocr_v3.py（增强版，支持低饱和花色识别）
import os
import cv2
import json
import logging
import re
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, Tuple, List

DEBUG_DIR = "data/debug"
PREVIEW_IMG = os.path.join(DEBUG_DIR, "regions_preview.png")
RESULT_IMG = os.path.join(DEBUG_DIR, "ocr_result.png")
os.makedirs(DEBUG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

REGION_RATIOS = {
    "PUBLIC_REGION": (0.15, 0.5, 0.85, 0.601),
    "HAND_REGION":   (0.3, 0.88, 0.7, 0.981)
}

VALID_CARD = set(["A","2","3","4","5","6","7","8","9","10","J","Q","K"])

ocr_en = PaddleOCR(use_angle_cls=False, lang="en", det_db_thresh=0.1, det_db_box_thresh=0.1,
                   det_db_unclip_ratio=2.0, det_limit_side_len=2000,
                   rec_char_dict_path='/opt/homebrew/Caskroom/miniconda/base/envs/gameai/lib/python3.8/site-packages/paddleocr/ppocr/utils/en_dict.txt',
                   det_model_dir='/Users/mac/.paddleocr/whl/det/en/en_PP-OCRv4_det_infer',
                   rec_model_dir='/Users/mac/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer')

ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch", det_db_thresh=0.1, det_db_box_thresh=0.1,
                   det_db_unclip_ratio=2.0, det_limit_side_len=2000)

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

def extract_cards(lines: List) -> List[Tuple[str,float,List]]:
    cards = []
    for box, (txt, conf) in lines:
        t = txt.strip().upper()
        if t in VALID_CARD:
            cards.append((t, conf, box))
            continue
        m = re.search(r'([A-Z]|\d+)', t)
        if m:
            t = m.group(1)
            if t == '0': t = '10'
            if t in VALID_CARD:
                cards.append((t, conf, box))
                continue
        if t == '1' or t == 'I':
            cards.append(('1', conf, box))
        elif t == 'O':
            cards.append(('0', conf, box))
    cards.sort(key=lambda c: np.mean([pt[0] for pt in c[2]]))
    return cards

def recognize_suit_by_color_and_shape(img: np.ndarray) -> str:
    img = cv2.resize(img, (32, 32))
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=30)  # 提亮增强
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])
    if s_mean < 50 and v_mean < 80:
        color = "black"
    elif (h_mean < 15 or h_mean > 150) and s_mean > 15:
        color = "red"
    elif 30 < h_mean < 90:
        color = "green"
    else:
        color = "unknown"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "?"
    c = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    if color == "black":
        return "♠" if hu[0] < 0.23 else "♣"
    elif color == "red":
        return "♥" if hu[0] < 0.19 else "♦"
    elif color == "green":
        return "♣"
    return "?"

def DualChannelPokerOCR(img: np.ndarray) -> Dict:
    if img is None:
        logging.error("无法读取图像")
        return {"success": False, "error": "无法读取图像"}

    h, w = img.shape[:2]
    regs = get_regions(w, h)
    vis = img.copy()
    for name, (x1, y1, x2, y2) in regs.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, name, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(PREVIEW_IMG, vis)

    result = {"success": True, "publicCards": [], "handCards": []}
    out = vis.copy()

    # 公牌识别
    x1, y1, x2, y2 = regs["PUBLIC_REGION"]
    roi_pub = img[y1:y2, x1:x2]
    res_pub = ocr_en.ocr(roi_pub, cls=False)
    for t, conf, box in extract_cards(res_pub[0] if res_pub else []):
        cx, cy = int(np.mean([p[0] for p in box])) + x1, int(np.mean([p[1] for p in box])) + y1
        sx1, sy1, sx2, sy2 = int(box[0][0]+x1), int(box[1][1]+y1), int(box[2][0]+x1), int(box[2][1]+y1)
        roi_suit = img[sy1:sy2+int((sy2-sy1)*0.3), sx1:sx2]  # 下移+扩展区域
        suit = recognize_suit_by_color_and_shape(roi_suit)
        result["publicCards"].append(f"{t}{suit}")
        cv2.putText(out, f"{t}{suit}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 手牌识别
    x1, y1, x2, y2 = regs["HAND_REGION"]
    roi_hand = img[y1:y2, x1:x2]
    res_hand = ocr_en.ocr(roi_hand, cls=False)
    for t, conf, box in extract_cards(res_hand[0] if res_hand else []):
        cx, cy = int(np.mean([p[0] for p in box])) + x1, int(np.mean([p[1] for p in box])) + y1
        sx1, sy1, sx2, sy2 = int(box[0][0]+x1), int(box[1][1]+y1), int(box[2][0]+x1), int(box[2][1]+y1)
        roi_suit = img[sy1:sy2+int((sy2-sy1)*0.3), sx1:sx2]  # 下移+扩展区域
        suit = recognize_suit_by_color_and_shape(roi_suit)
        result["handCards"].append(f"{t}{suit}")
        cv2.putText(out, f"{t}{suit}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    cv2.imwrite(RESULT_IMG, out)
    return result

def recognize_cards(image_path: str) -> Dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "hand_cards": [], "public_cards": [], "error": f"无法读取图像: {image_path}"}
    result = DualChannelPokerOCR(img)
    return {
        "success": result["success"],
        "hand_cards": result["handCards"],
        "public_cards": result["publicCards"],
        "error": result.get("error")
    }

if __name__ == "__main__":
    img = cv2.imread("data/templates/test.png")
    result = DualChannelPokerOCR(img)
    print("识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
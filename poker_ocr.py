#!/usr/bin/env python3
import os
import cv2
import json
import logging
import re
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, Tuple, List

# 输出目录
DEBUG_DIR = "data/debug"
PREVIEW_IMG = os.path.join(DEBUG_DIR, "regions_preview.png")
RESULT_IMG  = os.path.join(DEBUG_DIR, "ocr_result.png")
os.makedirs(DEBUG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 区域比例
REGION_RATIOS = {
    "PUBLIC_REGION": (0.2, 0.5, 0.8, 0.6),
    "HAND_REGION":   (0.3, 0.88, 0.7, 0.98),
    "CLICK_REGION":  (0.3, 0.65, 0.7, 0.85)
}

VALID_CARD = set(["A","2","3","4","5","6","7","8","9","10","J","Q","K"])
VALID_ACTION = set(["弃牌","加注","让牌","跟注"])

# OCR 实例
ocr_en = PaddleOCR(use_angle_cls=True, lang="en",
                   det_db_thresh=0.15, det_db_box_thresh=0.15,
                   det_db_unclip_ratio=1.5, det_limit_side_len=2000)
ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch",
                   det_db_thresh=0.15, det_db_box_thresh=0.15,
                   det_db_unclip_ratio=1.5, det_limit_side_len=2000)

def get_regions(w:int,h:int)->Dict[str,Tuple[int,int,int,int]]:
    pad = 10
    regs = {}
    for name,(l,t,r,b) in REGION_RATIOS.items():
        x1 = max(0, int(w*l) - pad)
        y1 = max(0, int(h*t) - pad)
        x2 = min(w, int(w*r) + pad)
        y2 = min(h, int(h*b) + pad)
        regs[name] = (x1,y1,x2,y2)
    return regs

def preprocess(roi:np.ndarray)->np.ndarray:
    """ROI 预处理：pad, resize, gray+red fusion, CLAHE"""
    h,w = roi.shape[:2]
    roi = cv2.copyMakeBorder(roi,10,10,10,10, cv2.BORDER_REPLICATE)
    roi = cv2.resize(roi, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    red  = roi[:,:,2]
    fused = cv2.max(gray, red)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(fused)

def extract_cards(lines:List)->List[Tuple[str,float,List]]:
    cards = []
    for box,(txt,conf) in lines:
        t = txt.strip().upper()
        if t not in VALID_CARD:
            m = re.search(r'([A-Z]|\d+)', t)
            if m:
                t = m.group(1)
                if t=='0': t='10'
        if t in VALID_CARD:
            cards.append((t,conf,box))
    # 排序
    cards.sort(key=lambda c: np.mean([pt[0] for pt in c[2]]))
    return cards

def extract_actions(lines:List)->List[str]:
    acts=[]
    for _,(txt,_) in lines:
        t=txt.strip()
        if t in VALID_ACTION:
            acts.append(t)
    return acts

def main():
    img = cv2.imread("data/templates/5.png")
    if img is None:
        logging.error("无法读取图像")
        return
    h,w = img.shape[:2]
    regs = get_regions(w,h)

    # 可视化 ROI
    vis = img.copy()
    for name,(x1,y1,x2,y2) in regs.items():
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(vis, name, (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imwrite(PREVIEW_IMG, vis)

    result = {"success":True, "publicCards":[], "handCards":[],"actions":[]}
    out = vis.copy()

    # 公共牌区 OCR
    x1,y1,x2,y2 = regs["PUBLIC_REGION"]
    roi = preprocess(img[y1:y2, x1:x2])
    res = ocr_en.ocr(roi, cls=True)
    cards = extract_cards(res[0] if res else [])
    vals = [c[0] for c in cards]
    result["publicCards"] = vals
    for v,_,box in cards:
        cx = int(np.mean([pt[0] for pt in box])//2 + x1)
        cy = int(np.mean([pt[1] for pt in box])//2 + y1)
        cv2.putText(out, v, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)

    # 手牌区 OCR
    x1,y1,x2,y2 = regs["HAND_REGION"]
    roi = preprocess(img[y1:y2, x1:x2])
    res = ocr_en.ocr(roi, cls=True)
    cards = extract_cards(res[0] if res else [])
    vals = [c[0] for c in cards]
    result["handCards"] = vals
    for v,_,box in cards:
        cx = int(np.mean([pt[0] for pt in box])//2 + x1)
        cy = int(np.mean([pt[1] for pt in box])//2 + y1)
        cv2.putText(out, v, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)

    # 按钮区 OCR
    x1,y1,x2,y2 = regs["CLICK_REGION"]
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi = clahe.apply(gray)
    res = ocr_ch.ocr(roi, cls=True)
    acts = extract_actions(res[0] if res else [])
    result["actions"] = acts
    for i,a in enumerate(acts):
        cv2.putText(out, a, (x1+10, y1+30+30*i),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

    cv2.imwrite(RESULT_IMG, out)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()

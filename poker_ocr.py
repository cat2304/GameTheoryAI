#!/usr/bin/env python3
# fixed_region_poker_ocr_dual_channel.py

import os
import cv2
import numpy as np
import logging
import json
import re
from paddleocr import PaddleOCR
from datetime import datetime
from typing import Tuple, Dict, List

# =============== OCR 配置 ===============
OCR_CONFIG = {
    'use_angle_cls':       True,
    'lang':                'en',
    'det_db_thresh':       0.05,
    'det_db_box_thresh':   0.05,
    'det_db_unclip_ratio': 1.5,
    'det_limit_side_len':  2000,
    'rec_char_dict_path':  '/opt/homebrew/Caskroom/miniconda/base/envs/gameai/lib/python3.8/site-packages/paddleocr/ppocr/utils/en_dict.txt'
}

# =============== 区域配置 ===============
REGION_RATIOS = {
    'PUBLIC_REGION': (0.2, 0.5, 0.8, 0.6),
    'HAND_REGION':   (0.3, 0.88, 0.7, 0.98)
}

# =============== 图像处理配置 ===============
IMG_CFG = {
    'resize_factor':    2,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size':  (8, 8),
    'blur_ksize':       3
}

# =============== 输出路径 ===============
DEBUG_DIR   = 'data/debug'
PREVIEW_IMG = os.path.join(DEBUG_DIR, 'regions_preview.png')
RESULT_IMG  = os.path.join(DEBUG_DIR, 'ocr_result.png')

# 有效牌值
VALID_CARD_VALUES = {'A','2','3','4','5','6','7','8','9','10','J','Q','K'}

def get_screen_regions(w: int, h: int) -> Dict[str,Tuple[int,int,int,int]]:
    regs = {}
    pad = 5
    for name,(l,t,r,b) in REGION_RATIOS.items():
        x1 = max(0, int(w*l)-pad)
        y1 = max(0, int(h*t)-pad)
        x2 = min(w, int(w*r)+pad)
        y2 = min(h, int(h*b)+pad)
        regs[name] = (x1,y1,x2,y2)
    return regs

class DualChannelPokerOCR:
    def __init__(self):
        os.makedirs(DEBUG_DIR, exist_ok=True)
        self.ocr = PaddleOCR(**OCR_CONFIG, show_log=False)
        self.logger = logging.getLogger("DualChannelPokerOCR")

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        # 灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 红色通道
        red  = roi[:,:,2]
        # 融合
        fused = cv2.max(gray, red)
        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=IMG_CFG['clahe_clip_limit'],
            tileGridSize=IMG_CFG['clahe_tile_size']
        )
        enhanced = clahe.apply(fused)
        # 轻度模糊
        return cv2.GaussianBlur(enhanced, (IMG_CFG['blur_ksize'],)*2, 0)

    def _extract(self, lines: List) -> List[Tuple[str,float,List]]:
        cards = []
        for box,(txt,conf) in lines:
            val = txt.strip().upper()
            if val not in VALID_CARD_VALUES:
                m = re.search(r'([A-Z]|\d+)', val)
                if m:
                    val = m.group(1)
                    if val=='0': val='10'
            if val in VALID_CARD_VALUES:
                cards.append((val, conf, box))
        return cards

    def recognize_region(self, roi_color: np.ndarray, region_name: str) -> List[Tuple[str,float,List]]:
        """
        对单个 ROI 进行识别：
        - 先在预处理通道跑 OCR
        - 如果是 HAND_REGION 且只识别到 <2 张牌，再用原图通道跑一次
        """
        # 放大
        roi_big = cv2.resize(
            roi_color, None,
            fx=IMG_CFG['resize_factor'],
            fy=IMG_CFG['resize_factor'],
            interpolation=cv2.INTER_LINEAR
        )

        # 通道 1：预处理
        proc = self._preprocess(roi_big)
        path1 = os.path.join(DEBUG_DIR, f"{region_name.lower()}_proc.png")
        cv2.imwrite(path1, proc)
        self.logger.debug(f"{region_name} 预处理图: {path1}")
        res1 = self.ocr.ocr(proc, cls=True)
        cards1 = self._extract(res1[0] if res1 else [])

        # 如果是手牌区且只识别到 0 or 1 张，再多试一次「原图通道」
        if region_name == "HAND_REGION" and len(cards1) < 2:
            path_raw = os.path.join(DEBUG_DIR, f"{region_name.lower()}_raw.png")
            cv2.imwrite(path_raw, roi_big)
            self.logger.debug(f"{region_name} 原图图: {path_raw}")
            res2 = self.ocr.ocr(roi_big, cls=True)
            cards2 = self._extract(res2[0] if res2 else [])
            # 合并去重，保留最高置信度
            merged = {c[0]:c for c in cards1}
            for val,conf,box in cards2:
                if val not in merged or conf > merged[val][1]:
                    merged[val] = (val,conf,box)
            cards1 = list(merged.values())

        return cards1

    def recognize(self, img_path: str) -> Dict:
        img = cv2.imread(img_path)
        if img is None:
            return {"success":False, "error":f"无法读取: {img_path}"}

        h,w = img.shape[:2]
        regions = get_screen_regions(w,h)

        # 画预览
        preview = img.copy()
        for name,(x1,y1,x2,y2) in regions.items():
            cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(preview,name,(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imwrite(PREVIEW_IMG, preview)
        self.logger.info(f"ROI 预览: {PREVIEW_IMG}")

        result = {"success":True, "publicCards":[], "handCards":[]}
        out = preview.copy()

        for name,(x1,y1,x2,y2) in regions.items():
            roi = img[y1:y2, x1:x2]
            cards = self.recognize_region(roi, name)
            self.logger.info(f"{name} 检测到: {cards}")

            # 可视化
            for val,conf,box in cards:
                pts = np.array(box,np.int32).reshape(-1,1,2)
                pts = (pts / IMG_CFG['resize_factor']).astype(int) + np.array([[[x1,y1]]])
                cv2.polylines(out,[pts],True,(255,0,0),2)
                cv2.putText(out,f"{val}({conf:.2f})",
                            tuple(pts[0][0]+np.array([0,-12])),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

            slot = "publicCards" if name=="PUBLIC_REGION" else "handCards"
            result[slot] = [v for v,_,_ in cards]

        cv2.imwrite(RESULT_IMG, out)
        self.logger.info(f"OCR 结果图: {RESULT_IMG}")

        return result

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    img_path = "data/templates/4.png"
    ocr = DualChannelPokerOCR()
    out = ocr.recognize(img_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()

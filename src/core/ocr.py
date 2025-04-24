import os
import cv2
import json
import logging
import re
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, Tuple, List

# ============ 配置常量 ============
DEBUG_DIR = "data/debug"
PREVIEW_IMG = os.path.join(DEBUG_DIR, "regions_preview.png")
RESULT_IMG = os.path.join(DEBUG_DIR, "ocr_result.png")
os.makedirs(DEBUG_DIR, exist_ok=True)

# 配置日志格式
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============ 区域配置 ============
REGION_RATIOS = {
    "PUBLIC_REGION": (0.15, 0.5, 0.85, 0.6),  # 扩大公牌区域范围
    "HAND_REGION":   (0.3, 0.88, 0.7, 0.98),
    "CLICK_REGION":  (0.3, 0.65, 0.7, 0.85)
}

# ============ 有效值配置 ============
VALID_CARD = set(["A","2","3","4","5","6","7","8","9","10","J","Q","K"])
VALID_ACTION = set(["弃牌","加注","让牌","跟注"])

# ============ OCR 模型 ============
ocr_en = PaddleOCR(use_angle_cls=False, lang="en",
                   det_db_thresh=0.1,  # 降低检测阈值
                   det_db_box_thresh=0.1,  # 降低框检测阈值
                   det_db_unclip_ratio=2.0,  # 增加文本框扩张比例
                   det_limit_side_len=2000,
                   rec_char_dict_path='/opt/homebrew/Caskroom/miniconda/base/envs/gameai/lib/python3.8/site-packages/paddleocr/ppocr/utils/en_dict.txt',  # 使用英文词典
                   det_model_dir='/Users/mac/.paddleocr/whl/det/en/en_PP-OCRv4_det_infer',  # 使用最新的检测模型
                   rec_model_dir='/Users/mac/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer')  # 使用最新的识别模型

ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch",
                   det_db_thresh=0.1,  # 降低检测阈值
                   det_db_box_thresh=0.1,  # 降低框检测阈值
                   det_db_unclip_ratio=2.0,  # 增加文本框扩张比例
                   det_limit_side_len=2000)

def is_action(text: str) -> bool:
    """判断是否为有效操作"""
    return text in VALID_ACTION or "底池" in text

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

def extract_cards(lines: List) -> List[Tuple[str,float,List]]:
    """提取卡牌信息"""
    if not lines:  # 如果识别结果为空
        return []
        
    cards = []
    for box, (txt, conf) in lines:
        t = txt.strip().upper()
        # 第一步：直接匹配
        if t in VALID_CARD:
            cards.append((t, conf, box))
            continue
            
        # 第二步：尝试提取数字或字母
        m = re.search(r'([A-Z]|\d+)', t)
        if m:
            t = m.group(1)
            if t == '0':
                t = '10'
            if t in VALID_CARD:
                cards.append((t, conf, box))
                continue
                
        # 第三步：特殊字符处理
        if t == '1' or t == 'I':  # 处理可能的1和I混淆
            cards.append(('1', conf, box))
        elif t == 'O':  # 处理可能的O和0混淆
            cards.append(('0', conf, box))
            
    # 按x坐标排序
    cards.sort(key=lambda c: np.mean([pt[0] for pt in c[2]]))
    return cards

def DualChannelPokerOCR(img: np.ndarray) -> Dict:
    """双通道扑克牌OCR识别"""
    if img is None:
        logging.error("无法读取图像")
        return {"success": False, "error": "无法读取图像"}

    h, w = img.shape[:2]
    regs = get_regions(w, h)

    # 第一步：可视化ROIs
    vis = img.copy()
    for name, (x1, y1, x2, y2) in regs.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, name, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(PREVIEW_IMG, vis)

    result = {"success": True, "publicCards": [], "handCards": [], "actions": []}
    out = vis.copy()

    # 第二步：识别公共牌
    x1, y1, x2, y2 = regs["PUBLIC_REGION"]
    roi_pub = img[y1:y2, x1:x2]
    
    # 保存公牌区域图像
    cv2.imwrite(os.path.join(DEBUG_DIR, "public_cards_region.png"), roi_pub)
    
    res_pub = ocr_en.ocr(roi_pub, cls=False)
    
    # 调试信息：打印原始识别结果
    if res_pub and res_pub[0]:
        print("公牌原始识别结果:", [(txt, conf) for box, (txt, conf) in res_pub[0]])
    
    for t, conf, box in extract_cards(res_pub[0] if res_pub else []):
        result["publicCards"].append(t)
        cx, cy = int(np.mean([p[0] for p in box])), int(np.mean([p[1] for p in box]))
        cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 第三步：识别手牌
    x1, y1, x2, y2 = regs["HAND_REGION"]
    roi_hand = img[y1:y2, x1:x2]
    res_hand = ocr_en.ocr(roi_hand, cls=False)
    for t, conf, box in extract_cards(res_hand[0] if res_hand else []):
        result["handCards"].append(t)
        cx, cy = int(np.mean([p[0] for p in box])), int(np.mean([p[1] for p in box]))
        cv2.putText(out, t, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 第四步：识别按钮
    x1, y1, x2, y2 = regs["CLICK_REGION"]
    roi_click = img[y1:y2, x1:x2]
    res_click = ocr_ch.ocr(roi_click, cls=False)
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
    """识别卡牌"""
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": f"无法读取图像: {image_path}"}
    return DualChannelPokerOCR(img)

class OCRProcessor:
    def __init__(self):
        self.ocr_en = ocr_en
        self.ocr_ch = ocr_ch

    def recognize(self, image_path: str) -> Tuple[bool, Dict]:
        """识别图像中的文本
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Tuple[bool, Dict]: (是否成功, 识别结果)
        """
        try:
            result = recognize_cards(image_path)
            return True, result
        except Exception as e:
            return False, {"error": str(e)}

if __name__ == "__main__":
    img = cv2.imread("data/templates/test.png")
    result = DualChannelPokerOCR(img)
    print("识别结果：", json.dumps(result, ensure_ascii=False))

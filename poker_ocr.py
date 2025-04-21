import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import re
import json

# =============== OCR配置 ===============
OCR_CONFIG = {
    'use_angle_cls': True,           # 是否使用角度分类器
    'lang': 'en',                    # 识别语言
    'det_db_thresh': 0.1,           # 文本检测阈值，值越小检测越敏感
    'det_db_box_thresh': 0.3,       # 文本检测框阈值，值越小框越多
    'det_db_unclip_ratio': 2.0,     # 文本检测框扩张比例，值越大框越大
    'det_limit_side_len': 2000,     # 图像边长限制，超过会进行缩放
    'rec_char_dict_path': '/opt/homebrew/Caskroom/miniconda/base/envs/gameai/lib/python3.8/site-packages/paddleocr/ppocr/utils/en_dict.txt'  # 字符字典路径
}

# =============== 区域配置 ===============
# 手牌区域坐标 (x1, y1, x2, y2) 格式，其中：
# x1, y1: 左上角坐标
# x2, y2: 右下角坐标
PLAYER_HAND_REGION = (200, 700, 600, 900)    # 玩家手牌区域
OPPONENT_HAND_REGION = (300, 1300, 650, 1450)  # 对手手牌区域

# =============== 图像处理配置 ===============
IMAGE_PROCESS_CONFIG = {
    # CLAHE (对比度受限的自适应直方图均衡化) 参数
    'clahe_clip_limit': 3.0,        # 对比度限制，值越大对比度越强
    'clahe_tile_size': (4, 4),      # 分块大小，值越大对比度越均匀
    
    # 高斯模糊参数
    'gaussian_kernel': (3, 3),      # 高斯核大小，必须是奇数
    
    # 自适应阈值参数
    'adaptive_threshold_block_size': 9,  # 邻域大小，必须是奇数
    'adaptive_threshold_c': 1,           # 常数，用于调整阈值
    
    # 形态学操作参数
    'morphology_kernel': (3, 3),     # 形态学操作核大小
    
    # 边缘增强参数
    'edge_weight': 1.8,              # 原始图像权重
    'edge_negative_weight': -0.8     # 边缘图像权重
}

# =============== 字体配置 ===============
FONT_CONFIG = {
    'path': "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # 字体文件路径
    'size': 28,                      # 字体大小
    'color': 'green',                # 字体颜色
    'outline_color': 'red',          # 边界框颜色
    'outline_width': 2               # 边界框宽度
}

# =============== 输出配置 ===============
OUTPUT_CONFIG = {
    'preprocessed_image': 'data/debug/poker_ocr_pre.png',    # 预处理后的图像保存路径
    'result_image': 'data/debug/poker_ocr_result.jpg',      # 最终结果图像保存路径
    'debug': True,                               # 是否输出调试信息
    'show_confidence': True                      # 是否显示置信度
}

# =============== 扑克牌配置 ===============
VALID_CARD_VALUES = [
    'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'  # 有效的扑克牌值
]

# 初始化OCR
ocr = PaddleOCR(**OCR_CONFIG)

def is_valid_card(text, confidence):
    """验证识别的文本是否为有效的扑克牌值"""
    text = text.strip().upper()
    if text in VALID_CARD_VALUES:
        return True, text
    match = re.search(r'([A-Z]|\d+)', text)
    if match and match.group(1) in VALID_CARD_VALUES:
        return True, match.group(1)
    return False, None

def enhance_image(image):
    """图像增强处理"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=IMAGE_PROCESS_CONFIG['clahe_clip_limit'],
        tileGridSize=IMAGE_PROCESS_CONFIG['clahe_tile_size']
    )
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, IMAGE_PROCESS_CONFIG['gaussian_kernel'], 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        IMAGE_PROCESS_CONFIG['adaptive_threshold_block_size'],
        IMAGE_PROCESS_CONFIG['adaptive_threshold_c']
    )
    kernel = np.ones(IMAGE_PROCESS_CONFIG['morphology_kernel'], np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Laplacian(morph, cv2.CV_8U)
    return cv2.addWeighted(
        morph, 
        IMAGE_PROCESS_CONFIG['edge_weight'], 
        edge, 
        IMAGE_PROCESS_CONFIG['edge_negative_weight'], 
        0
    )

def draw_result(image, valid_cards, regions):
    """绘制识别结果"""
    font = ImageFont.truetype(FONT_CONFIG['path'], FONT_CONFIG['size'])
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for card_value, confidence, box in valid_cards:
        points = [(int(p[0]), int(p[1])) for p in box]
        draw.polygon(points, outline=FONT_CONFIG['outline_color'], width=FONT_CONFIG['outline_width'])
        if OUTPUT_CONFIG['show_confidence']:
            draw.text((int(box[0][0]), int(box[0][1]) - 30), 
                     f"{card_value} ({confidence:.2f})", 
                     font=font, 
                     fill=FONT_CONFIG['color'])
        else:
            draw.text((int(box[0][0]), int(box[0][1]) - 30), 
                     card_value, 
                     font=font, 
                     fill=FONT_CONFIG['color'])
    
    for x1, y1, x2, y2 in regions:
        draw.rectangle([x1, y1, x2, y2], 
                      outline=FONT_CONFIG['outline_color'], 
                      width=FONT_CONFIG['outline_width'])
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_region(image, region, region_name):
    """处理指定区域"""
    x1, y1, x2, y2 = region
    roi = image[y1:y2, x1:x2]
    result = ocr.ocr(roi)
    valid_cards = []
    
    if result and result[0]:
        if OUTPUT_CONFIG['debug']:
            print(f"\n{region_name}牌值:")
        for line in result[0]:
            if not line[0]:
                continue
            text, confidence = line[1]
            is_valid, card_value = is_valid_card(text, confidence)
            if is_valid:
                adjusted_box = [(p[0] + x1, p[1] + y1) for p in line[0]]
                valid_cards.append((card_value, confidence, adjusted_box))
                if OUTPUT_CONFIG['debug']:
                    print(f"牌值: {card_value}, 置信度: {confidence:.2f}")
    
    return valid_cards

def main():
    """主函数"""
    image = cv2.imread("data/templates/4.png")
    if image is None:
        print("Error: Could not read image")
        return
    
    if OUTPUT_CONFIG['debug']:
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    processed_image = enhance_image(image)
    cv2.imwrite(OUTPUT_CONFIG['preprocessed_image'], processed_image)
    
    regions = [PLAYER_HAND_REGION, OPPONENT_HAND_REGION]
    region_names = ["对手手牌区域", "玩家手牌区域"]  # 交换区域名称
    valid_cards = []
    hand_cards = []
    public_cards = []
    
    for region, name in zip(regions, region_names):
        cards = process_region(processed_image, region, name)
        valid_cards.extend(cards)
        if name == "玩家手牌区域":
            hand_cards = [card[0] for card in cards]
        else:
            public_cards = [card[0] for card in cards]
    
    if valid_cards:
        final = draw_result(image, valid_cards, regions)
        cv2.imwrite(OUTPUT_CONFIG['result_image'], final)
        
        # 返回JSON格式结果，使用实际识别到的牌值
        result = {
            "success": True,
            "handCards": hand_cards,  # 使用实际识别到的手牌
            "publicCards": public_cards  # 使用实际识别到的公共牌
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("未识别到有效的扑克牌值")

if __name__ == "__main__":
    main()

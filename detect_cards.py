import cv2
import os
import json
import numpy as np
from typing import Tuple, List, Dict

TEMPLATE_DIR = "data/game/card"
TEST_IMAGE = "data/templates/test.png"
DEBUG_DIR = "data/debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# 区域设置（使用相对比例，最小调整）
REGION_RATIOS = {
    "PUBLIC_REGION": (0.28, 0.5, 0.69, 0.59),  # 扩大公牌区域范围
    "HAND_REGION":   (0.42, 0.89, 0.61, 0.97),
}

def get_region_coordinates(image_shape: Tuple[int, int], region_ratio: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """根据图片尺寸和区域比例计算实际坐标"""
    height, width = image_shape[:2]
    x1 = int(width * region_ratio[0])
    y1 = int(height * region_ratio[1])
    x2 = int(width * region_ratio[2])
    y2 = int(height * region_ratio[3])
    return (x1, y1, x2, y2)

def save_debug_image(img: np.ndarray, name: str):
    """保存调试图片"""
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)
    print(f"保存调试图片: {path}")

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """图像预处理"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # 去噪
    denoised = cv2.fastNlMeansDenoising(enhanced)
    return denoised

# 加载模板牌图（标准大小）
def load_templates() -> Dict[str, np.ndarray]:
    templates = {}
    for fname in os.listdir(TEMPLATE_DIR):
        if fname.endswith(".png"):
            name = fname.replace(".png", "")
            path = os.path.join(TEMPLATE_DIR, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"警告: 无法加载模板图片 {path}")
                continue
            # 预处理模板图片
            templates[name] = preprocess_image(img)
            # 保存预处理后的模板图片
            save_debug_image(templates[name], f"template_{name}.png")
    return templates

# 提取每张牌图像（切分等宽）
def extract_cards_from_region(image: np.ndarray, region: Tuple[int, int, int, int], count: int, prefix: str) -> List[np.ndarray]:
    x1, y1, x2, y2 = region
    print(f"提取区域: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"图片尺寸: {image.shape}")
    
    # 检查区域是否在图片范围内
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print(f"警告: 区域超出图片范围")
        return []
        
    region_img = image[y1:y2, x1:x2]
    if region_img.size == 0:
        print(f"警告: 提取的区域为空")
        return []
    
    # 保存整个区域图片
    save_debug_image(region_img, f"{prefix}_region.png")
        
    card_width = (x2 - x1) // count
    print(f"每张牌宽度: {card_width}")
    
    cards = []
    for i in range(count):
        card = region_img[:, i * card_width:(i + 1) * card_width]
        if card.size > 0:
            # 保存原始卡牌图片
            save_debug_image(card, f"{prefix}_card_{i+1}_original.png")
            # 预处理每张牌的图像
            processed_card = preprocess_image(card)
            # 保存预处理后的卡牌图片
            save_debug_image(processed_card, f"{prefix}_card_{i+1}_processed.png")
            cards.append(processed_card)
        else:
            print(f"警告: 第 {i+1} 张牌提取失败")
    
    return cards

# 单张牌模板匹配
def match_card(img: np.ndarray, templates: Dict[str, np.ndarray]) -> str:
    if img is None or img.size == 0:
        print("警告: 输入图片为空")
        return "UNKNOWN"
    
    best_score = 0
    best_label = "UNKNOWN"
    
    for label, tmpl in templates.items():
        try:
            tmpl_resized = cv2.resize(tmpl, (img.shape[1], img.shape[0]))
            result = cv2.matchTemplate(img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            print(f"模板 {label} 得分: {score:.2f}")
            if score > best_score:
                best_score = score
                best_label = label
        except Exception as e:
            print(f"匹配模板 {label} 时出错: {str(e)}")
    
    print(f"最佳匹配: {best_label}, 得分: {best_score:.2f}")
    return best_label if best_score > 0.5 else "UNKNOWN"  # 降低阈值

# 主识别逻辑
def detect_cards(image_path: str) -> Dict[str, List[str]]:
    print(f"开始处理图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return {"hand_cards": [], "public_cards": []}
        
    print(f"图片尺寸: {image.shape}")
    templates = load_templates()
    print(f"加载了 {len(templates)} 个模板")

    # 计算实际区域坐标
    hand_region = get_region_coordinates(image.shape, REGION_RATIOS["HAND_REGION"])
    public_region = get_region_coordinates(image.shape, REGION_RATIOS["PUBLIC_REGION"])

    # 保存原始图片的副本，并在其上绘制区域框
    debug_img = image.copy()
    cv2.rectangle(debug_img, (hand_region[0], hand_region[1]), (hand_region[2], hand_region[3]), (0, 255, 0), 2)
    cv2.rectangle(debug_img, (public_region[0], public_region[1]), (public_region[2], public_region[3]), (0, 0, 255), 2)
    save_debug_image(debug_img, "regions_marked.png")

    hand_imgs = extract_cards_from_region(image, hand_region, 2, "hand")
    public_imgs = extract_cards_from_region(image, public_region, 5, "public")

    print(f"提取到手牌数量: {len(hand_imgs)}")
    print(f"提取到公共牌数量: {len(public_imgs)}")

    hand_cards = [match_card(img, templates) for img in hand_imgs]
    public_cards = [match_card(img, templates) for img in public_imgs]

    # 过滤无效识别
    hand_cards = [c for c in hand_cards if c != "UNKNOWN"]
    public_cards = [c for c in public_cards if c != "UNKNOWN"]

    return {
        "hand_cards": hand_cards,
        "public_cards": public_cards
    }

# 运行入口
if __name__ == "__main__":
    result = detect_cards(TEST_IMAGE)
    print("\n最终识别结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

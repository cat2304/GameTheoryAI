import cv2
import pytesseract
import numpy as np
from PIL import Image
from typing import List, Tuple

class CardRecognizer:
    def __init__(self, tesseract_cmd: str = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return image

    def detect_cards(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_imgs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 40 and h > 60:
                card = image[y:y+h, x:x+w]
                card_imgs.append(card)

        card_imgs = sorted(card_imgs, key=lambda x: x.shape[1] * x.shape[0], reverse=True)
        return card_imgs[:3]  # 假设图片最多有三张牌

    def preprocess_card(self, card_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def recognize_card(self, card_img: np.ndarray) -> Tuple[str, str]:
        preprocessed = self.preprocess_card(card_img)
        h, w = preprocessed.shape
        number_area = preprocessed[0:int(h*0.4), 0:int(w*0.5)]
        pil_img = Image.fromarray(number_area)

        text = pytesseract.image_to_string(pil_img, config='--psm 8 -c tessedit_char_whitelist=A23456789JQK10')
        text = text.strip().replace("\n", "")

        # 花色识别可扩展为 CNN 分类
        suit = "Unknown"
        suit_area = card_img[int(h*0.6):, int(w*0.1):int(w*0.9)]
        avg_color = np.mean(suit_area, axis=(0, 1))

        if avg_color[1] > 100 and avg_color[0] < 100:  # 绿色 → 梅花
            suit = '♣'
        elif avg_color[0] > 100 and avg_color[2] > 100:  # 红色 → ♥ or ♦
            suit = '♥'
        elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:  # 黑色 → ♠
            suit = '♠'

        return (text, suit)

    def recognize_all(self, image_path: str) -> List[Tuple[str, str]]:
        image = self.load_image(image_path)
        cards = self.detect_cards(image)
        results = []
        for card_img in cards:
            result = self.recognize_card(card_img)
            results.append(result)
        return results

if __name__ == '__main__':
    recognizer = CardRecognizer()
    results = recognizer.recognize_all('data/templates/public.png')
    for idx, (value, suit) in enumerate(results):
        print(f'Card {idx+1}: {value}{suit}')
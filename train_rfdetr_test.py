import onnxruntime as ort
import numpy as np
import cv2
import json
import os
import time
from PIL import Image, ImageDraw, ImageFont

POKER_CLASSES = [
    "objects", "10f", "10h", "10m", "10s", "2f", "2h", "2m", "2s", "3f",
    "3h", "3m", "3s", "4f", "4h", "4m", "4s", "5f", "5h", "5m", "5s", "6f", "6h", "6m",
    "6s", "7f", "7h", "7m", "7s", "8f", "8h", "8m", "8s", "9f", "9h", "9m", "9s", "Af",
    "Ah", "Am", "As", "Jf", "Jh", "Jm", "Js", "Kf", "Kh", "Km", "Ks", "Qf", "Qh", "Qm", "Qs"
]

CONFIDENCE_THRESHOLD = 0.1

model_path = "data/inference_model.onnx"

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    h, w = image.shape[:2]
    image = cv2.resize(image, (560, 560))
    image = image[:, :, ::-1] / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image, original, (h, w)


def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / x_exp.sum(axis=-1, keepdims=True)


def postprocess(boxes, logits, orig_shape):
    h, w = orig_shape
    boxes = boxes[0]
    logits = softmax(logits[0])
    scores = np.max(logits, axis=1)
    classes = np.argmax(logits, axis=1)
    results = []

    for i in range(len(scores)):
        if scores[i] < CONFIDENCE_THRESHOLD:
            continue

        cx, cy, bw, bh = boxes[i]
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        label = POKER_CLASSES[classes[i]]
        score = float(scores[i])

        results.append({
            "label": label,
            "confidence": round(score, 4),
            "box": [x1, y1, x2, y2]
        })

    return results


def visualize(original, detections, save_path):
    image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['label']}:{det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1 - 10), label, fill='red', font=font)

    image.save(save_path)


def detect(image_path):
    img, orig, shape = preprocess_image(image_path)
    boxes, logits = session.run(None, {input_name: img})
    detections = postprocess(boxes, logits, shape)

    timestamp = int(time.time())
    result_img_path = f"data/temp/result_{timestamp}.png"
    result_json_path = f"data/temp/result_{timestamp}.json"

    visualize(orig, detections, result_img_path)

    with open(result_json_path, "w") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)

    print("📌 JSON 输出结果:")
    print(json.dumps(detections, indent=2, ensure_ascii=False))
    print(f"✅ 检测图像保存至: {result_img_path}")
    print(f"✅ JSON 文件保存至: {result_json_path}")

    return detections


if __name__ == "__main__":
    os.makedirs("data/temp", exist_ok=True)
    image_path = "data/templates/test.png"
    detect(image_path)

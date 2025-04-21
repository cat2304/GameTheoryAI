from inference_sdk import InferenceHTTPClient
import os

# 检查图片文件是否存在
image_path = "data/templates/4.png"
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit(1)

try:
    # 初始化客户端
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="RXjUbbBocpB50sJM4MUf"
    )

    # 执行推理
    print(f"Processing image: {image_path}")
    result = CLIENT.infer("data/templates/4.png", model_id="pokerstar-cards-detection/1")
    
    # 打印结果
    print("\nDetection Results:")
    if 'predictions' in result:
        for pred in result['predictions']:
            print(f"Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
            print(f"Bounding Box: {pred['x']}, {pred['y']}, {pred['width']}, {pred['height']}")
    else:
        print("No detections found")

except Exception as e:
    print(f"Error during inference: {str(e)}")

import os
import supervision as sv
from PIL import Image
from io import BytesIO
import requests

# === 下载远程图片 ===
image_url = "https://source.roboflow.com/lSpmw3Tr6YXjOje5YdgODT1qKTZ2/T2uoidvFPwpPno9TAeUG/original.jpg"
print(f"下载图片: {image_url}")
response = requests.get(image_url)

if response.status_code != 200:
    print(f"图片下载失败: {response.status_code}")
    exit(1)

image_data = BytesIO(response.content)
image = Image.open(image_data)
print("✅ 图片下载成功")

try:
    # 导入inference库
    from inference import get_model
    print("✅ 已导入inference库")
except ImportError:
    print("❌ inference库未安装，请使用以下命令安装:")
    print("pip install inference")
    exit(1)

# === 使用用户的自定义模型 ===
# 模型ID来自用户的Roboflow项目
CUSTOM_MODEL_ID = "my-first-project-i9qhe/25"
# 用户提供的API密钥
API_KEY = "RXjUbbBocpB50sJM4MUf"
print(f"加载自定义模型: {CUSTOM_MODEL_ID}...")

try:
    model = get_model(CUSTOM_MODEL_ID, api_key=API_KEY)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# === 执行推理 ===
print("执行推理...")
predictions = model.infer(image, confidence=0.5)[0]
print("✅ 推理完成")

# === 解析与可视化 ===
detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]
print(f"检测到的对象: {labels}")

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# 显示 + 保存结果图
sv.plot_image(annotated_image)
# 转换为RGB格式再保存为JPEG
if annotated_image.mode == 'RGBA':
    annotated_image = annotated_image.convert('RGB')
annotated_image.save("annotated_result.jpg")
print("✅ The results have been saved as annotated_result.jpg")
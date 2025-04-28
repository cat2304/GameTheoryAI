import os
import zipfile
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import json
from datetime import datetime

# ==== 1. 配置 ====
# 数据集zip文件路径
dataset_zip = "/Users/mac/ai/Project.v9i.yolov8.zip"

# 解压目录
dataset_dir = "/Users/mac/ai/datasets"

# 训练输出目录
project_dir = "./data"

# 本次训练模型名称（带时间戳）
model_name = f"mac_m2_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 使用设备
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🚀 当前使用设备: {'MPS加速' if device == 'mps' else 'CPU'}")

# ==== 2. 解压数据集 ====
os.makedirs(dataset_dir, exist_ok=True)
print(f"📦 正在解压数据集: {dataset_zip}")
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)
print(f"✅ 解压完成")

# ==== 3. 检查 data.yaml ====
yaml_path = os.path.join(dataset_dir, "data.yaml")
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"❌ 找不到 data.yaml，请确认数据集解压路径正确。")
else:
    print(f"✅ 找到 data.yaml: {yaml_path}")

# ==== 4. 训练参数（模仿Roboflow v9）====
train_args = {
    'data': yaml_path,
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'patience': 50,
    'device': device,
    'project': project_dir,
    'name': model_name,
    'exist_ok': True,
    'optimizer': 'Adam',          # Roboflow默认Adam
    'amp': True,                  # 混合精度训练，加速
    'workers': 4,
    'resume': False,
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'close_mosaic': 10,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'save_period': 10,
    'cache': True,
    'rect': False,
    'cos_lr': True,
    'label_smoothing': 0.0,
    'nbs': 64,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'plots': True,
}

# ==== 5. 开始训练 ====
print(f"🔄 初始化YOLOv8模型")
model = YOLO('yolov8n.pt')  # 选择nano版，也可以换成 yolov8s.pt
print(f"🚀 开始训练，输出到: {project_dir}/{model_name}")
results = model.train(**train_args)

# ==== 6. 验证模型 ====
print(f"✅ 训练完成，开始验证...")
metrics = model.val()

# ==== 7. 保存训练记录 (完整版修正版) ====

print(f"✅ 开始保存训练曲线与指标...")

train_log_dir = os.path.join(project_dir, model_name)

# 检查 CSV 文件
csv_path = os.path.join(train_log_dir, 'results.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ 找不到 results.csv，请确认训练过程成功。")

# 读取训练记录
df = pd.read_csv(csv_path)

# 提取需要的指标
history_data = {
    "train/box_loss": df['train/box_loss'].tolist() if 'train/box_loss' in df else [],
    "val/box_loss": df['val/box_loss'].tolist() if 'val/box_loss' in df else [],
    "metrics/mAP50": df['metrics/mAP_0.5'].tolist() if 'metrics/mAP_0.5' in df else [],
    "metrics/mAP50-95": df['metrics/mAP_0.5:0.95'].tolist() if 'metrics/mAP_0.5:0.95' in df else [],
}

# 保存训练历史为 JSON
history_path = os.path.join(train_log_dir, 'training_results.json')
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=4)

print(f"✅ 训练历史保存到: {history_path}")

# ==== 8. 绘制训练曲线 ====

print(f"📈 绘制训练曲线...")

plt.figure(figsize=(12, 6))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
if history_data["train/box_loss"] and history_data["val/box_loss"]:
    plt.plot(history_data["train/box_loss"], label="训练损失")
    plt.plot(history_data["val/box_loss"], label="验证损失")
    plt.title("训练/验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
else:
    plt.title("训练/验证损失（无数据）")

# 绘制 mAP 曲线
plt.subplot(1, 2, 2)
if history_data["metrics/mAP50"] and history_data["metrics/mAP50-95"]:
    plt.plot(history_data["metrics/mAP50"], label="mAP50")
    plt.plot(history_data["metrics/mAP50-95"], label="mAP50-95")
    plt.title("mAP 曲线")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
else:
    plt.title("mAP 曲线（无数据）")

plt.tight_layout()
plt.savefig(os.path.join(train_log_dir, 'training_curves.png'))
plt.close()

print(f"✅ 训练曲线保存完成！")

# ==== 9. 总结输出 ====

print(f"""
🎯 训练总结：
- 最佳模型权重: {train_log_dir}/weights/best.pt
- 最后一次模型权重: {train_log_dir}/weights/last.pt
- 验证集 mAP50: {metrics.box.map50:.4f}
- 验证集 mAP50-95: {metrics.box.map:.4f}
""")
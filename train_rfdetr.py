from rfdetr import RFDETRBase
import os
import torch

torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 定义数据集位置
DATASET_DIR = "dataset"
# 定义输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = RFDETRBase(
    encoder="dinov2_windowed_small",    # ✅ 只能使用这个
    resolution=392,                     # ✅ 输入图像尺寸最小化
    multi_scale=False,                  # ✅ 关闭多尺度增强
    expanded_scales=False,              # ✅ 关闭扩展缩放
    dec_layers=2,                       # ✅ 降低 transformer decoder 层数
    dim_feedforward=1024,
    hidden_dim=192,
    sa_nheads=4,
    ca_nheads=8,
    num_queries=100,
    pretrain_weights=None,              # ✅ 不加载预训练（节省加载阶段显存）
    force_no_pretrain=True              # ✅ 禁止自动下载 Dinov2 预训练权重
)

print(f"开始训练模型，数据集: {DATASET_DIR}")
model.train(
    dataset_dir=DATASET_DIR, 
    epochs=1, 
    batch_size=2, 
    grad_accum_steps=8, 
    lr=1e-4,
    output_dir=OUTPUT_DIR,
    device="cpu"  # 强制使用CPU
)

# 导出ONNX模型
print("训练完成，开始导出ONNX模型...")
onnx_path = os.path.join(OUTPUT_DIR, "rf-detr.onnx")
model.export(output_path=onnx_path, device="cpu")
print(f"✅ ONNX 模型导出成功：{onnx_path}")
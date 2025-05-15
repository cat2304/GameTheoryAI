import os
import torch
from rfdetr import RFDETRBase

def main():
    # 在脚本开始时就设置环境变量 - 彻底禁用MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 使用环境变量禁用MPS设备
    os.environ["PYTORCH_NO_MPS"] = "1"
    
    # 确保不使用MPS设备
    device = torch.device('cpu')
    
    torch.cuda.empty_cache()
    # 禁用多进程
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 数据路径
    DATASET_DIR = "dataset"
    OUTPUT_DIR = "data/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = RFDETRBase(
        encoder="dinov2_windowed_small",
        resolution=392,
        multi_scale=False,
        expanded_scales=False,
        dec_layers=2,
        dim_feedforward=1024,
        hidden_dim=192,
        sa_nheads=4,
        ca_nheads=8,
        num_queries=100,
        pretrain_weights=None,
        force_no_pretrain=True
    )

    print(f"🚀 开始训练模型，数据集路径: {DATASET_DIR}")
    model.train(
        dataset_dir=DATASET_DIR,
        epochs=1,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-4,
        output_dir=OUTPUT_DIR,
        num_workers=0,      # ✅ 关键点：macOS 必须为 0
        device="cpu"        # ✅ 不用 GPU 时需指定
    )

    # 直接保存PyTorch模型
    print("📦 保存PyTorch模型...")
    torch_path = os.path.join(OUTPUT_DIR, "rf-detr.pth")
    
    # 直接保存整个模型而不是state_dict
    torch.save(model, torch_path)
    print(f"✅ PyTorch模型已保存: {torch_path}")
    
    # 根据GitHub文档，尝试使用保存的模型直接导出ONNX
    print("📦 尝试使用训练好的模型直接导出ONNX...")
    try:
        # 安装ONNX导出依赖
        import subprocess
        print("安装ONNX导出依赖...")
        subprocess.run(["pip", "install", "rfdetr[onnxexport]"], check=True)
        
        # 直接从训练好的模型导出，而不是加载保存的模型
        print("开始导出ONNX...")
        onnx_path = os.path.join(OUTPUT_DIR, "rf-detr.onnx")
        
        # 这里使用原始模型直接导出，避免重新加载
        model.export(output_path=onnx_path, device="cpu")
        print(f"✅ ONNX导出成功: {onnx_path}")
    except Exception as e:
        print(f"❌ 直接导出ONNX失败: {str(e)}")
        
        # 使用安全模式加载模型
        try:
            print("尝试使用安全模式加载已保存的模型...")
            
            # 创建单独的导出脚本，解决安全加载问题
            export_script = f"""
import os
import torch
import sys
from rfdetr import RFDETRBase, RFDETR
from rfdetr.detr import RFDETRBase as RFDETRBaseClass
from torch.serialization import add_safe_globals

# 禁用MPS设备
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_NO_MPS"] = "1"

# 添加安全加载的类
add_safe_globals([RFDETRBaseClass, RFDETR, RFDETRBase])

# 加载模型，显式设置weights_only=False
try:
    print("使用安全方式加载模型...")
    model_path = "{torch_path}"
    
    # 首先尝试加载模型架构
    empty_model = RFDETRBase(
        encoder="dinov2_windowed_small",
        resolution=392,
        multi_scale=False,
        expanded_scales=False,
        dec_layers=2,
        dim_feedforward=1024,
        hidden_dim=192,
        sa_nheads=4,
        ca_nheads=8,
        num_queries=100,
        pretrain_weights=None,
        force_no_pretrain=True
    )
    
    # 加载模型权重到空模型
    loaded_dict = torch.load(model_path, weights_only=False)
    
    # 使用预训练权重初始化模型
    # 方法1: 如果saved_model是state_dict格式
    if isinstance(loaded_dict, dict) and "model" in loaded_dict:
        empty_model.load_state_dict(loaded_dict["model"])
        model = empty_model
    else:
        # 方法2: 直接使用加载的模型
        model = loaded_dict
    
    # 导出ONNX
    print("开始导出ONNX...")
    onnx_path = "{os.path.join(OUTPUT_DIR, 'rf-detr.onnx')}"
    model.export(output_path=onnx_path, device="cpu")
    print(f"✅ ONNX导出成功: {{onnx_path}}")
    sys.exit(0)
except Exception as e:
    print(f"❌ 导出失败: {{str(e)}}")
    sys.exit(1)
"""
            
            with open("export_onnx.py", "w") as f:
                f.write(export_script)
            
            # 使用Python解释器运行
            print("启动导出进程...")
            result = subprocess.run(["python", "export_onnx.py"], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"导出失败:\n{result.stderr}")
                print(f"详细信息:\n{result.stdout}")
                print("请使用已保存的PyTorch模型(.pth)文件")
        except Exception as e:
            print(f"❌ 尝试加载模型失败: {str(e)}")
            print("请使用已保存的PyTorch模型(.pth)文件进行推理")

if __name__ == "__main__":
    main()

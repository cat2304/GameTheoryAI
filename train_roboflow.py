import os
import zipfile
import urllib.request
import yaml
import time
import cv2
import torch
from datetime import datetime
from super_gradients.training import Trainer, models
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.losses import PPYoloELoss
from super_gradients.common.plugins.wandb import WandBDetectionValidationPredictionLoggerCallback, plot_detection_dataset_on_wandb
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import wandb

# ==== 初始化配置 ====
class TrainingConfig:
    def __init__(self):
        # 数据集配置
        self.dataset_url = "https://storage.googleapis.com/roboflow-platform-regional-exports/lSpmw3Tr6YXjOje5YdgODT1qKTZ2/cKaEhFSoEdmMMq1hymq6/25/yolov8.zip"
        self.dataset_dir = "poker_cards_dataset"
        self.dataset_zip = "poker_cards.zip"
        
        # 模型配置
        self.model_arch = "yolo_nas_l"  # yolo_nas_s/m/l
        self.pretrained_weights = "coco"
        
        # 训练参数
        self.epochs = 50
        self.batch_size = 16
        self.img_size = 640
        self.initial_lr = 0.001
        self.warmup_epochs = 3
        
        # W&B配置
        self.wandb_project = "poker-cards-detection"
        self.wandb_entity = None  # 团队名称（如果有）
        self.log_predictions_every = 2  # 每N个epoch记录一次预测
        
        # 路径配置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.model_arch}_{timestamp}"
        self.checkpoint_dir = "./checkpoints"

# ==== 主训练函数 ====
def train_poker_cards_detector():
    # 初始化配置
    cfg = TrainingConfig()
    
    # ==== 1. 初始化W&B ====
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.experiment_name,
        config={
            "model": cfg.model_arch,
            "batch_size": cfg.batch_size,
            "img_size": cfg.img_size,
            "epochs": cfg.epochs,
            "pretrained": cfg.pretrained_weights
        }
    )
    
    # ==== 2. 数据集准备 ====
    if not os.path.exists(cfg.dataset_dir):
        os.makedirs(cfg.dataset_dir, exist_ok=True)
        
        if not os.path.exists(cfg.dataset_zip):
            print("📦 下载数据集...")
            try:
                clean_url = cfg.dataset_url.split('?')[0]
                urllib.request.urlretrieve(clean_url, cfg.dataset_zip)
            except Exception as e:
                raise RuntimeError(f"下载失败: {str(e)}")
        
        print("✅ 下载完成，解压中...")
        with zipfile.ZipFile(cfg.dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(cfg.dataset_dir)
    
    # 读取类别信息
    with open(os.path.join(cfg.dataset_dir, "data.yaml"), 'r') as f:
        data_yaml = yaml.safe_load(f)
        classes = data_yaml['names']
    
    # ==== 3. 数据加载器 ====
    train_loader = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': cfg.dataset_dir,
            'images_dir': 'train/images',
            'labels_dir': 'train/labels',
            'classes': classes,
            'input_dim': (cfg.img_size, cfg.img_size),
            'transforms': [
                {'DetectionMosaic': {'prob': 0.8}},
                {'DetectionMixup': {'prob': 0.3}}, 
                {'DetectionHSV': {'prob': 0.5}},
                {'DetectionRandomAffine': {'degrees': 10, 'scale': (0.8, 1.2)}}
            ]
        },
        dataloader_params={
            'batch_size': cfg.batch_size,
            'num_workers': min(4, os.cpu_count()),
            'shuffle': True
        }
    )
    
    val_loader = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': cfg.dataset_dir,
            'images_dir': 'valid/images',
            'labels_dir': 'valid/labels',
            'classes': classes,
            'input_dim': (cfg.img_size, cfg.img_size)
        },
        dataloader_params={
            'batch_size': cfg.batch_size,
            'num_workers': min(4, os.cpu_count())
        }
    )
    
    # ==== 4. 可视化数据集 ====
    print("📊 可视化训练集样本...")
    plot_detection_dataset_on_wandb(
        dataset=train_loader.dataset,
        max_examples=20,
        dataset_name="train_samples"
    )
    
    print("📊 可视化验证集样本...")
    plot_detection_dataset_on_wandb(
        dataset=val_loader.dataset,
        max_examples=20,
        dataset_name="val_samples"
    )
    
    # ==== 5. 模型初始化 ====
    setup_device(num_gpus=torch.cuda.device_count())
    
    model = models.get(
        cfg.model_arch,
        num_classes=len(classes),
        pretrained_weights=cfg.pretrained_weights
    )
    
    # ==== 6. 训练配置 ====
    trainer = Trainer(
        experiment_name=cfg.experiment_name,
        ckpt_root_dir=cfg.checkpoint_dir
    )
    
    training_params = {
        'max_epochs': cfg.epochs,
        'initial_lr': cfg.initial_lr,
        'lr_mode': 'cosine',
        'cosine_final_lr_ratio': 0.01,
        'warmup_initial_lr': cfg.initial_lr * 0.1,
        'warmup_mode': 'linear_epoch_step',
        'warmup_epochs': cfg.warmup_epochs,
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 0.0001},
        'loss': PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(classes),
            reg_max=16
        ),
        'mixed_precision': True,
        'ema': True,
        'metric_to_watch': 'mAP@0.50',
        'greater_metric_to_watch_is_better': True,
        'valid_metrics_list': [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(classes),
                normalize_targets=True
            )
        ],
        'sg_logger': 'wandb_sg_logger',
        'sg_logger_params': {
            'project_name': cfg.wandb_project,
            'save_checkpoints_remote': True,
            'save_tensorboard_remote': True,
            'save_logs_remote': True,
            'save_checkpoint_as_artifact': True
        },
        'phase_callbacks': [
            WandBDetectionValidationPredictionLoggerCallback(
                class_names=classes,
                log_metrics=True,
                log_images=True,
                log_metrics_every=cfg.log_predictions_every,
                log_images_every=cfg.log_predictions_every
            )
        ]
    }
    
    # ==== 7. 开始训练 ====
    print("🚀 开始训练...")
    trainer.train(
        model=model,
        training_params=training_params,
        train_loader=train_loader,
        valid_loader=val_loader,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.25,
            nms_top_k=300,
            max_predictions=300,
            nms_threshold=0.7
        )
    )
    
    # ==== 8. 模型导出 ====
    print("💾 导出最佳模型...")
    best_model = models.get(
        cfg.model_arch,
        num_classes=len(classes),
        checkpoint_path=os.path.join(
            cfg.checkpoint_dir, 
            cfg.experiment_name, 
            'ckpt_best.pth'
        )
    )
    
    # 导出ONNX
    output_onnx = f"{cfg.experiment_name}.onnx"
    best_model.export(
        output_onnx,
        engine='onnx',
        input_size=(3, cfg.img_size, cfg.img_size)
    )
    
    # 记录模型到W&B
    artifact = wandb.Artifact(
        name=f"{cfg.experiment_name}_model",
        type="model"
    )
    artifact.add_file(output_onnx)
    wandb.log_artifact(artifact)
    
    print(f"✅ 训练完成! 模型已导出为 {output_onnx}")
    wandb.finish()

if __name__ == "__main__":
    train_poker_cards_detector()
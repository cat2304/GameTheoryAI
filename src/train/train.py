import torch
from ultralytics import YOLO
import yaml
import os
import time
import logging
import onnx
import onnxruntime as ort
import shutil
import cv2
import numpy as np
from datetime import datetime
import psutil
import GPUtil

def setup_logging(config):
    """设置日志系统"""
    # 创建logs目录
    log_dir = os.path.abspath(config['dataset']['debug'])
    os.makedirs(log_dir, exist_ok=True)
    
    # 清除现有的日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志，使用单个文件，每次追加
    logging.basicConfig(
        level=logging.DEBUG,  # 设置为DEBUG级别
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log"), mode='w'),  # 使用'a'模式追加
            logging.StreamHandler()
        ]
    )
    
    # 设置ultralytics的日志级别
    logging.getLogger('ultralytics').setLevel(logging.DEBUG)
    
    # 确保日志输出到文件
    logging.info("=== 训练开始 ===")
    logging.info(f"日志文件保存在: {os.path.join(log_dir, 'train.log')}")
    logging.info(f"日志级别: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    
def log_system_info():
    """记录系统信息"""
    logging.info("=== 系统信息 ===")
    logging.info(f"CPU使用率: {psutil.cpu_percent()}%")
    logging.info(f"内存使用率: {psutil.virtual_memory().percent}%")
    
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logging.info(f"GPU {gpu.id}: {gpu.name}")
            logging.info(f"  GPU使用率: {gpu.load*100}%")
            logging.info(f"  GPU内存使用率: {gpu.memoryUtil*100}%")
    except:
        logging.info("无法获取GPU信息")

def preprocess_dataset(config):
    """预处理数据集"""
    try:
        source_path = config['dataset']['source']
        processed_path = config['dataset']['processed']
        
        # 确保路径是绝对路径
        if not os.path.isabs(source_path):
            source_path = os.path.abspath(source_path)
        if not os.path.isabs(processed_path):
            processed_path = os.path.abspath(processed_path)
            
        logging.info(f"数据集源路径: {source_path}")
        logging.info(f"处理后的数据集路径: {processed_path}")
        
        # 检查源路径是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"数据集源路径不存在: {source_path}")
            
        # 创建处理后的数据集目录
        os.makedirs(processed_path, exist_ok=True)
        
        # 检查是否是Label Studio导出的数据
        if os.path.exists(os.path.join(source_path, 'export.json')):
            logging.info("检测到Label Studio导出数据，开始处理...")
            process_label_studio_data(source_path, processed_path, config)
        else:
            # 直接复制数据集
            logging.info("复制数据集文件...")
            for item in os.listdir(source_path):
                src = os.path.join(source_path, item)
                dst = os.path.join(processed_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        # 检查处理后的数据集结构
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = os.path.join(processed_path, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"处理后的数据集缺少必需目录: {dir_name}")
            
            # 检查目录是否为空
            if not os.listdir(dir_path):
                raise ValueError(f"目录 {dir_name} 为空")
        
        logging.info("数据集预处理完成")
        return processed_path
        
    except Exception as e:
        logging.error(f"数据集预处理失败: {str(e)}")
        raise

def validate_dataset(dataset_dir, config):
    """验证数据集格式和完整性，并拆分训练验证集"""
    try:
        # 创建类别ID映射
        class_mapping = {}
        for i, class_name in enumerate(config['class_names']):
            class_mapping[i] = i  # 保持原始ID不变，因为配置文件中的顺序就是我们想要的顺序
            
        logging.info(f"类别映射: {class_mapping}")
        
        # 检查目录结构
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError("数据集目录结构不完整")
            
        # 获取所有图片和标签文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        
        # 使用更健壮的文件名处理方式
        label_files = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            label_files.append(label_file)
        
        # 验证标签文件是否存在
        missing_labels = [f for f in label_files if not os.path.exists(os.path.join(labels_dir, f))]
        if missing_labels:
            raise FileNotFoundError(f"缺少标签文件: {missing_labels}")
            
        # 验证标签格式和收集类别信息
        classes = set()
        valid_pairs = []  # 存储有效的图片-标签对
        
        # 检查标签文件中的类别ID
        for img_file, label_file in zip(image_files, label_files):
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                valid = True
                new_lines = []
                for line in lines:
                    try:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_id = int(float(parts[0]))
                        if class_id in class_mapping:
                            mapped_id = class_mapping[class_id]
                            new_line = f"{mapped_id} {' '.join(parts[1:])}\n"
                            new_lines.append(new_line)
                            classes.add(mapped_id)
                        else:
                            logging.warning(f"标签文件 {label_file} 中存在未知类别ID: {class_id}")
                            valid = False
                    except (ValueError, IndexError) as e:
                        logging.warning(f"处理标签文件 {label_file} 中的行时出错: {str(e)}")
                        valid = False
                
                if valid and new_lines:
                    with open(label_path, 'w') as f:
                        f.writelines(new_lines)
                    valid_pairs.append((img_file, label_file))
                
            except Exception as e:
                logging.warning(f"处理标签文件 {label_file} 时出错: {str(e)}")
                continue
                        
        # 验证类别数量
        if len(classes) == 0:
            raise ValueError("数据集中没有有效的类别")
            
        # 创建训练和验证集目录
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val')
        for d in [train_dir, val_dir]:
            os.makedirs(os.path.join(d, 'images'), exist_ok=True)
            os.makedirs(os.path.join(d, 'labels'), exist_ok=True)
            
        # 随机拆分数据集
        val_ratio = config.get('training', {}).get('val_ratio', 0.2)
        np.random.shuffle(valid_pairs)
        split_idx = int(len(valid_pairs) * (1 - val_ratio))
        train_pairs = valid_pairs[:split_idx]
        val_pairs = valid_pairs[split_idx:]
        
        # 移动文件到对应目录
        for pairs, target_dir in [(train_pairs, train_dir), (val_pairs, val_dir)]:
            for img_file, label_file in pairs:
                # 复制图片
                shutil.copy2(
                    os.path.join(images_dir, img_file),
                    os.path.join(target_dir, 'images', img_file)
                )
                # 复制标签
                shutil.copy2(
                    os.path.join(labels_dir, label_file),
                    os.path.join(target_dir, 'labels', label_file)
                )
        
        logging.info(f"数据集验证和拆分完成:")
        logging.info(f"  - 总数据量: {len(valid_pairs)} 对")
        logging.info(f"  - 训练集: {len(train_pairs)} 对")
        logging.info(f"  - 验证集: {len(val_pairs)} 对")
        logging.info(f"  - 类别数量: {len(classes)}")
        logging.info(f"  - 类别ID已重映射为连续值 (0-{len(classes)-1})")
        
        return {
            'classes': sorted(list(classes)),
            'train_dir': train_dir,
            'val_dir': val_dir,
            'train_count': len(train_pairs),
            'val_count': len(val_pairs)
        }
        
    except Exception as e:
        logging.error(f"数据集验证失败: {str(e)}")
        raise

def train_model(dataset_path, config):
    """训练YOLO模型"""
    try:
        # 获取绝对路径
        abs_dataset_path = os.path.abspath(dataset_path)
        
        # 创建输出目录
        output_dir = os.path.abspath(config['dataset']['output'])
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建数据集配置文件
        dataset_yaml = {
            "path": abs_dataset_path,
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 28,
            "names": ['-', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 
                     'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 
                     's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
        }
        
        yaml_path = os.path.join(abs_dataset_path, "dataset.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml, f, allow_unicode=True)
        
        # 加载预训练模型
        model = YOLO('yolov8s.pt')
        
        # 开始训练
        results = model.train(
            data=yaml_path,
            epochs=config['training']['epochs'],
            imgsz=config['training']['image_size'],
            batch=config['training']['batch_size'],
            device="cpu",  # 强制使用CPU
            project=output_dir,
            name="train",
            exist_ok=True,
            augment=config['training']['augment'],
            optimizer=config['training']['optimizer'],
            lr0=config['training']['lr0'],
            lrf=config['training']['lrf'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay'],
            patience=config['training']['patience'],
            save_period=config['training']['save_period'],
            cache=config['dataset']['cache_images'],
            workers=config['training']['workers'],
            freeze=config['fine_tune']['freeze'] if config['fine_tune']['enabled'] else None,
            conf=config['confidence_threshold'],
            iou=config['nms_threshold'],
            max_det=config['nms_control']['max_det'],
            verbose=True
        )
        
        # 获取最佳模型路径
        best_model_path = os.path.join(output_dir, "train", "weights", "best.pt")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"训练完成但未找到最佳模型文件: {best_model_path}")
            
        logging.info(f"训练完成，模型保存在: {best_model_path}")
        return best_model_path
        
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}")
        raise

def export_to_onnx(model_path, output_path):
    """导出模型到ONNX格式"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        model = YOLO(model_path)
        
        # 导出ONNX
        model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            opset=12
        )
        
        # 查找导出的ONNX文件
        model_dir = os.path.dirname(model_path)
        base_name = os.path.basename(model_path)
        onnx_name = os.path.splitext(base_name)[0] + ".onnx"
        onnx_path = os.path.join(model_dir, onnx_name)
        
        if not os.path.exists(onnx_path):
            # 如果在模型目录中找不到，尝试在当前目录中查找
            onnx_path = onnx_name
            
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"找不到导出的ONNX文件: {onnx_name}")
            
        # 移动文件到指定位置
        shutil.move(onnx_path, output_path)
        
        logging.info(f"✅ ONNX模型已导出到: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ 导出ONNX模型失败: {str(e)}")
        return False

def download_pretrained_model(model_size="s"):
    """下载预训练模型"""
    try:
        model_name = f"yolov8{model_size}.pt"
        model_path = os.path.join("data/yolov8/weights", model_name)
        
        # 创建权重目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 如果模型文件不存在，则下载
        if not os.path.exists(model_path):
            logging.info(f"⏳ 下载预训练模型: {model_name}")
            model = YOLO(model_name)
            # 直接复制下载的模型文件
            shutil.copy2(model_name, model_path)
            logging.info(f"✅ 预训练模型已保存到: {model_path}")
        else:
            logging.info(f"✅ 使用已存在的预训练模型: {model_path}")
        
        return model_path
    except Exception as e:
        logging.error(f"❌ 下载预训练模型失败: {str(e)}")
        raise

def process_label_studio_data(source_dir, target_dir, config):
    """处理 Label Studio 导出的数据"""
    try:
        # 创建目标目录结构
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
        
        # 从配置文件中获取类别信息
        classes = config['class_names']
        
        # 创建类别文件
        classes_file = os.path.join(target_dir, "classes.txt")
        with open(classes_file, "w", encoding="utf-8") as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        # 处理 Label Studio 导出的数据
        # 1. 处理图片
        image_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, "images", file)
                    shutil.copy2(src_path, dst_path)
                    image_files.append(file)
        
        # 2. 处理标注文件
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.txt'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, "labels", file)
                    shutil.copy2(src_path, dst_path)
        
        if not image_files:
            logging.error(f"❌ 未找到任何图片文件在目录: {source_dir}")
            raise FileNotFoundError("未找到训练图片")
        
        logging.info(f"✅ Label Studio 数据处理完成")
        logging.info(f"   - 处理图片数量: {len(image_files)}")
        logging.info(f"   - 目标目录: {target_dir}")
        
        return target_dir
        
    except Exception as e:
        logging.error(f"❌ Label Studio 数据处理出错: {str(e)}")
        raise

def validate_config(config):
    """验证配置文件的完整性"""
    required_fields = {
        'dataset': {
            'source': '数据集源目录',
            'processed': '处理后的数据集目录',
            'output': '输出目录',
            'debug': '日志目录',
            'cache_images': '是否缓存图片'
        },
        'training': {
            'model_size': '模型大小 (n/s/m/l/x)',
            'epochs': '训练轮数',
            'batch_size': '批次大小',
            'image_size': '图片尺寸',
            'device': '训练设备',
            'optimizer': '优化器',
            'lr0': '初始学习率',
            'lrf': '最终学习率',
            'momentum': '动量',
            'weight_decay': '权重衰减',
            'warmup_epochs': '预热轮数',
            'patience': '早停耐心值',
            'save_period': '保存间隔',
            'workers': '数据加载线程数',
            'val_ratio': '验证集比例'
        },
        'fine_tune': {
            'enabled': '是否启用微调',
            'freeze': '冻结层数'
        },
        'nms_control': {
            'max_det': '最大检测数量',
            'max_nms': '最大NMS数量',
            'soft_nms': '是否使用软NMS'
        },
        'class_names': '类别名称列表',
        'confidence_threshold': '置信度阈值',
        'nms_threshold': 'NMS阈值'
    }
    
    missing_fields = []
    
    def check_section(config_section, required_section, path=''):
        for key, desc in required_section.items():
            if isinstance(desc, dict):
                if key not in config_section:
                    missing_fields.append(f"{path}{key}")
                else:
                    check_section(config_section[key], desc, f"{path}{key}.")
            else:
                if key not in config_section:
                    missing_fields.append(f"{path}{key}")
    
    check_section(config, required_fields)
    
    if missing_fields:
        missing_str = '\n  - '.join([''] + missing_fields)
        raise ValueError(f"配置文件缺少以下必需字段:{missing_str}")
        
    # 验证特定字段的值
    if config['training']['model_size'] not in ['n', 's', 'm', 'l', 'x']:
        raise ValueError(f"无效的模型大小: {config['training']['model_size']}, 必须是 n/s/m/l/x 之一")
        
    if not isinstance(config['class_names'], list) or len(config['class_names']) == 0:
        raise ValueError("class_names 必须是非空列表")
        
    if not 0 <= config['confidence_threshold'] <= 1:
        raise ValueError(f"confidence_threshold 必须在 0-1 之间，当前值: {config['confidence_threshold']}")
        
    if not 0 <= config['nms_threshold'] <= 1:
        raise ValueError(f"nms_threshold 必须在 0-1 之间，当前值: {config['nms_threshold']}")
        
    logging.info("✅ 配置验证通过")
    return True

def main():
    """主函数"""
    try:
        # 加载配置文件
        config_path = 'config/config_train.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 验证配置
        validate_config(config)
            
        # 设置日志
        setup_logging(config)
        
        # 记录系统信息
        log_system_info()
        
        # 预处理数据集
        processed_dir = preprocess_dataset(config)
        
        # 验证数据集
        dataset_info = validate_dataset(processed_dir, config)
        
        # 训练模型
        model_path = train_model(processed_dir, config)
        
        # 导出模型
        export_to_onnx(model_path, os.path.join(os.path.dirname(model_path), "best.onnx"))
        
        logging.info("✅ 训练流程完成")
        
    except Exception as e:
        logging.error(f"❌ 训练流程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
import yaml
from PIL import Image
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_annotations(data_yaml):
    """验证数据集标注文件
    
    Args:
        data_yaml: YAML文件路径
        
    Returns:
        list: 错误信息列表
    """
    try:
        # 获取根目录
        root_dir = "/Users/mac/ai/Project.v9i.yolov8"
        
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        
        errors = []
        for split in ['train', 'val']:
            if split not in data:
                errors.append(f"Missing {split} split in data.yaml")
                continue
                
            # 将相对路径转换为绝对路径
            img_dir = os.path.join(root_dir, data[split])
            if not os.path.exists(img_dir):
                errors.append(f"Image directory not found: {img_dir}")
                continue
                
            label_dir = img_dir.replace('images', 'labels')
            if not os.path.exists(label_dir):
                errors.append(f"Label directory not found: {label_dir}")
                continue
            
            for img_file in os.listdir(img_dir):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                # 检查图像-标签匹配
                label_file = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
                if not os.path.exists(label_file):
                    errors.append(f"Missing label: {label_file}")
                
                # 检查图像有效性
                try:
                    img_path = os.path.join(img_dir, img_file)
                    Image.open(img_path).verify()
                except Exception as e:
                    errors.append(f"Corrupted image {img_file}: {str(e)}")
        
        return errors
    except Exception as e:
        logger.error(f"验证过程中出错: {str(e)}")
        return [f"验证失败: {str(e)}"]

def main():
    """主函数"""
    try:
        # 使用指定的data.yaml路径
        data_yaml = "/Users/mac/ai/Project.v9i.yolov8/data.yaml"
        
        if not os.path.exists(data_yaml):
            logger.error(f"找不到数据配置文件: {data_yaml}")
            sys.exit(1)
            
        logger.info(f"开始验证数据集: {data_yaml}")
        errors = validate_annotations(data_yaml)
        
        if errors:
            logger.error("❌ 发现数据问题:")
            for error in errors[:3]:
                logger.error(f"- {error}")
            if len(errors) > 3:
                logger.error(f"... 还有 {len(errors)-3} 个问题未显示")
            sys.exit(1)
        else:
            logger.info("✅ 数据验证通过")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
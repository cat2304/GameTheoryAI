import os
import cv2
import sys
import signal
import numpy as np
import argparse
from vision.predictor import VisionPredictor
from vision.utils import draw_results

def signal_handler(sig, frame):
    print("\n🛑 正在退出程序...")
    cv2.destroyAllWindows()
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description='麻将牌识别测试')
    parser.add_argument('--image', type=str, default='data/templates/1.png', help='要识别的图片路径')
    parser.add_argument('--model', type=str, default='data/yolov8/train/train/weights/best.onnx', 
                        help='ONNX模型路径')
    parser.add_argument('--config', type=str, default='config/config_train.yaml',
                        help='配置文件路径')
    parser.add_argument('--no-show', action='store_true', help='不显示结果窗口')
    return parser.parse_args()

def main():
    try:
        # 注册 Ctrl+C 信号处理
        signal.signal(signal.SIGINT, signal_handler)
        
        # 解析命令行参数
        args = parse_args()
        
        print(f"⚙️ 配置信息:")
        print(f"  图片路径: {args.image}")
        print(f"  配置文件: {args.config}")
        
        # -------- 检查文件存在性 --------
        if not os.path.exists(args.image):
            print(f"❌ 测试图像不存在: {args.image}")
            return
            
        if not os.path.exists(args.model):
            print(f"❌ ONNX模型不存在: {args.model}")
            return
            
        if not os.path.exists(args.config):
            print(f"❌ 配置文件不存在: {args.config}")
            return

        # -------- 加载图像 --------
        print("\n📷 正在加载图像...")
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ 无法加载图像: {args.image}")
            return
        print(f"✅ 成功加载图像: {image.shape}")

        # -------- 初始化识别器 --------
        print("\n🔄 正在初始化识别器...")
        predictor = VisionPredictor(args.config)

        # -------- 执行识别 --------
        print("\n🔍 正在执行识别...")
        results = predictor.recognize_objects(image)
        print("\n🎯 识别结果:")
        if len(results) == 0:
            print("  ⚠️ 没有检测到任何物体！")
        else:
            for i, obj in enumerate(results):
                print(f"  {i+1}. 类别: {obj['label']}, 置信度: {obj['score']:.4f}, 位置: {obj['bbox']}")

        # -------- 绘制可视化 --------
        debug_image = image.copy()
        debug_image = draw_results(debug_image, results)
        
        # 保存结果图像
        output_path = f"result_{os.path.basename(args.image)}"
        cv2.imwrite(output_path, debug_image)
        print(f"\n💾 结果已保存至 {output_path}")
        
        # 显示结果（最多等待2秒）
        if not args.no_show:
            cv2.imshow("识别结果预览", debug_image)
            print("📷 显示结果窗口（2秒后自动关闭）...")
            cv2.waitKey(2000)  # 等待2秒
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保在程序结束时关闭所有窗口
        cv2.destroyAllWindows()
        print("\n👋 程序已退出")
        sys.exit(0)

if __name__ == "__main__":
    main()

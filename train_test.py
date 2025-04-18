import os
import cv2
import sys
import signal
import numpy as np
from vision.predictor import VisionPredictor
from vision.utils import draw_results

def signal_handler(sig, frame):
    print("\n🛑 正在退出程序...")
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    try:
        # 注册 Ctrl+C 信号处理
        signal.signal(signal.SIGINT, signal_handler)
        
        # -------- 配置部分 --------
        config_path = "configs/vision_config.yaml"
        image_path = "1.png"  # 使用当前目录下的测试图片

        # -------- 加载图像 --------
        if not os.path.exists(image_path):
            print(f"❌ 测试图像不存在: {image_path}")
            return
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法加载图像: {image_path}")
            return
        print(f"✅ 成功加载图像: {image.shape}")

        # -------- 初始化识别器 --------
        predictor = VisionPredictor(config_path)

        # -------- 执行识别 --------
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
        cv2.imwrite("result.png", debug_image)
        print("\n💾 结果已保存至 result.png")
        
        # 显示结果（最多等待5秒）
        cv2.imshow("识别结果预览", debug_image)
        print("📷 按任意键关闭窗口（5秒后自动关闭）...")
        key = cv2.waitKey(5000)  # 等待5秒或按键
        
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

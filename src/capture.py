import cv2
import numpy as np
import subprocess

def start_stream():
    while True:
        try:
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5
            )
            if result.returncode != 0 or not result.stdout:
                print("[捕获] ADB 截图失败:", result.stderr.decode())
                continue

            img_array = np.frombuffer(result.stdout, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                print("[捕获] 无法解码图像帧")
                continue

            yield frame
        except Exception as e:
            print("[捕获] 异常：", str(e))
            break
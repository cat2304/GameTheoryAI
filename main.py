from src.capture import start_stream
from src.recognize import detect_tiles
from src.decision import decide_tile_to_play
from src.executor import click_tile
import time
import traceback
import os

LOG_FILE = "logs/runtime.log"

def log(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

if __name__ == "__main__":
    log("[系统] Mahjong AI MVP 识别可视化调试模式 启动")
    try:
        for i, frame in enumerate(start_stream()):
            log(f"[捕获] 第{i+1}帧 屏幕截图已获取")
            hand_tiles, debug_frame = detect_tiles(frame, log)
            action_tile = decide_tile_to_play(hand_tiles, log)
            click_tile(action_tile, log)

            import cv2
            cv2.imshow("识别可视化调试", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.3)
    except KeyboardInterrupt:
        log("[退出] 用户中断退出")
    except Exception as e:
        log(f"[错误] 程序异常：{str(e)}")
        traceback.print_exc()
    finally:
        log("[系统] 程序结束，资源清理完毕")
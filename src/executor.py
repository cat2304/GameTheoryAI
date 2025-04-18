import subprocess

def click_tile(tile, log):
    if tile is None:
        log("[执行] 无需点击")
        return
    x, y = 500, 600
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])
    log(f"[执行] 已点击 {tile}，坐标: ({x}, {y})")
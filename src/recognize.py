import cv2
import os
import numpy as np

def detect_tiles(frame, log):
    templates_dir = 'data/templates/'
    detected_tiles = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    debug_image = frame.copy()

    if not os.listdir(templates_dir):
        log("[识别] 模板目录为空，请添加牌面模板到 data/templates/")
        return [], frame

    for template_name in os.listdir(templates_dir):
        template_path = os.path.join(templates_dir, template_name)
        template_img = cv2.imread(template_path, 0)
        if template_img is None:
            continue

        for scale in [1.0, 0.9, 1.1]:
            resized_template = cv2.resize(template_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if resized_template.shape[0] > frame_gray.shape[0] or resized_template.shape[1] > frame_gray.shape[1]:
                continue

            res = cv2.matchTemplate(frame_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            locs = np.where(res >= threshold)

            for pt in zip(*locs[::-1]):
                tile_name = template_name.split('.')[0]
                detected_tiles.append(tile_name)
                top_left = pt
                bottom_right = (pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0])
                cv2.rectangle(debug_image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(debug_image, tile_name, (pt[0], pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                log(f"[识别] 匹配成功: {tile_name} (scale={scale})")
                break  # 一张模板只匹配一次

    if detected_tiles:
        log(f"[识别] 当前识别结果: {detected_tiles}")
    else:
        log("[识别] 未识别到任何手牌")

    return detected_tiles, debug_image
import requests
import json
import time

# API基础URL
BASE_URL = "http://localhost:8000/api"

# 测试设备ID
DEVICE_ID = "127.0.0.1:16384"

# 测试图片路径
TEST_IMAGE = "data/templates/test.png"

def test_api(endpoint: str, data: dict, description: str):
    """测试API接口"""
    print(f"\n测试 {description}")
    print(f"请求: {json.dumps(data, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{BASE_URL}/{endpoint}", json=data)
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return result["success"]
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def main():
    # 测试设备列表
    test_api("device/list", {}, "获取设备列表")
    
    # 测试设备连接
    test_api("device/connect", {"device_id": DEVICE_ID}, "连接设备")
    
    # 测试获取设备信息
    test_api("device/current", {"device_id": DEVICE_ID}, "获取设备信息")
    
    # 测试点击操作
    test_api("device/click", {
        "device_id": DEVICE_ID,
        "x": 100,
        "y": 100
    }, "点击操作")
    
    # 测试截屏
    test_api("device/screenshot", {
        "device_id": DEVICE_ID
    }, "截屏操作")
    
    # 测试全图OCR
    test_api("ocr/recognize_all", {
        "image_path": TEST_IMAGE
    }, "全图OCR识别")
    
    # 测试区域复制
    test_api("region/copy", {
        "image_path": TEST_IMAGE,
        "region": (100, 100, 200, 200),
        "type": 1
    }, "区域复制")
    
    # 测试屏幕区域截图
    test_api("device/screen/region", {
        "device_id": DEVICE_ID,
        "type": 1,
        "region": (100, 100, 200, 200)
    }, "屏幕区域截图")
    
    # 测试卡牌识别
    test_api("ai/recognize_cards", {
        "image_path": TEST_IMAGE
    }, "卡牌识别")
    
    # 测试断开设备连接
    test_api("device/disconnect", {
        "device_id": DEVICE_ID
    }, "断开设备连接")

if __name__ == "__main__":
    main() 
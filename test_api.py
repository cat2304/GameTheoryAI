import pytest
import os
import json
from fastapi.testclient import TestClient
from src.api.routes import app

# 创建测试客户端
client = TestClient(app)

# 测试数据
TEST_DEVICE_ID = "127.0.0.1:16384"
TEST_IMAGE_PATH = "data/templates/test.png"
TEST_REGION = (100, 100, 200, 200)  # x, y, width, height

def test_list_devices():
    """测试获取设备列表"""
    response = client.post("/api/device/list", json={})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    assert "devices" in data["data"]

def test_get_current_device():
    """测试获取当前设备信息"""
    response = client.post("/api/device/current", json={"device_id": TEST_DEVICE_ID})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "device_id" in data["data"]
        assert "status" in data["data"]
        assert "screen_size" in data["data"]

def test_connect_device():
    """测试连接设备"""
    response = client.post("/api/device/connect", json={"device_id": TEST_DEVICE_ID})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "message" in data

def test_disconnect_device():
    """测试断开设备连接"""
    response = client.post("/api/device/disconnect", json={"device_id": TEST_DEVICE_ID})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "message" in data

def test_adb_click():
    """测试点击操作"""
    response = client.post("/api/device/click", json={
        "device_id": TEST_DEVICE_ID,
        "x": 100,
        "y": 200
    })
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "x" in data["data"]
        assert "y" in data["data"]
        assert "timestamp" in data["data"]

def test_screenshot():
    """测试截屏操作"""
    response = client.post("/api/device/screenshot", json={"device_id": TEST_DEVICE_ID})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "path" in data["data"]
        assert "timestamp" in data["data"]

def test_ocr_recognize_all():
    """测试全图文字识别"""
    # 确保测试图片存在
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"测试图片不存在: {TEST_IMAGE_PATH}")
        
    response = client.post("/api/ocr/recognize_all", json={"image_path": TEST_IMAGE_PATH})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "texts" in data["data"]

def test_ocr_recognize_region():
    """测试区域文字识别"""
    # 确保测试图片存在
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"测试图片不存在: {TEST_IMAGE_PATH}")
        
    response = client.post("/api/ocr/recognize_region", json={
        "image_path": TEST_IMAGE_PATH,
        "region": TEST_REGION,
        "type": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "texts" in data["data"]
        assert "type" in data["data"]

def test_recognize_cards():
    """测试卡牌识别"""
    # 确保测试图片存在
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"测试图片不存在: {TEST_IMAGE_PATH}")
        
    response = client.post("/api/ocr/recognize_cards", json={"image_path": TEST_IMAGE_PATH})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "hand_cards" in data["data"]
        assert "public_cards" in data["data"]

def test_copy_region():
    """测试复制指定区域"""
    # 确保测试图片存在
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"测试图片不存在: {TEST_IMAGE_PATH}")
        
    response = client.post("/api/region/copy", json={
        "image_path": TEST_IMAGE_PATH,
        "region": TEST_REGION,
        "type": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "path" in data["data"]

def test_screen_region():
    """测试屏幕区域截图"""
    response = client.post("/api/device/screen/region", json={
        "device_id": TEST_DEVICE_ID,
        "type": 1,
        "region": TEST_REGION
    })
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "path" in data["data"]

def test_recognize_color():
    """测试颜色识别"""
    # 确保测试图片存在
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"测试图片不存在: {TEST_IMAGE_PATH}")
        
    response = client.post("/api/color/recognize", json={
        "image_path": TEST_IMAGE_PATH,
        "region": {"x": 151, "y": 97, "width": 49, "height": 40}
    })
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    if data["success"]:
        assert "color" in data["data"]
        color = data["data"]["color"]
        assert "r" in color
        assert "g" in color
        assert "b" in color
        assert "hex" in color

if __name__ == "__main__":
    pytest.main(["-v", "test_api.py"]) 
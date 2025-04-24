from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import time

from src.core.adb import ADBController
from src.core.screen import ScreenCapture
from src.core.ocr import OCRProcessor
from src.core.ocrall import OCRProcessor as FullOCRProcessor

# 初始化FastAPI应用
app = FastAPI(
    title="Mumu模拟器原子服务",
    description="提供ADB点击、截屏和OCR识别服务",
    version="1.0.0"
)

# 初始化服务
adb_controller = ADBController()
screen_capture = ScreenCapture()
ocr_processor = OCRProcessor()
full_ocr_processor = FullOCRProcessor()

# 请求模型
class ClickRequest(BaseModel):
    x: int
    y: int

class OCRRequest(BaseModel):
    image_path: str

class DeviceRequest(BaseModel):
    device_id: Optional[str] = None

# 响应模型
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class DeviceInfo(BaseModel):
    device_id: str
    status: str
    screen_size: Optional[Dict[str, int]] = None

@app.get("/api/device/list", response_model=ApiResponse)
async def list_devices():
    """获取所有已连接的设备列表"""
    try:
        devices = adb_controller.list_devices()
        return ApiResponse(
            success=True,
            message="获取设备列表成功",
            data={"devices": devices}
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"获取设备列表失败: {str(e)}"
        )

@app.get("/api/device/current", response_model=ApiResponse)
async def get_current_device():
    """获取当前连接的设备信息"""
    try:
        device_id = adb_controller.device_id
        if not device_id:
            return ApiResponse(
                success=False,
                message="当前没有连接的设备"
            )
        
        screen_size = adb_controller.get_screen_size()
        return ApiResponse(
            success=True,
            message="获取设备信息成功",
            data={
                "device_id": device_id,
                "status": "connected",
                "screen_size": screen_size
            }
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"获取设备信息失败: {str(e)}"
        )

@app.post("/api/device/connect", response_model=ApiResponse)
async def connect_device(request: DeviceRequest):
    """连接指定设备"""
    try:
        success = adb_controller.connect_device(request.device_id)
        if success:
            return ApiResponse(
                success=True,
                message="设备连接成功",
                data={"device_id": request.device_id}
            )
        else:
            return ApiResponse(
                success=False,
                message="设备连接失败"
            )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"设备连接失败: {str(e)}"
        )

@app.post("/api/device/disconnect", response_model=ApiResponse)
async def disconnect_device():
    """断开当前设备连接"""
    try:
        success = adb_controller.disconnect_device()
        if success:
            return ApiResponse(
                success=True,
                message="设备断开连接成功"
            )
        else:
            return ApiResponse(
                success=False,
                message="设备断开连接失败"
            )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"设备断开连接失败: {str(e)}"
        )

@app.post("/api/mumu/click", response_model=ApiResponse)
async def adb_click(request: ClickRequest):
    """ADB点击接口"""
    success, message = adb_controller.click(request.x, request.y)
    
    return ApiResponse(
        success=success,
        message=message,
        data={
            "x": request.x,
            "y": request.y,
            "timestamp": time.time()
        } if success else None
    )

@app.post("/api/mumu/screenshot", response_model=ApiResponse)
async def mumu_screenshot():
    """截屏接口"""
    success, result = screen_capture.capture()
    
    return ApiResponse(
        success=success,
        message="截图成功" if success else f"截图失败: {result}",
        data={
            "path": result,
            "timestamp": time.time()
        } if success else None
    )

@app.post("/api/ocr/recognize", response_model=ApiResponse)
async def ocr_recognize(request: OCRRequest):
    """OCR识别接口"""
    success, result = ocr_processor.recognize(request.image_path)
    
    return ApiResponse(
        success=success,
        message="识别成功" if success else f"识别失败: {result.get('error', '未知错误')}",
        data=result if success else None
    )

@app.post("/api/ocr/recognize_all", response_model=ApiResponse)
async def ocr_recognize_all(request: OCRRequest):
    """全图文字识别接口
    
    识别图片中的所有文字，并返回每个文字的位置、置信度等信息。
    同时会生成一个标注了识别结果的图片。
    """
    success, result = full_ocr_processor.recognize_all_text(request.image_path)
    
    return ApiResponse(
        success=success,
        message="识别成功" if success else f"识别失败: {result.get('error', '未知错误')}",
        data=result if success else None
    ) 
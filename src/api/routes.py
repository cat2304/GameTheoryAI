from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Tuple
import time

from src.core.adb import ADBController
from src.core.screen import ScreenCapture
from src.core.ocr import OCRProcessor
from src.core.ocrall import OCRProcessor as FullOCRProcessor
from src.core.ocr_region import OCRRegionProcessor

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
region_ocr_processor = OCRRegionProcessor()

# 请求模型
class ClickRequest(BaseModel):
    device_id: str
    x: int
    y: int

class OCRRequest(BaseModel):
    image_path: str

class OCRRegionRequest(BaseModel):
    image_path: str
    region: Tuple[int, int, int, int]  # (x, y, width, height)

class DeviceRequest(BaseModel):
    device_id: str

class EmptyRequest(BaseModel):
    pass

# 响应模型
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class DeviceInfo(BaseModel):
    device_id: str
    status: str
    screen_size: Optional[Dict[str, int]] = None

@app.post("/api/device/list", response_model=ApiResponse)
async def list_devices(request: EmptyRequest):
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

@app.post("/api/device/current", response_model=ApiResponse)
async def get_current_device(request: DeviceRequest):
    """获取当前连接的设备信息"""
    try:
        screen_size = adb_controller.get_screen_size(request.device_id)
        if not screen_size:
            return ApiResponse(
                success=False,
                message="获取设备信息失败"
            )
        
        return ApiResponse(
            success=True,
            message="获取设备信息成功",
            data={
                "device_id": request.device_id,
                "status": "connected",
                "screen_size": {
                    "width": screen_size[0],
                    "height": screen_size[1]
                }
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
async def disconnect_device(request: DeviceRequest):
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
    success, message = adb_controller.click(request.device_id, request.x, request.y)
    
    return ApiResponse(
        success=success,
        message=message,
        data={
            "device_id": request.device_id,
            "x": request.x,
            "y": request.y,
            "timestamp": time.time()
        } if success else None
    )

@app.post("/api/mumu/screenshot", response_model=ApiResponse)
async def mumu_screenshot(request: DeviceRequest):
    """截屏接口"""
    success, result = screen_capture.capture(request.device_id)
    
    return ApiResponse(
        success=success,
        message="截图成功" if success else f"截图失败: {result}",
        data={
            "device_id": request.device_id,
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
    """
    success, result = full_ocr_processor.recognize_all_text(request.image_path)
    
    return ApiResponse(
        success=success,
        message="识别成功" if success else f"识别失败: {result.get('error', '未知错误')}",
        data=result if success else None
    )

@app.post("/api/ocr/recognize_region", response_model=ApiResponse)
async def ocr_recognize_region(request: OCRRegionRequest):
    """区域文字识别接口
    
    识别图片中指定区域的文字，并返回每个文字的位置、置信度等信息。
    """
    success, result = region_ocr_processor.recognize_region(request.image_path, request.region)
    
    return ApiResponse(
        success=success,
        message="识别成功" if success else f"识别失败: {result.get('error', '未知错误')}",
        data=result if success else None
    ) 
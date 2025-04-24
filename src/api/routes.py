from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time

from src.core.adb import ADBController
from src.core.screen import ScreenCapture
from src.core.ocr import OCRProcessor

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

# 请求模型
class ClickRequest(BaseModel):
    x: int
    y: int

class OCRRequest(BaseModel):
    image_path: str

# 响应模型
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

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
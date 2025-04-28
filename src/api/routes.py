from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Tuple
import time

from src.core.adb import ADBController
from src.core.screen_all import ScreenCapture
from src.core.ocr_all import OCRProcessor
from src.core.ocr_card import recognize_cards
from src.core.copy_region import RegionProcessor
from src.core.screen_region import ScreenRegionProcessor
from src.core.enums import RegionType
from src.core.ai_card import YOLOCard

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
screen_region_processor = ScreenRegionProcessor()

# 请求模型
class ClickRequest(BaseModel):
    device_id: str
    x: int
    y: int

class OCRRequest(BaseModel):
    image_path: str

class DeviceRequest(BaseModel):
    device_id: str

class EmptyRequest(BaseModel):
    pass

class RegionCopyRequest(BaseModel):
    image_path: str
    region: Tuple[int, int, int, int]  # (x, y, width, height)
    type: int = 1  # 区域类型：1=公牌, 2=手牌, 3=操作

class ScreenRegionRequest(BaseModel):
    device_id: str
    type: int = 1  # 区域类型：1=公牌, 2=手牌, 3=操作
    region: Tuple[int, int, int, int]  # (x, y, width, height)

# 响应模型
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class DeviceInfo(BaseModel):
    device_id: str
    status: str
    screen_size: Optional[Dict[str, int]] = None

class OCRResponse(BaseModel):
    success: bool
    texts: List[str]
    error: Optional[str] = None

class OCRCardResponse(BaseModel):
    success: bool
    hand_cards: List[str]
    public_cards: List[str]
    error: Optional[str] = None

# 初始化路由
router = APIRouter(prefix="/api")

@router.post("/device/list", response_model=ApiResponse)
async def list_devices(request: EmptyRequest):
    """获取设备列表
    请求: {}
    响应: {"success": true, "data": {"devices": ["127.0.0.1:16384"]}}"""
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

@router.post("/device/current", response_model=ApiResponse)
async def get_current_device(request: DeviceRequest):
    """获取当前设备信息
    请求: {"device_id": "127.0.0.1:16384"}
    响应: {"success": true, "data": {"device_id": "127.0.0.1:16384", "status": "connected", "screen_size": {"width": 1920, "height": 1080}}}"""
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

@router.post("/device/connect", response_model=ApiResponse)
async def connect_device(request: DeviceRequest):
    """连接设备
    请求: {"device_id": "127.0.0.1:16384"}
    响应: {"success": true, "data": {"device_id": "127.0.0.1:16384"}}"""
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

@router.post("/device/disconnect", response_model=ApiResponse)
async def disconnect_device(request: DeviceRequest):
    """断开设备连接
    请求: {"device_id": "127.0.0.1:16384"}
    响应: {"success": true, "message": "设备断开连接成功"}"""
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

@router.post("/device/click", response_model=ApiResponse)
async def adb_click(request: ClickRequest):
    """点击操作
    请求: {"device_id": "127.0.0.1:16384", "x": 100, "y": 200}
    响应: {"success": true, "data": {"device_id": "127.0.0.1:16384", "x": 100, "y": 200, "timestamp": 1234567890.123}}"""
    try:
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
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"点击失败: {str(e)}"
        )

@router.post("/device/screenshot", response_model=ApiResponse)
async def mumu_screenshot(request: DeviceRequest):
    """截屏操作
    请求: {"device_id": "127.0.0.1:16384"}
    响应: {"success": true, "data": {"device_id": "127.0.0.1:16384", "path": "data/screenshots/screenshot_1234567890.png", "timestamp": 1234567890}}"""
    try:
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
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"截图失败: {str(e)}"
        )

@router.post("/ocr/recognize_all", response_model=ApiResponse)
async def ocr_recognize_all(request: OCRRequest):
    """全图文字识别
    请求: {"image_path": "data/screenshots/screenshot_1234567890.png"}
    响应: {"success": true, "data": {"texts": [{"text": "识别结果", "confidence": 0.95, "position": {"x": 100, "y": 200, "width": 50, "height": 30}}]}}"""
    try:
        success, result = ocr_processor.recognize_all_text(request.image_path)
        return ApiResponse(
            success=success,
            message="识别成功" if success else f"识别失败: {result.get('error', '未知错误')}",
            data=result if success else None
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"识别失败: {str(e)}"
        )

@router.post("/ai/recognize_cards", response_model=ApiResponse)
async def ai_cards_endpoint(request: OCRRequest):
    """卡牌识别
    请求: {"image_path": "data/screenshots/screenshot_1234567890.png"}
    响应: {"success": true, "data": {"hand_cards": ["Ah", "Kd"], "public_cards": ["Qc", "Jh", "Ts"]}}"""
    try:
        result = recognize_cards(request.image_path)
        return ApiResponse(
            success=result["success"],
            message="识别成功" if result["success"] else f"识别失败: {result.get('error', '未知错误')}",
            data=result if result["success"] else None
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"识别失败: {str(e)}"
        )

@router.post("/region/copy", response_model=ApiResponse)
async def copy_region_endpoint(request: RegionCopyRequest):
    """复制指定区域
    请求: {"image_path": "data/screenshots/screenshot_1234567890.png", "region": [100, 100, 200, 200], "type": 1}
    响应: {"success": true, "data": {"path": "data/screenshots/public/1.png"}}"""
    try:
        region_processor = RegionProcessor()
        success, result = region_processor.copy_region(
            request.image_path,
            request.region,
            RegionType(request.type)
        )
        return ApiResponse(
            success=success,
            message="区域复制成功" if success else f"区域复制失败: {result.get('error', '未知错误')}",
            data=result if success else None
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"区域复制失败: {str(e)}"
        )

@router.post("/device/screen/region", response_model=ApiResponse)
async def screen_region(request: ScreenRegionRequest):
    """屏幕区域截图
    请求: {"device_id": "127.0.0.1:16384", "type": 1, "region": [100, 100, 200, 200]}
    响应: {"success": true, "data": {"path": "data/screenshots/public/1.png"}}"""
    try:
        success, result = screen_region_processor.capture_region(
            request.device_id,
            RegionType(request.type),
            request.region
        )
        return ApiResponse(
            success=success,
            message="区域截图成功" if success else f"区域截图失败: {result.get('error', '未知错误')}",
            data=result if success else None
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"区域截图失败: {str(e)}"
        )

# 注册路由
app.include_router(router) 
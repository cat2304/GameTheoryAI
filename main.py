import uvicorn
import logging
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.api.routes import app
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        # 确保截图目录存在
        os.makedirs("data/screenshots", exist_ok=True)
        
        # 启动服务
        logger.info("启动Mumu模拟器原子服务...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        sys.exit(1) 
import uvicorn
import logging
import os
import sys
import subprocess
import time

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

def kill_port_process(port):
    """杀死占用指定端口的进程"""
    try:
        # 查找占用端口的进程
        cmd = f"lsof -i :{port} | awk 'NR>1 {{print $2}}'"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        
        if output:
            # 如果找到进程，强制杀死
            pids = output.split('\n')
            for pid in pids:
                if pid:
                    logger.info(f"正在终止进程 {pid}")
                    subprocess.run(['kill', '-9', pid])
            
            # 等待端口释放
            time.sleep(1)
            return True
    except subprocess.CalledProcessError:
        # 没有找到占用端口的进程，说明端口是空闲的
        return True
    except Exception as e:
        logger.error(f"终止进程时出错: {e}")
        return False

if __name__ == "__main__":
    try:
        # 确保截图目录存在
        os.makedirs("data/screenshots", exist_ok=True)
        
        # 在启动前清理端口
        logger.info("检查并清理端口 8000...")
        if not kill_port_process(8000):
            logger.error("无法清理端口 8000，服务可能无法正常启动")
        
        # 启动服务
        logger.info("启动Mumu模拟器原子服务...")
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            workers=1,
            loop="asyncio",
            reload=False
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        sys.exit(1)

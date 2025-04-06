"""Web界面模块: {
    "描述": "提供麻将AI系统的Web可视化界面和控制接口",
    "主要特性": {
        "Web仪表盘": ["实时游戏状态可视化", "AI决策过程监控", "性能指标显示"],
        "API接口": ["游戏状态管理", "AI控制接口", "系统配置"],
        "实时更新": ["WebSocket集成", "实时游戏更新", "交互控制"]
    },
    "组件": {
        "Flask应用": ["路由定义", "请求处理", "响应格式化"],
        "WebSocket服务器": ["实时通信", "事件广播", "客户端状态管理"],
        "前端界面": ["React基础UI", "交互组件", "数据可视化"]
    },
    "使用示例": "from web.app import create_app\nfrom web.socket import init_socketio\n\napp = create_app()\nsocketio = init_socketio(app)\n\nif __name__ == '__main__':\n    socketio.run(app, debug=True)",
    "目录结构": ["app/: Flask应用和路由", "socket/: WebSocket服务器实现", "static/: 前端资源和React组件", "templates/: HTML模板"],
    "API文档": {
        "游戏状态API": ["GET /api/state: 获取当前游戏状态", "POST /api/action: 提交游戏动作", "GET /api/history: 获取游戏历史"],
        "系统API": ["GET /api/status: 获取系统状态", "POST /api/config: 更新配置", "GET /api/metrics: 获取性能指标"]
    }
}"""

# 版本信息
__version__ = '1.0.0'

# 模块级导入
from .app import create_app
from .socket import init_socketio

# 公共接口列表
__all__ = ['create_app', 'init_socketio']

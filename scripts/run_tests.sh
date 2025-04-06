#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录（脚本目录的父目录）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 设置环境变量
export PYTHONDONTWRITEBYTECODE=1
export FLASK_ENV=development

# 运行环境设置脚本
source "$SCRIPT_DIR/setup_env.sh"

# 清理已存在的字节码缓存
echo "清理现有的字节码缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# 运行测试
echo "运行测试..."
PYTHONDONTWRITEBYTECODE=1 python -m pytest "$@" -v 
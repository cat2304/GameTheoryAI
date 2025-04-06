#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录（脚本目录的父目录）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 在最开始就设置Python不生成字节码文件（开发环境）
if [ "${FLASK_ENV:-development}" = "development" ]; then
    export PYTHONDONTWRITEBYTECODE=1
fi

# 检查是否在项目根目录
if [ ! -d "src" ] || [ ! -d "config" ]; then
    echo "错误：项目结构不完整，请确保存在 src 和 config 目录"
    exit 1
fi

# 检查conda环境
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "错误：请先激活conda环境"
    echo "运行: conda activate mahjong"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" != "3.8" ]]; then
    echo "错误：需要Python 3.8版本，当前版本为 ${PYTHON_VERSION}"
    echo "请使用正确的conda环境："
    echo "conda create -n mahjong python=3.8"
    echo "conda activate mahjong"
    exit 1
fi

# 安装依赖
echo "安装项目依赖..."
conda install -y pytest pytest-cov opencv pillow numpy flask pyyaml
conda install -y -c conda-forge pytesseract black flake8

# 设置环境变量
export FLASK_ENV=${FLASK_ENV:-development}
export FLASK_APP=src/web/app.py

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src

# 根据环境设置Python优化选项
if [ "$FLASK_ENV" = "development" ]; then
    echo "配置开发环境..."
    # 开发环境：启用调试
    export PYTHONUNBUFFERED=1         # 禁用输出缓冲
    export PYTHONASYNCIODEBUG=1       # 启用异步IO调试
    export PYTHONDEVMODE=1            # 启用开发模式
    export FLASK_DEBUG=1
    
    # 清理已存在的字节码缓存
    echo "清理Python字节码缓存..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    
    # 清理构建和分发文件
    echo "清理构建和分发文件..."
    rm -rf build/ dist/ *.egg-info/
else
    echo "配置生产环境..."
    # 生产环境：启用优化
    export PYTHONOPTIMIZE=2           # 启用优化
    unset PYTHONDONTWRITEBYTECODE     # 允许生成.pyc文件
    export PYTHONUNBUFFERED=1         # 禁用输出缓冲
    export FLASK_DEBUG=0
fi

# 设置配置文件路径
export CONFIG_PATH=$PROJECT_ROOT/config/app_config.yaml
export OCR_CONFIG_PATH=$PROJECT_ROOT/config/ocr_config.json

# 设置数据目录
export DATA_DIR=$PROJECT_ROOT/data
export LOG_DIR=$DATA_DIR/logs
export SCREENSHOT_DIR=$DATA_DIR/screenshots

# 创建必要的目录
mkdir -p $LOG_DIR $SCREENSHOT_DIR

# 输出环境信息
echo -e "\n环境设置完成！"
echo "----------------------------------------"
echo "项目根目录: $PROJECT_ROOT"
echo "运行环境: $FLASK_ENV"
echo "Python版本: $PYTHON_VERSION (Conda: $CONDA_DEFAULT_ENV)"
echo "Python路径: $PYTHONPATH"
echo "配置文件: $CONFIG_PATH"
echo "数据目录: $DATA_DIR"
echo "字节码缓存: $([ "$FLASK_ENV" = "development" ] && echo "禁用 (开发模式)" || echo "启用 (生产模式)")"
echo "调试模式: $([ "$FLASK_DEBUG" = "1" ] && echo "启用" || echo "禁用")"
echo "----------------------------------------"

echo "Virtual environment: $(which python)"

# 显示Python优化设置
if [ -n "$PYTHONDONTWRITEBYTECODE" ]; then
    echo "Bytecode cache: disabled (development mode)"
else
    echo "Bytecode cache: enabled (production mode)"
fi 
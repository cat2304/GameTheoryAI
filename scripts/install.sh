#!/bin/bash

# 检查Python版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.6" | bc -l) )); then
    echo "Error: Python 3.6 or higher is required"
    exit 1
fi

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# 检查Tesseract是否安装
if ! command -v tesseract &> /dev/null; then
    echo "Warning: Tesseract OCR is not installed"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing Tesseract using Homebrew..."
        brew install tesseract
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing Tesseract using apt..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    else
        echo "Please install Tesseract OCR manually"
    fi
fi

# 创建必要的目录
mkdir -p data/screenshots data/logs

echo "Installation completed successfully!" 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    required_version = "3.8"
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    if current_version != required_version:
        print(f"Error: Python {required_version} is required, but found {current_version}")
        sys.exit(1)

def check_tesseract():
    """检查Tesseract是否安装"""
    tesseract_path = "/usr/local/bin/tesseract"  # 默认路径
    
    try:
        subprocess.run([tesseract_path, '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Tesseract OCR is not installed")
        if platform.system() == 'Darwin':  # macOS
            print("Installing Tesseract using Homebrew...")
            subprocess.run(['brew', 'install', 'tesseract'], check=True)
        elif platform.system() == 'Linux':
            print("Installing Tesseract using apt...")
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'tesseract-ocr'], check=True)
        else:
            print("Please install Tesseract OCR manually")
            sys.exit(1)

def create_directories():
    """创建必要的目录"""
    base_dir = Path(__file__).parent / 'data'
    dirs = ['logs', 'database', 'cache', 'tententimg']
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def setup_environment():
    """设置环境变量"""
    # 设置Python路径
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    # 设置日志环境
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['LOG_DIR'] = str(Path(__file__).parent / 'data' / 'logs')
    
    # 设置数据目录
    os.environ['DATA_DIR'] = str(Path(__file__).parent / 'data')
    os.environ['DATABASE_DIR'] = str(Path(__file__).parent / 'data' / 'database')
    os.environ['CACHE_DIR'] = str(Path(__file__).parent / 'data' / 'cache')
    
    # 设置配置文件路径
    os.environ['CONFIG_PATH'] = str(Path(__file__).parent / 'config' / 'app_config.yaml')

if __name__ == '__main__':
    # 检查Python版本
    check_python_version()
    
    # 检查Tesseract
    check_tesseract()
    
    # 创建目录
    create_directories()
    
    # 设置环境
    setup_environment()
    
    print("Setup completed successfully!") 
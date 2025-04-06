#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import platform
import yaml
from pathlib import Path
from setuptools import setup, find_namespace_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / 'config' / 'app_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_python_version():
    """检查Python版本"""
    config = load_config()
    required_version = config['environment']['python_version']
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    if current_version != required_version:
        print(f"Error: Python {required_version} is required, but found {current_version}")
        sys.exit(1)

def check_tesseract():
    """检查Tesseract是否安装"""
    config = load_config()
    tesseract_path = config['environment']['tesseract_path']
    
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
    config = load_config()
    directories = config['directories']['data']
    
    for name, path in directories.items():
        if name != 'root':  # 跳过根目录，因为它可能已经存在
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

def setup_environment():
    """设置环境变量"""
    config = load_config()
    
    # 设置Python路径
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    # 设置日志环境
    os.environ['LOG_LEVEL'] = config['logging']['level']
    os.environ['LOG_DIR'] = config['directories']['data']['logs']
    
    # 设置数据目录
    os.environ['DATA_DIR'] = config['directories']['data']['root']
    os.environ['SCREENSHOT_DIR'] = config['directories']['data']['screenshots']
    os.environ['DATABASE_DIR'] = config['directories']['data']['database']
    os.environ['CACHE_DIR'] = config['directories']['data']['cache']
    
    # 设置配置文件路径
    os.environ['CONFIG_PATH'] = str(Path(__file__).parent / 'config' / 'app_config.yaml')

class CustomInstall(install):
    """自定义安装命令"""
    def run(self):
        check_python_version()
        check_tesseract()
        create_directories()
        setup_environment()
        install.run(self)

class CustomDevelop(develop):
    """自定义开发模式安装命令"""
    def run(self):
        check_python_version()
        check_tesseract()
        create_directories()
        setup_environment()
        develop.run(self)

setup(
    name="gameai",
    version="0.1.0",
    description="Game AI Assistant",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_namespace_packages(include=['src.*']),
    package_dir={'': '.'},
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'pytesseract>=0.3.8',
        'PyYAML>=6.0',
        'pillow>=8.3.0',
        'requests>=2.26.0',
        'python-dotenv>=0.19.0',
    ],
    python_requires=f">={load_config()['environment']['python_version']}",
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
) 
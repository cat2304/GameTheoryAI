#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GameTheoryAI 项目主程序

此脚本有两个主要功能：
1. 项目安装和配置（setup）：
   - 环境检查（Python版本、Tesseract OCR）
   - 目录结构创建
   - 环境变量设置
   - 项目安装配置

2. 项目运行（run）：
   - 游戏监控
   - AI决策
   - 状态分析

使用说明：
1. 安装项目：python setup.py install
2. 开发模式：python setup.py develop
3. 运行项目：python setup.py run
4. 环境配置：python setup.py setup
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
from setuptools import setup, find_namespace_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from datetime import datetime
import argparse
import time
import yaml
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """检查Python版本"""
    required_version = "3.8"
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    logger.info(f"Checking Python version... (Current: {current_version}, Required: {required_version})")
    
    if current_version != required_version:
        logger.error(f"Python {required_version} is required, but found {current_version}")
        sys.exit(1)
    
    logger.info("Python version check passed")

def check_tesseract():
    """检查Tesseract OCR安装"""
    tesseract_path = "/usr/local/bin/tesseract"  # 默认路径
    
    logger.info("Checking Tesseract OCR installation...")
    
    try:
        result = subprocess.run([tesseract_path, '--version'], check=True, capture_output=True, text=True)
        logger.info(f"Tesseract found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Tesseract OCR is not installed")
        if platform.system() == 'Darwin':  # macOS
            logger.info("Installing Tesseract using Homebrew...")
            try:
                subprocess.run(['brew', 'install', 'tesseract'], check=True)
                logger.info("Tesseract installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Tesseract: {e}")
                sys.exit(1)
        elif platform.system() == 'Linux':
            logger.info("Installing Tesseract using apt...")
            try:
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'tesseract-ocr'], check=True)
                logger.info("Tesseract installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Tesseract: {e}")
                sys.exit(1)
        else:
            logger.error("Please install Tesseract OCR manually")
            sys.exit(1)

def create_directories():
    """创建项目必要的目录结构"""
    base_dir = Path(__file__).parent / 'data'
    dirs = ['logs', 'database', 'cache', 'tententimg', 'screenshots']
    
    logger.info("Creating project directories...")
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            sys.exit(1)

def setup_environment():
    """设置项目环境变量"""
    logger.info("Setting up environment variables...")
    
    # 设置Python路径
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    logger.debug(f"PYTHONPATH set to: {os.environ['PYTHONPATH']}")
    
    # 设置日志环境
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['LOG_DIR'] = str(Path(__file__).parent / 'data' / 'logs')
    logger.debug(f"LOG_DIR set to: {os.environ['LOG_DIR']}")
    
    # 设置数据目录
    os.environ['DATA_DIR'] = str(Path(__file__).parent / 'data')
    os.environ['DATABASE_DIR'] = str(Path(__file__).parent / 'data' / 'database')
    os.environ['CACHE_DIR'] = str(Path(__file__).parent / 'data' / 'cache')
    logger.debug(f"DATA_DIR set to: {os.environ['DATA_DIR']}")
    
    # 设置配置文件路径
    os.environ['CONFIG_PATH'] = str(Path(__file__).parent / 'config' / 'app_config.yaml')
    logger.debug(f"CONFIG_PATH set to: {os.environ['CONFIG_PATH']}")
    
    logger.info("Environment setup completed")

def setup_project():
    """设置项目环境"""
    logger.info("Starting setup process...")
    
    # 检查Python版本
    check_python_version()
    
    # 检查Tesseract
    check_tesseract()
    
    # 创建目录
    create_directories()
    
    # 设置环境
    setup_environment()
    
    logger.info("Setup completed successfully!")

def run_project():
    """运行项目"""
    try:
        # 加载配置
        config = load_config()
        if not config:
            raise Exception("配置加载失败")
        logger.info("配置加载成功")
        
        # 验证环境
        if not verify_environment():
            raise Exception("环境验证失败")
        logger.info("环境验证通过")
        
        # 导入并调用 game_monitor 中的 run_project
        from src.core.game.game_monitor import run_project as game_run_project
        game_run_project(config)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        print(f"\n发生错误: {str(e)}")
        print("按回车键返回主菜单...")
        input()

def load_config():
    """加载配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'app_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return None

def verify_environment():
    """验证运行环境"""
    try:
        # 检查Python版本
        if sys.version_info < (3, 8) or sys.version_info >= (3, 9):
            logger.error("Python版本不兼容，需要3.8.x")
            return False
            
        # 检查必要的包
        required_packages = ['cv2', 'numpy', 'pytesseract']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                logger.error(f"缺少必要的包: {package}")
                return False
                
        # 检查配置文件
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'config', 'app_config.yaml')):
            logger.error("配置文件不存在")
            return False
            
        return True
    except Exception as e:
        logger.error(f"环境验证失败: {str(e)}")
        return False

def main():
    """主函数"""
    while True:
        print("\n=== GameTheoryAI 项目管理工具 ===")
        print("1. 环境配置 (setup)")
        print("2. 运行项目 (run)")
        print("3. 安装项目 (install)")
        print("4. 开发模式 (develop)")
        print("0. 退出")
        
        try:
            choice = input("\n请选择操作 (0-4): ").strip()
            
            if choice == '0':
                print("感谢使用，再见！")
                break
            elif choice == '1':
                setup_project()
            elif choice == '2':
                run_project()
            elif choice == '3':
                print("\n正在安装项目...")
                setup(
                    name="game-theory-ai",
                    version="0.1.0",
                    description="Game Theory AI Assistant",
                    long_description=open("README.md").read(),
                    long_description_content_type="text/markdown",
                    author="Your Name",
                    author_email="your.email@example.com",
                    packages=find_namespace_packages(include=['src*']),
                    package_dir={'': '.'},
                    python_requires=">=3.8,<3.9",
                    cmdclass={
                        'install': install,
                        'develop': develop,
                    },
                )
            elif choice == '4':
                print("\n正在进入开发模式...")
                setup(
                    name="game-theory-ai",
                    version="0.1.0",
                    description="Game Theory AI Assistant",
                    long_description=open("README.md").read(),
                    long_description_content_type="text/markdown",
                    author="Your Name",
                    author_email="your.email@example.com",
                    packages=find_namespace_packages(include=['src*']),
                    package_dir={'': '.'},
                    python_requires=">=3.8,<3.9",
                    cmdclass={
                        'install': install,
                        'develop': develop,
                    },
                )
            else:
                print("无效的选择，请重新输入！")
                
        except KeyboardInterrupt:
            print("\n\n程序已终止")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            continue

if __name__ == '__main__':
    main() 
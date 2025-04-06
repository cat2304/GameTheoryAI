"""
GameTheoryAI
===========

博弈AI辅助系统

环境要求
-------
- Python: Conda 3.8
- 依赖包:
  - pytest: 测试框架
  - pyyaml: 配置文件处理
  - opencv: 图像处理
  - pillow: 图像处理
  - numpy: 数值计算
  - pytesseract: OCR文字识别

项目结构
-------
src/
├── core/          # 核心游戏逻辑
├── utils/         # 工具函数
├── storage/       # 数据存储
├── tasks/         # 任务管理
└── web/          # Web界面

环境设置
-------
```bash
# 创建conda环境
conda create -n gameai python=3.8

# 激活环境
conda activate gameai

# 安装依赖
conda install pytest pyyaml opencv pillow numpy
pip install pytesseract
```

使用说明
-------
1. 确保使用正确的Conda环境：
   ```bash
   conda activate gameai
   ```

2. 运行测试：
   ```bash
   python -m pytest tests/
   ```

3. 检查环境：
   ```python
   import sys
   print(sys.version)  # 应该显示Python 3.8.x
   ```

注意事项
-------
1. 推荐使用Conda 3.8环境
2. 所有依赖都需要通过conda安装（除了pytesseract）
3. 运行前确保环境激活正确
"""

# 版本信息
__version__ = '1.0.0' 
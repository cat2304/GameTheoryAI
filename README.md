# Game Theory AI

### 环境硬性要求
1. Python 3.8 (必须使用 Conda 环境)
2. Tesseract OCR
3. ADB 工具
4. 必须使用 Conda 3.8 环境
5. 不要使用 venv
6. 确保 ADB 工具可用
7. 运行前检查配置文件
8. 保持依赖包版本一致

### 环境配置步骤
```bash
# 1. 创建并激活conda环境
conda create -n gameai python=3.8
conda activate gameai

# 2. 安装项目依赖
pip install -e .

# 3. 安装Tesseract OCR
# Mac
brew install tesseract
# Ubuntu
sudo apt-get install tesseract-ocr
# Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装

# 4. 安装深度学习框架
conda install pytorch torchvision torchaudio -c pytorch
conda install opencv scikit-learn matplotlib
```

### 使用说明
1. 配置说明
```bash
# 编辑配置文件
vim config/app_config.yaml  # 设置ADB路径等参数
vim config/ocr_config.json  # 设置OCR参数
```

2. 运行说明
```bash
# 初始化环境
python setup.py

# 启动游戏监控
python src/core/game/monitor.py

# 启动Web界面
python src/web/app.py
```

## 项目简介
博弈AI分析工具

## 项目结构
```
GameTheoryAI/
├── src/                    # 源代码
│   ├── core/              # 核心功能
│   │   ├── ocr/          # OCR识别
│   │   └── game/         # 游戏逻辑
│   ├── utils/            # 工具函数
│   └── web/             # Web界面
├── tests/                # 测试代码
├── data/                 # 数据目录
│   ├── screenshots/     # 截图
│   └── logs/           # 日志
├── config/              # 配置文件
└── docs/               # 文档
```

## 开发任务

### 第一阶段：基础框架
1. 环境配置
2. 截屏功能
   - 优化延迟 (<16ms)
   - 多分辨率适配
3. OCR集成
   - 训练字库
   - 优化准确率

### 第二阶段：核心功能
1. 策略引擎
   - 规则树系统
   - Q-learning模块
2. 操作模块
   - 输入事件模拟
   - 优化延迟 (<50ms)

### 第三阶段：优化商用
1. 性能优化
   - TensorRT部署
   - 多线程优化
2. 稳定性测试
3. 系统集成

## 开发目标
| 指标 | 目标值 |
|------|--------|
| 截屏延迟 | <16ms |
| OCR准确率 | >95% |
| 决策延迟 | <200ms |
| 内存占用 | <8GB |

class HybridRecognizer:
    def __init__(self):
        self.cv_recognizer = CVRecognizer()  # 传统CV识别
        self.cnn_recognizer = CNNRecognizer()  # CNN识别
        
    def recognize(self, image):
        # 1. 快速预筛选
        cv_results = self.cv_recognizer.recognize(image)
        
        # 2. 对低置信度结果使用CNN
        low_confidence = [r for r in cv_results if r.confidence < 0.9]
        if low_confidence:
            cnn_results = self.cnn_recognizer.recognize(image, low_confidence)
            
        # 3. 结果融合
        return self._merge_results(cv_results, cnn_results)
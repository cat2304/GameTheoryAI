# Game Theory AI

博弈AI分析工具，支持OCR识别和游戏监控。

## 功能特性

- OCR游戏状态识别
- 实时游戏监控
- 自动截图和分析
- Web界面展示
- ADB设备控制

## 项目结构

```
GameTheoryAI/
├── src/                    # 源代码目录
│   ├── core/              # 核心功能
│   │   ├── ocr/          # OCR相关功能
│   │   └── game/         # 游戏相关功能
│   ├── utils/            # 工具函数
│   └── web/             # Web界面
├── tests/                # 测试目录
├── data/                 # 数据目录
│   ├── screenshots/     # 截图存储
│   └── logs/           # 日志存储
├── config/              # 配置文件
└── docs/               # 文档
```

## 开发计划

### 第一阶段：基础框架搭建 (Days 1-3)

#### Day 1: 环境配置
- [x] 配置开发环境 (Python 3.8+, CUDA 11.0+)
- [x] 安装必要依赖 (PaddleOCR, OpenCV, NumPy)
- [x] 搭建基础项目结构

#### Day 2-3: 截屏与识别模块
- [ ] 实现 DXGI 截屏功能
  - [ ] 优化截屏延迟 (<16ms)
  - [ ] 实现多分辨率适配
- [ ] 集成 PaddleOCR
  - [ ] 训练自定义字库
  - [ ] 优化识别准确率

### 第二阶段：核心功能开发 (Days 4-7)

#### Day 4-5: 策略引擎
- [ ] 实现规则树系统
  - [ ] 定义基础规则集
  - [ ] 构建决策树结构
- [ ] 开发 Q-learning 模块
  - [ ] 设计状态空间
  - [ ] 实现奖励机制
  - [ ] 训练基础模型

#### Day 6-7: 操作模块
- [ ] 实现标准输入事件
  - [ ] 鼠标操作模拟
  - [ ] 键盘事件模拟
- [ ] 优化操作延迟 (<50ms)

### 第三阶段：性能优化 (Days 8-10)

#### Day 8: 性能优化
- [ ] TensorRT 模型部署
- [ ] 多线程优化
- [ ] 内存使用优化

#### Day 9: 稳定性测试
- [ ] 高负载测试
- [ ] 长时间运行测试
- [ ] 异常处理机制

#### Day 10: 系统集成
- [ ] 模块整合
- [ ] 性能指标验证
- [ ] 文档完善

## 性能指标

| 指标 | 目标值 | 当前值 | 状态 |
|------|---------|--------|------|
| 截屏延迟 | <16ms | - | ⬜ |
| OCR准确率 | >95% | - | ⬜ |
| 决策延迟 | <200ms | - | ⬜ |
| 内存占用 | <8GB | - | ⬜ |

## 环境要求

- Python: Conda 环境 3.8
  ```bash
  # 创建conda环境
  conda create -n gameai python=3.8
  
  # 激活环境
  conda activate gameai
  
  # 安装依赖
  conda install pytest pyyaml opencv pillow numpy
  pip install pytesseract
  ```

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/GameTheoryAI.git
cd GameTheoryAI
```

2. 安装依赖：
```bash
# 确保在conda环境中
conda activate gameai

# 安装项目依赖
pip install -e .
```

3. 安装Tesseract OCR：
```bash
# Mac
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Windows
# 下载安装器：https://github.com/UB-Mannheim/tesseract/wiki
```

## 使用说明

1. 配置：
   - 编辑 `config/app_config.yaml` 设置ADB路径和其他参数
   - 编辑 `config/ocr_config.json` 设置OCR识别参数

2. 运行游戏监控：
```python
from src.core.game.monitor import GameMonitor

monitor = GameMonitor()
monitor.start()
```

3. 运行Web界面：
```bash
# 确保在conda环境中
conda activate gameai
python src/web/app.py
```

## 开发指南

1. 环境设置
   - 确保使用 Conda 3.8 环境
   - 安装所有必要依赖
   - 配置 ADB 工具（用于Android设备交互）

2. 运行测试
   ```bash
   # 确保在conda环境中
   conda activate gameai
   
   # 运行测试
   python -m pytest tests/
   ```

3. 代码规范
   - 使用 Python 类型注解
   - 遵循 PEP 8 编码规范
   - 所有函数和类都需要添加文档字符串

## 配置说明

主配置文件：`app_config.yml`

```yaml
# 示例配置
adb:
  path: "adb"
  screenshot:
    remote_path: "/sdcard/screenshot.png"
    local_dir: "data/screenshots"
    
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 注意事项

1. 环境配置
   - 使用Conda管理Python环境
   - 确保ADB工具正确配置
   - 保持依赖包版本一致

2. 开发规范
   - 遵循PEP 8编码规范
   - 保持代码注释完整
   - 及时更新文档

3. 测试要求
   - 新功能必须包含单元测试
   - 保持测试覆盖率
   - 定期运行测试套件

4. 其他
   - 必须使用 Conda 3.8 环境
   - 运行前检查配置文件是否正确
   - 确保 ADB 工具可用且设备已连接
   - 不要使用 venv，本项目只支持 conda 环境

## 许可证

MIT License

## 作者

GameTheoryAI Team
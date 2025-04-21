# Game Theory AI

## ADB工具包 (Android Debug Bridge Tools)
这个包提供了一系列用于Android设备自动化操作的工具。

### 主要功能
1. 设备截图获取
   - 支持自定义保存路径
   - 自动生成时间戳文件名
   - 错误重试机制

2. ADB工具管理
   - 自动验证ADB可用性
   - 支持自定义ADB路径
   - 异常处理和日志记录

### 配置说明
配置文件位置: config/config_app.yaml
主要配置项:
- adb.path: ADB工具路径
- adb.screenshot.remote_path: 设备端截图路径
- adb.screenshot.local_dir: 本地保存目录
- adb.screenshot.filename_format: 文件名格式
- logging: 日志配置

### 使用示例
```python
from src.core.mahjong import MahjongAI
ai = MahjongAI()
screenshot_path = ai.take_screenshot()
print(f"截图保存在: {screenshot_path}")
```

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
```

### 使用说明
1. 配置说明
```bash
# 编辑配置文件
vim config/config_app.yaml  # 设置ADB路径等参数
```

2. 运行说明
```bash
# 初始化环境
python setup.py

# 启动麻将AI
python -m src.core.mahjong
```

## 项目简介
这是一个麻将AI项目，提供基础的配置管理和日志功能。

## 项目结构
```
.
├── config/               # 配置文件
│   └── config_app.yaml   # 应用配置
├── README.md            # 项目文档
├── requirements.txt     # 依赖管理
└── mahjong.py          # 主程序
```

## 功能特点
- 配置管理：支持从配置文件加载设置
- 日志系统：支持文件和控制台日志输出
- 错误处理：包含完整的错误处理机制

## 使用说明
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行程序：
```bash
python mahjong.py
```

## 配置说明
配置文件位置: config/config_app.yaml
主要配置项:
- logging.level: 日志级别
- logging.log_dir: 日志目录
- system.debug: 是否启用调试模式
- system.max_retries: 最大重试次数
- system.retry_delay: 重试延迟时间

## 开发说明
- 代码遵循 PEP 8 规范
- 使用类型注解提高代码可读性
- 完整的错误处理和日志记录
- 模块化设计，便于扩展

## 开发目标
| 指标 | 目标值 |
|------|--------|
| 截屏延迟 | <16ms |
| OCR准确率 | >95% |
| 决策延迟 | <200ms |

## 依赖要求
- Python >= 3.8
- opencv-python
- numpy
- pytesseract
- pyyaml
- adb工具

## 作者信息
- 作者: GameTheoryAI Team
- 版本: 1.0.0
- 许可: MIT



# 🀄️ 麻将AI辅助系统项目 · MVP快速验证方案

---

## 🎯 项目目标

在 **一天内** 快速验证麻将游戏AI辅助系统的可行性，形成**最简可运行演示系统（MVP）**，支持完整流程：视频采集 → 图像识别 → 决策分析 → 自动出牌。

---

## 🚀 MVP核心功能（4个模块）

| 模块 | 功能目标 | 实现方式 | 延迟目标 |
|------|-----------|----------|-----------|
| 🎬 视频采集 | 实时采集画面，帧率30FPS | `scrcpy --video-socket` + `PyAV` | < 100ms |
| 🧠 图像识别 | 识别画面内13张手牌 | OpenCV模板匹配 | < 30ms |
| 🧩 决策引擎 | 输出建议出牌 | 规则/频率分析 | < 20ms |
| 🖱 自动点击 | 控制点击打牌 | ADB坐标点击 | < 10ms |
| 🔁 总延迟 | 从画面变化 → 自动点击 | - | < 150ms |

---

## 📂 项目结构规范（必须遵守）

\`\`\`bash
mahjong_ai/
├── data/
│   └── templates/        # 模板图
├── debug_imgs/           # 可视化调试图
├── logs/
│   └── runtime.log        # 全流程日志记录
├── src/
│   ├── capture/           # 视频帧采集
│   ├── recognition/       # 图像识别
│   ├── logic/             # 决策引擎
│   ├── executor/          # ADB点击
│   ├── utils/             # 配置/日志/工具      
└──main.py                 # MVP入口，可运行验证
└── README.md              # 快速说明
\`\`\`

---

## ⚙️ 环境搭建步骤（快速复制粘贴）

\`\`\`bash
conda create -n gameai python=3.8 -y
conda activate gameai
pip install numpy opencv-python PyAV
brew install scrcpy
\`\`\`



## ⚙️ 验证设备已连接

\`\`\`bash
adb devices
adb kill-server && adb start-server
\`\`\`

---

## ⚡ MVP执行步骤（务必逐步完成）

1. **视频帧采集**
   - 启动 `scrcpy --video-socket` 并用 PyAV 解析 socket 流。
   - 封装 `CaptureManager` 支持异步帧捕获 + debug 图保存。

2. **图像识别模块**
   - 采集 34种牌的模板图，统一命名。
   - 使用 `cv2.matchTemplate` 实现手牌识别逻辑。
   - 添加可视化标注（绿色框、文字标签），保存 debug 图。

3. **基础决策模块**
   - 实现 `analyze_hand(cards: List[str]) -> str` 返回建议出牌。
   - 规则可简单基于频率、是否成对、缺一门等策略。

4. **出牌执行模块**
   - 记录每张牌屏幕坐标（初期手动标注）。
   - ADB 发送点击命令打出建议牌，需写日志记录行为。

---

## 🧪 MVP验证标准

- ✅ 输入截图后 1 秒内完成识别、分析、点击。
- ✅ 日志输出每一步（识别结果 / 决策 / 点击位置）。
- ✅ debug_imgs/ 下能看到可视化效果图。
- ✅ main.py 可一键运行完整流程（含日志）。

---

## 🔮 后续扩展计划

- 引入 YOLOv8/RT-DETR 替代模板匹配，支持动态识别。
- 设计神经网络或强化学习出牌策略模型。
- 接入屏幕坐标自动标定模块（基于图像特征或固定偏移）。
- 增加多轮推理与记忆系统（如听牌状态、其他玩家牌面估计）。

---

## 🧑‍💻 ChatGPT 协作规范

我作为你的唯一架构师：

- 所有模块必须按结构输出至 `src/` 下，并有调用入口。
- 日志与调试图必须完整支持，方便你随时追查。
- 模块之间调用清晰、解耦、可替换。
- 你每次输入任务，我会按"模块化 + 可验证 + 可运行"形式返回。

---

## ✅ 当前任务建议（现在就可以执行）

你可以说：
- `请生成 src/capture 目录下的 CaptureManager 模块`
- `识别模板不准，帮我调试识别模块并加图框保存`
- `请创建 main.py 实现完整的 MVP 验证流程`
- `我采集了截图，请帮我识别牌型并建议出牌`


阶段任务安排建议（按优先级排序）
优先级	模块	目标	产出
1️⃣	替换模板为 CNN 分类器	提升识别准确性	cnn_classifier.py + 训练集
2️⃣	点击坐标映射系统	准确控制出牌	牌名 → 点击坐标自动转换
3️⃣	Debug 图自动保存	用于模型训练 + 标注	每一帧输出 debug_imgs 保存图
4️⃣	听牌/胡牌逻辑	AI策略原型	logic/rule_engine.py
5️⃣	CLI控制主程序	独立测试识别/出牌	main.py 参数化控制模块
6️⃣	模块 watchdog/状态监控	提升稳定性	超时保护、模块异常恢复机制
# Vest - 基于YOLO的图像识别系统

## 一、项目概述
Vest是一个专注于图像目标检测的开源项目，基于YOLO（You Only Look Once）技术栈构建。项目整合了从数据预处理、模型训练到实际部署的完整流程，并提供了直观易用的Web界面，支持用户上传图像并获取实时检测结果。

## 二、项目结构
项目采用前后端分离架构，主要组件包括：
- **前端界面**：基于HTML/CSS/JavaScript构建，提供用户交互入口
- **后端服务**：Python脚本实现的YOLO模型推理引擎
- **数据处理模块**：包含数据集转换、划分和配置生成工具

### 核心文件与目录
```
vest/
├── frontend/                  # 前端界面文件
│   └── templates/             # HTML模板
│       ├── index.html         # 主页面
│       ├── logs.html          # 检测日志页面
│       └── team.html          # 团队介绍页面
├── yoloserver/                # 后端服务
│   ├── initialize_project.py  # 项目初始化脚本
│   └── scripts/               # 功能脚本
│       ├── yolo_detector.py   # 图像检测核心逻辑
│       └── yolo_trans.py      # 数据集处理工具
├── LICENSE                    # 开源许可证文件
└── README.md                  # 项目说明文档
```

## 三、功能特性
1. **自动化项目初始化**：自动创建必要的目录结构和配置文件
2. **数据集预处理**：支持多种格式转换和数据集划分策略
3. **多模型支持**：兼容不同版本的YOLO模型（v5/v8等）
4. **实时检测**：高效处理用户上传的图像并返回检测结果
5. **结果可视化**：生成标注后的图像和详细的检测报告
6. **日志管理**：记录历史检测结果，支持查询和统计分析

## 四、安装与使用

### 环境准备
```bash
# 克隆项目仓库
git clone https://github.com/renewablechip/vest.git
cd vest

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt  # 需自行准备requirements.txt
```

### 项目初始化
```bash
python yoloserver/initialize_project.py
```
该脚本会自动创建必要的目录结构：
- `datasets/`：存放原始数据集和处理后的数据集
- `checkpoints/`：模型权重文件
- `results/`：检测结果保存目录
- `logs/`：系统日志

### 数据集处理
使用`yolo_trans.py`进行数据集转换和划分：
```python
from yoloserver.scripts.yolo_trans import YOLODatasetProcessor

# 初始化处理器
processor = YOLODatasetProcessor(train_rate=0.9, valid_rate=0.05)

# 执行数据集处理
processor.process_dataset()
```

### 模型检测
```python
from yoloserver.scripts.yolo_detector import load_yolo_model, detect_image

# 加载模型
model = load_yolo_model(model_path="checkpoints/best.pt", device="cuda")

# 执行检测
detections, output_image = detect_image(
    image_input_path="path/to/your/image.jpg",
    model=model
)

# 保存结果
output_image.save("results/detected_image.jpg")
```

### 启动Web服务
```bash
# 启动Flask服务（需自行实现）
python app.py
```
访问`http://localhost:5000`使用Web界面

## 五、技术栈
- **模型框架**：YOLOv5/v8
- **后端语言**：Python
- **Web框架**：Flask/Django（需自行选择）
- **前端技术**：HTML5, CSS3, JavaScript
- **部署环境**：支持CPU/GPU

## 六、许可证
本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。
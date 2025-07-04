from .logging_utils import setup_logger
from .performance_utils import time_it
from pathlib import Path

YOLOSERVER_ROOT = Path(__file__).resolve().parents[1]  # 项目根目录
CONFIGS_DIR = YOLOSERVER_ROOT / "configs"       # 配置文件目录
DATA_DIR = YOLOSERVER_ROOT / "data"             # 数据集目录
RUNS_DIR = YOLOSERVER_ROOT / "runs"             # 模型运行结果目录
LOGS_DIR = YOLOSERVER_ROOT / "logs"             # 日志目录
MODEL_DIR = YOLOSERVER_ROOT / "models"          # 模型目录
PRETRAINED_DIR = MODEL_DIR / "pretrained" # 预训练模型目录
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints" # 检查点目录
SCRIPTS_DIR = YOLOSERVER_ROOT / "scripts"       # 脚本目录
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images" # 原始图像目录
ORIGINAL_ANNOTATIONS_DIR = DATA_DIR / "raw" / "original_annotations" # 原始标注目录
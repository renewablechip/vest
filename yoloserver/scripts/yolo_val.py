#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :yolo_val.py
# @Time     :2025/6/27 09:10:45
# @Author   :雨霓同学
# @Project  :BTD
# @Function :模型验证

from ultralytics import YOLO
import argparse
import logging
from pathlib import Path
import sys

# 将项目根目录（BTD）添加到系统搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloserver.utils.logging_utils import setup_logger, log_parameters, rename_log_file
from yoloserver.utils.datainfo_utils import log_dataset_info
from yoloserver.utils.model_utils import copy_checkpoint_models
from yoloserver.utils.performance_utils import time_it
from yoloserver.utils.result_utils import log_results
from yoloserver.utils.system_utils import log_device_info
from yoloserver.utils.config_utils import load_config, merge_config
from yoloserver.utils.paths import CONFIGS_DIR, LOGS_DIR, CHECKPOINTS_DIR


def parse_args(): # 1 个用法
    parser = argparse.ArgumentParser(description="YOLO Validation")
    parser.add_argument("--data", type=str, default="data.yaml", help="YAML配置文件路径")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备")
    parser.add_argument("--weights", type=str, default="best_2.pt", help="预训练模型路径")
    parser.add_argument("--workers", type=int, default=8, help="训练数据加载线程数")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="数据集分割")
    parser.add_argument("--use_yaml", type=bool, default=True, help="使用yaml配置文件")
    return parser.parse_args()


def validate_model(model, yolo_args): # 1 个用法
    results = model.val(**vars(yolo_args))
    return results


def main(): # 1 个用法
    args = parse_args()
    logger = setup_logger(base_path=LOGS_DIR,
                          log_type="yolo_val",
                          model_name=args.weights.replace(".pt", ""),
                          log_level=logging.INFO,
                          temp_log=True,
                          logger_name="YOLO_Validation")

    try:
        yaml_config = load_config(config_type='val')
        if args.use_yaml:
            yaml_config = load_config(config_type='val')

        # 合并参数
        yolo_args, project_args = merge_config(args, yaml_config, mode='val')

        # 记录设备信息
        log_device_info(logger)

        # 记录参数
        log_parameters(project_args, logger=logger)

        # 记录数据集信息
        log_dataset_info(args.data, mode=args.split, logger=logger)

        # 检查数据集配置
        data_path = Path(project_args.data)
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path

        if not data_path.exists():
            logger.error(f"数据集配置文件 '{data_path}' 不存在")
            raise ValueError(f"数据集配置文件 '{data_path}' 不存在")

        # 检查模型文件并加载模型
        model_path = Path(project_args.weights)
        if not model_path.is_absolute():
            model_path = CHECKPOINTS_DIR / model_path

        if not model_path.exists():
            logger.error(f"模型文件 '{model_path}' 不存在")
            raise ValueError(f"模型文件 '{model_path}' 不存在")

        logger.info(f"加载待验证模型: {project_args.weights}")
        model = YOLO(str(model_path))

        def add_save_dir_train(trainer):
            trainer.validator.metrics.save_dir = trainer.save_dir

        def add_save_dir_val(validator):
            validator.metrics.save_dir = validator.save_dir

        model.add_callback("on_train_end", add_save_dir_train)
        model.add_callback("on_val_end", add_save_dir_val)

        # 执行模型验证
        decorated_validate_model = time_it(iterations=1, name="模型验证", logger_instance=logger)(validate_model)
        logger.info(f"开始模型验证")
        results = decorated_validate_model(model, yolo_args)

        # 记录模型的验证结果
        log_results(results, logger = logger)

        # 日志文件重命名
        model_name_for_log = project_args.weights.replace(".pt", "")
        rename_log_file(logger, results.save_dir, model_name_for_log)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info(f"YOLO 检测验证脚本结束")


if __name__ == "__main__":
    main()
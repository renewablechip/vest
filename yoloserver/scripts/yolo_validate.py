#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_validate.py
# @Time      :2025/6/26 10:30:00
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :YOLO 数据集验证入口脚本
import argparse
import sys
from pathlib import Path
# 将项目根目录（BTD）添加到系统搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloserver.utils.dataset_validation import (delete_invalid_files,
                                                 verify_dataset_config,
                                                 verify_split_uniqueness)
from yoloserver.utils.logging_utils import setup_logger
from yoloserver.utils.paths import CONFIGS_DIR, LOGS_DIR, YOLOSERVER_ROOT


def main():
    parser = argparse.ArgumentParser(description="YOLO 数据集验证工具",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", type=str,
                        default=str(CONFIGS_DIR / "data.yaml"),
                        help="指向 data.yaml 配置文件的路径。")

    parser.add_argument("--mode", type=str,
                        default="FULL",
                        choices=["FULL", "SAMPLE"],
                        help="验证模式: 'FULL' (验证所有文件) 或 'SAMPLE' (随机抽样验证)。")

    parser.add_argument("--task", type=str,
                        default="detection",
                        choices=["detection", "segmentation"],
                        help="任务类型，决定了标签文件的格式要求。")

    parser.add_argument("--delete-invalid",
                        action="store_true",
                        default="True",
                        help="如果设置此标志，在发现无效文件后会提示用户是否删除。")

    args = parser.parse_args()

    # 为此脚本设置 logger
    logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="yolo_validate",
        logger_name="YOLO Validate"
    )

    logger.info("================== 开始 YOLO 数据集验证 ==================")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"验证模式: {args.mode}")
    logger.info(f"任务类型: {args.task}")
    logger.info(f"删除无效文件选项: {'启用' if args.delete_invalid else '禁用'}")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.critical(f"指定的配置文件不存在: {config_path}")
        sys.exit(1)

 # 1.验证数据集配置和内容
    config_valid, invalid_files = verify_dataset_config(
        yaml_path=config_path,
        current_logger=logger,
        mode=args.mode,
        task_type=args.task
    )

    if not config_valid:
        logger.error(f"数据集内容验证失败！发现 {len(invalid_files)} 个无效的数据项。")
        logger.error("无效文件详情:")
        for item in invalid_files:
            logger.error(f"  - 图像: {item.get('image_path', 'N/A')}, "
                         f"标签: {item.get('label_path', 'N/A')}, "
                         f"原因: {item['error_message']}")

        if args.delete_invalid:
            user_input = input(
                f"\n检测到 {len(invalid_files)} 个无效数据项。是否要删除所有对应的图像和标签文件？ (y/N): ").strip().lower()
            if user_input == 'y':
                delete_invalid_files(invalid_files, logger)
              # 删除后，数据集为 “cleaner”，但初始检查仍然失败。
            else:
                logger.info("用户选择不删除无效文件。")
    else:
        logger.info("数据集内容验证通过！")

    # 2. 验证拆分唯一性
    uniqueness_valid = verify_split_uniqueness(config_path, logger)
    if not uniqueness_valid:
        logger.error("数据集分割唯一性验证失败！存在文件重叠。")
    else:
        logger.info("数据集分割唯一性验证通过！")

   # 3.最终总结
    logger.info("================== 验证结果总结 ==================")
    if config_valid and uniqueness_valid:
        logger.info("🎉 恭喜！您的数据集已通过所有验证。")
    else:
        logger.error("❌ 数据集验证未通过。请检查以上日志输出，修复问题后重试。")
        logger.error(f"详细日志文件位于: {LOGS_DIR.relative_to(YOLOSERVER_ROOT)}/validation/")

    logger.info("================== 验证流程结束 ==================")


if __name__ == "__main__":
    main()
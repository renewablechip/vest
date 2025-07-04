#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logging_utils.py
# @Time      :2025/6/23 14:28:17
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :日志相关的工具类函数

import logging
from datetime import datetime
from pathlib import Path
from yoloserver.utils.paths import RUNS_DIR

def setup_logger(base_path: Path, log_type: str = "general",
                 model_name: str = None,
                 encoding: str = "utf-8",
                 log_level: int = logging.INFO,
                 temp_log: bool = False,
                 logger_name: str = "YOLO Default"
                 ):
    """
    配置日志记录器，将日志保存到指定路径的子目录当中，并同时输出到控制台，日志文件名为类型 + 时间戳
    :param model_name: 模型训练可能需要一个模型的名字，我们可以传入日志记录器，生成带模型名的日志文件
    :param log_type: 日志的类型
    :param base_path: 日志文件的根路径
    :param encoding: 文件编码
    :param log_level: 日志等级
    :param temp_log: 是否启动临时文件名. True会生成 'temp_...' 前缀的文件，便于后续重命名.
    :param logger_name: 日志记录器的名称
    :return: logging.logger: 返回一个日志记录器实例
    """
    # 1. 构建日志文件完整的存放路径
    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成一个带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 根据temp_log参数，生成不同的日志文件名
    prefix = "temp" if temp_log else log_type.replace(" ", "-")  # 简化前缀为 'temp'
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "-"))
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename

    # 3. 获取或创建指定的名称logger实例
    logger = logging.getLogger(logger_name)
    # 设定日志记录器记录最低记录级别
    logger.setLevel(log_level)
    # 阻止日志事件传播到父级logger
    logger.propagate = False

    # 4. 需要避免重复添加日志处理器，因此先检查日志处理器列表中是否已经存在了指定的日志处理器
    if logger.handlers:
        for handler in logger.handlers:
            # 确保关闭旧的 handler，避免资源泄露
            handler.close()
            logger.removeHandler(handler)

    # 5.创建文件处理器，将日志写入到文件当中
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s : %(message)s"))
    # 将文件处理器添加到logger实例中
    logger.addHandler(file_handler)

    # 6.创建控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s : %(message)s"))
    # 将控制台处理器添加到logger实例中
    logger.addHandler(console_handler)

    # 输出一些初始化信息到日志，确认配置成功
    logger.info(f"日志记录器初始化开始".center(60, "="))
    logger.info(f"日志文件路径: {log_file}")
    logger.info(f"当前日志记录器的名称: {logger_name}")
    logger.info(f"当前日志记录器的类型: {log_type}")
    logger.info(f"当前日志记录器的级别: {logging.getLevelName(log_level)}")
    logger.info("日志记录器初始化成功".center(60, "="))
    return logger


def rename_log_file(logger_obj: logging.Logger, save_dir: str, model_name: str, encoding: str = 'utf-8'):
    """
    重命名临时日志文件，使其与最终的训练/验证目录名和模型名保持一致。

    :param logger_obj: 要操作的日志记录器实例。
    :param save_dir: 最终的保存目录 (e.g., 'runs/train/exp1')。
    :param model_name: 模型名称 (e.g., 'yolov8n.pt')。
    :param encoding: 文件编码。
    """
    # 1. 遍历日志器处理器列表的副本
    for handler in list(logger_obj.handlers):
        # 2. 识别文件处理器
        if isinstance(handler, logging.FileHandler):
            # 3. 获取旧日志文件路径
            old_log_file = Path(handler.baseFilename)

            # 确保操作的是临时文件
            if not old_log_file.name.startswith('temp'):
                logger_obj.warning(f"日志文件 '{old_log_file.name}' 不是临时文件, 跳过重命名。")
                continue

            # 4. 解析时间戳和构建新文件名
            try:
                # 假设格式为 'temp_YYYYMMDD-HHMMSS_model.log' 或 'temp_YYYYMMDD-HHMMSS.log'
                name_parts = old_log_file.stem.split('_')
                timestamp = name_parts[1]
            except IndexError:
                # 如果格式不符，使用新的时间戳作为备用
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                logger_obj.warning(f"无法从 '{old_log_file.name}' 解析时间戳, 将使用当前时间。")

            # 获取目录名作为前缀，并清洗模型名
            new_prefix = Path(save_dir).name
            clean_model_name = Path(model_name).stem  # e.g., 'yolov8n.pt' -> 'yolov8n'
            new_log_name = f"{new_prefix}_{timestamp}_{clean_model_name}.log"
            new_log_file = old_log_file.parent / new_log_name

            if old_log_file == new_log_file:
                logger_obj.info("新旧日志文件名相同, 无需重命名。")
                break  # 无需继续，直接退出

            # 5. 关闭旧的文件处理器
            handler.close()
            # 6. 从日志器中移除旧的处理器
            logger_obj.removeHandler(handler)

            # 7. 执行文件重命名
            if old_log_file.exists():
                try:
                    old_log_file.rename(new_log_file)
                    logger_obj.info(f"日志文件已成功重命名为: {new_log_file}")
                except OSError as e:
                    logger_obj.error(f"重命名日志文件失败: {e}. 日志将继续写入旧文件。")
                    # 重命名失败，回退以保证日志连续性
                    new_log_file = old_log_file
            else:
                logger_obj.warning(f"原始日志文件 '{old_log_file}' 不存在, 将直接创建新日志文件。")

            # 8. 添加新的文件处理器
            new_handler = logging.FileHandler(new_log_file, encoding=encoding)
            # 使用与旧handler相同的formatter
            new_handler.setFormatter(handler.formatter)
            logger_obj.addHandler(new_handler)

            # 9. 成功处理后跳出循环
            break


def log_parameters(args, exclude_params=None, logger=None):
    """
    记录命令行和YAML参数信息，返回结构化字典。

    Args:
        args: 命令行参数 (Namespace 对象)
        exclude_params: 不记录的参数键列表
        logger: 日志记录器实例

    Returns:
        dict: 参数字典
    """
    if logger is None:
        logger = logging.getLogger("YOLO_Training")
    if exclude_params is None:
        exclude_params = ['log_encoding', 'use_yaml', 'log_level', 'extra_args']

    logger.info("开始模型参数信息".center(__width := 40, __fillchar := '='))
    logger.info("Parameters")
    logger.info("-" * 40)

    params_dict = {}
    for key, value in vars(args).items():
        if key not in exclude_params and not key.endswith('_specified'):
            source = '命令行' if getattr(args, f'{key}_specified', False) else 'YAML'
            logger.info(f"{key:<20}: {value} (来源: [{source}])")
            params_dict[key] = {"value": value, "source": source}

    return params_dict

if __name__ == "__main__":
    from paths import LOGS_DIR
    import time

    # --- 测试场景 ---
    # 1. 初始化一个临时 logger
    main_logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="test_runs",
        model_name="initial_model",
        temp_log=True,  # 启用临时日志
        logger_name="MyTestApp"
    )
    main_logger.info("程序启动，正在使用临时日志...")

    time.sleep(1)  # 模拟一些工作

    # 2. 假设任务完成，确定了最终保存目录和模型
    final_save_dir = RUNS_DIR / "train" / "exp42"
    final_save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    final_model_name = "yolov8s_finetuned.pt"

    main_logger.info("任务配置确定，准备重命名日志...")

    # 3. 调用重命名函数
    rename_log_file(
        logger_obj=main_logger,
        save_dir=str(final_save_dir),
        model_name=final_model_name
    )

    # 4. 继续使用 logger，日志现在会写入新文件
    main_logger.info("日志文件已重命名，后续日志将写入新文件。")
    main_logger.error("这是一个在重命名后记录的错误。")
    main_logger.info("测试完成。")
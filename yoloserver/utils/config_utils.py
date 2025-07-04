#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :config_utils.py
# @Time     :2025/6/26 09:22:17
# @Author   :雨霄同学
# @Project  :BTD
# @Function :加载配置文件，生成默认的配置文件

import logging
from pathlib import Path
from yoloserver.utils.paths import RUNS_DIR
import yaml
import ast

from yoloserver.utils.paths import CONFIGS_DIR
from yoloserver.utils.configs import COMMENTED_INFER_CONFIG, DEFAULT_INFER_CONFIG
from yoloserver.utils.configs import COMMENTED_TRAIN_CONFIG,DEFAULT_TRAIN_CONFIG
from yoloserver.utils.configs import COMMENTED_VAL_CONFIG,DEFAULT_VAL_CONFIG

import argparse
VALID_YOLO_TRAIN_ARGS = set(DEFAULT_TRAIN_CONFIG)
VALID_YOLO_VAL_ARGS = set(DEFAULT_VAL_CONFIG)
VALID_YOLO_INFER_ARGS = set(DEFAULT_INFER_CONFIG)

logger = logging.getLogger(__name__)


# 确保从 YAML 加载的颜色值是元组，而不是字符串
def parse_color_string(color_val):
    if isinstance(color_val, str):
        try:
            # 使用 ast.literal_eval 安全地将字符串 "(r, g, b)" 转换为元组
            return ast.literal_eval(color_val)
        except (ValueError, SyntaxError):
            logger.error(f"无法解析颜色值: '{color_val}'。请确保格式为 '(R, G, B)'。将使用默认红色。")
            return (0, 0, 255)  # 错误时返回默认颜色（红色）
    return color_val  # 如果已经是元组，直接返回

def load_config(config_type='train'):  # 1 个用法
    """
    加载配置文件, 如果文件不存在, 尝试调用用生成默认的配置文件, 然后加载并返回
    :param config_type: 配置文件类型
    :return: 配置文件内容
    """
    config_path = CONFIGS_DIR / f'{config_type}.yaml'

    if not config_path.exists():
        logger.warning(f"配置文件{config_path}不存在, 尝试生成默认的配置文件")
        if config_type in ['train', 'val', 'infer']:
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                generate_default_config(config_type)
                logger.info(f"生成默认的配置文件成功: {config_path}")
            except Exception as e:
                logger.error(f"创建配置文件目录失败: {e}")
                raise FileNotFoundError(f"创建配置文件目录失败: {e}")
        else:
            logger.error(f"配置文件类型错误: {config_type}")
            raise ValueError(f"配置文件类型错误: {config_type}, 目前仅支持train, val, infer这三种模式")

    # 加载配置文件
    logger.info(f"正在加载配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件{config_path}失败: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件{config_path}失败: {e}")
        raise

def generate_default_config(config_type):  # 1 个用法
    """
    生成默认的配置文件
    :param config_type: 配置文件类型
    :return:
    """
    config_path = CONFIGS_DIR / f'{config_type}.yaml'
    if config_type == 'train':
        config = COMMENTED_TRAIN_CONFIG
    elif config_type == 'val':
        config = COMMENTED_VAL_CONFIG
    elif config_type == 'infer':
        config = COMMENTED_INFER_CONFIG
    else:
        logger.error(f"未知的配置文件类型: {config_type}")
        raise ValueError(f"配置文件类型错误: {config_type}, 目前仅支持train, val, infer这三种模式")

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config)
        logger.info(f"生成默认 {config_type} 配置文件成功, 文件路径为: {config_path}")
    except IOError as e:
        logger.error(f"写入默认 {config_type} 配置文件失败, 请检查文件权限和路径是否正确, 失败: {e}")
    except Exception as e:
        logger.error(f"生成配置文件 '{config_path.name}' 发生未知错误: {e}")
        raise

def _process_params_value(value):
    """
    处理和转换从命令行或YAML文件接收的单个参数值。
    此函数旨在将字符串 'true', 'false', 'none', 'null' 以及数字字符串
    转换为对应的Python原生类型 (bool, None, int, float)。

    :param value: 待处理的参数值
    :return: 转换后类型的值，如果无法转换则返回原始字符串
    """
    # 如果值不是字符串，假定它已经具有正确的类型（例如，来自默认配置字典）
    if not isinstance(value, str):
        return value

    # 转换为小写以便进行不区分大小写的比较
    val_lower = value.lower()

    # 1. 检查布尔值
    if val_lower == 'true':
        return True
    if val_lower == 'false':
        return False

    # 2. 检查None/Null值
    if val_lower in ('none', 'null'):
        return None

    # 3. 检查是否为数字（整数或浮点数）
    #    - .lstrip('-') 支持负数
    #    - .replace('.', '', 1) 支持一个小数点
    if value.lstrip('-').replace('.', '', 1).isdigit():
        try:
            # 如果包含小数点，则尝试转为浮点数
            if '.' in value:
                return float(value)
            # 否则，尝试转为整数
            else:
                return int(value)
        except ValueError:
            # 如果转换失败（例如 "1.2.3"），则按原样返回字符串
            return value

    # 4. 如果以上所有检查都不匹配，则返回原始字符串
    return value


def _validate_final_args(args, mode):
    """
    对最终合并后的参数进行合法性校验。

    :param args: 包含所有最终参数的 argparse.Namespace 对象。
    :param mode: 当前的运行模式 ('train', 'val', 'infer')。
    :raises ValueError: 如果参数不符合要求。
    """
    if mode == 'train':
        # 检查 epochs
        if not (isinstance(args.epochs, int) and args.epochs > 0):
            raise ValueError(f"训练参数 'epochs' 必须是一个正整数, 当前值为: {args.epochs}")

        # 检查 imgsz
        if not (isinstance(args.imgsz, int) and args.imgsz > 0 and args.imgsz % 32 == 0):
            # YOLOv8 推荐的 stride 是 32，所以 imgsz 最好是 32 的倍数
            logger.warning(f"训练参数 'imgsz' ({args.imgsz}) 不是 32 的倍数, 可能会影响模型性能或导致错误。")
        elif not (isinstance(args.imgsz, int) and args.imgsz > 0):
            raise ValueError(f"训练参数 'imgsz' 必须是一个正整数, 当前值为: {args.imgsz}")

        # 检查 batch
        if args.batch is not None and not (isinstance(args.batch, int) and args.batch > 0):
            raise ValueError(f"训练参数 'batch' 必须是一个正整数或 None, 当前值为: {args.batch}")

        # 检查 data 文件是否存在 (路径标准化后)
        if not Path(args.data).exists():
            raise FileNotFoundError(f"训练所需的数据集配置文件 'data' 未找到: {args.data}")

    elif mode == 'val':
        # 确保 split 参数存在 (若无则设置默认值 'val')
        if not hasattr(args, 'split') or args.split is None:
            setattr(args, 'split', 'val')
            logger.info("验证参数 'split' 未设置, 自动设为默认值 'val'")

        # 检查 data 文件是否存在
        if not Path(args.data).exists():
            raise FileNotFoundError(f"验证所需的数据集配置文件 'data' 未找到: {args.gata}")

    elif mode == 'infer':
        # 在推理模式下，'model' 参数通常是必须的，但您的默认配置里没有它。
        # 这里的校验假设 'model' 参数会通过命令行传入。
        # 如果您的主程序逻辑在别处处理模型加载，这里的校验可以省略或调整。
        if not hasattr(args, 'weights') or not args.weights:
            logger.warning("推理参数 'weights' 未指定, 请确保在调用推理引擎时提供了模型路径。")
        elif not Path(args.weights).exists():
            raise FileNotFoundError(f"推理所需的模型文件 'weights' 未找到: {args.weights}")

def merge_config(args, yaml_config, mode='train'):
    """
    合并命令行参数, YAML配置文件参数和默认参数,按优先级CIL > YAML > 默认值

    :param args: 通过argparse解析的参数
    :param yaml_config: 从YAML配置文件中加载的参数
    :param mode: 运行模式.支持train,val, infer
    :return:
    """

    # 1. 确定运行模式和相关配置, 根据传入的mode,选择有效的YOLO参数合集
    if mode == 'train':
        valid_args = VALID_YOLO_TRAIN_ARGS
        default_config = DEFAULT_TRAIN_CONFIG
    elif mode == 'val':
        valid_args = VALID_YOLO_VAL_ARGS
        default_config = DEFAULT_VAL_CONFIG
    elif mode == 'infer':
        valid_args = VALID_YOLO_INFER_ARGS
        default_config = DEFAULT_INFER_CONFIG
    else:
        logger.error(f"{mode} 模式不存在,仅支持train/val/infer三种模式")
        raise ValueError(f"{mode} 模式不存在,仅支持train/val/infer三种模式")

    # 2. 初始化参数存储,project_args用于存储所有最终合并的参数,yolo_args用于存储yolo的参数
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()
    merged_params = default_config.copy()

    # 3. 合并YAML参数,按优先级合并,只有当命令行指定了使用YAML文件,才进行合并,
    # 且yaml_config不是空的时候,才合并
    if hasattr(args, 'use_yaml') and args.use_yaml and yaml_config:
        for key, value in yaml_config.items():
            merged_params[key] = _process_params_value(value)
        logger.debug(f"合并YAML参数后: {merged_params}")

    # 4. 合并命令行参数,具有最高的优先级,会覆盖YAML参数和默认值
    cmd_args = {k: v for k, v in vars(args).items() if k != 'extra_args' and v is not None}
    for key, value in cmd_args.items():
        # 未参数标记来源
        merged_params[key] = _process_params_value(value)
        setattr(project_args, f"{key}_specified", True)

    # 处理动态参数
    if hasattr(args, 'extra_args'):
        if len(args.extra_args) % 2 != 0:
            logger.error("额外参数格式错误, 参数列表必须成对出现,如 --key value")
            raise ValueError("额外参数格式错误, 参数列表必须成对出现,如 --key value")
        for i in range(0, len(args.extra_args), 2):
            key = args.extra_args[i].lstrip("--")
            value = args.extra_args[i + 1]
            # 直接调用并赋值
            merged_params[key] = _process_params_value(value)
            # 标记额外的参数来源
            setattr(project_args, f"{key}_specified", True)

    # 路径标准化策略
    if 'data' in merged_params and merged_params['data']:
        data_path = Path(merged_params['data'])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path
            merged_params['data'] = str(data_path.resolve())
        # 验证数据集配置文件是否存在
        if not data_path.exists():
            logger.warning(f"数据集配置文件 '{data_path}' 不存在")
        logger.info(f"标准化数据集路径: '{merged_params['data']}'")

    # 标准化project参数
    if 'project' in merged_params and merged_params['project']:
        project_path = Path(merged_params['project'])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path
            merged_params['project'] = str(project_path)
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"PermissionError: 无权限创建目录 {project_path}")
            raise ValueError(f"PermissionError: 无权限创建目录 {project_path}")
        logger.info(f"标准化project路径, {merged_params['project']}")

    # 6. 分离yolo_args和 project_args
    for key, value in merged_params.items():
        setattr(project_args, key, value)
        if key in valid_args:
            setattr(yolo_args, key, value)
            # 标记通过YAML文件设置且未被命令行覆盖的参数
        if yaml_config and key in yaml_config and not hasattr(project_args, f"{key}_specified"):
            setattr(project_args, f"{key}_specified", False)

    # 7.参数验证,先pass
    _validate_final_args(project_args, mode)
    logger.info(f"模式 '{mode}' 的所有参数已成功合并和验证。")
    # 返回分离课后的两组参数
    return yolo_args, project_args


if __name__ == '__main__':
    load_config(config_type="train")
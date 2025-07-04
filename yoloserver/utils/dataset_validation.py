#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset_validation.py
# @Time      :2025/6/26 10:30:00
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :包含所有具体的、可复用的数据验证逻辑函数
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from yoloserver.utils.performance_utils import time_it
import yaml

# 配置验证模式和参数
SAMPLE_SIZE = 0.1  # 抽样比例 10%
MIN_SAMPLES = 10   # 最少抽样 10 个

# 日志临时记录
logger = logging.getLogger("YOLO_DatasetVerification")
@time_it(name="数据集配置验证", logger_instance=logger)
def verify_dataset_config(yaml_path: Path, current_logger: logging.Logger, mode: str, task_type: str) -> Tuple[
    bool, List[Dict]]:
    """
    根据 data.yaml 文件验证数据集的结构、内容和配置。

    :param yaml_path: data.yaml 文件的路径。
    :param current_logger: 用于记录消息的 Logger 实例。
    :param mode: 验证模式，“FULL” 或 “SAMPLE”。
    :param task_type: 任务类型，“detection”或“segmentation”。
    :return: 一个包含以下内容的元组：
             - bool：如果验证通过，则为 True，否则为 False。
             - List[Dict]：字典列表，每个字典描述一个无效文件。
    """
    invalid_data_list = []
    failed_file_stems: Set[str] = set()  # 为避免同一文件出现重复错误报告

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
        current_logger.info(f"成功加载并解析 'data.yaml' 文件: {yaml_path}")
    except FileNotFoundError:
        current_logger.critical(f"'data.yaml' 文件未找到: {yaml_path}")
        return False, []
    except yaml.YAMLError as e:
        current_logger.critical(f"'data.yaml' 文件格式错误，无法解析: {e}")
        return False, []

    # 1.验证基本 YAML 结构和 nc/names 一致性
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    if not all(key in data_cfg for key in required_keys):
        current_logger.error(f"'data.yaml' 缺少必要键值。需要: {required_keys}")
        return False, []

    nc = data_cfg.get('nc')
    names = data_cfg.get('names')
    if not isinstance(nc, int) or not isinstance(names, list) or nc != len(names):
        current_logger.error(f"类别数量不匹配: 'nc' ({nc}) 与 'names' 列表长度 ({len(names)}) 不一致。")
        # 这是一个严重错误，但我们可以继续根据定义的 nc 检查文件内容
        # return False， [] # 或者选择在此处停止

# 2.迭代数据分片（train、val、test）
    for split in ['train', 'val', 'test']:
        if split not in data_cfg or not data_cfg[split]:
            current_logger.info(f"在 'data.yaml' 中跳过 '{split}' 数据集（未定义或路径为空）。")
            continue

        current_logger.info(f"--- 开始验证 '{split}' 数据集 ---")
        image_dir = Path(data_cfg[split])
        if not image_dir.is_absolute():
            image_dir = yaml_path.parent / image_dir

        if not image_dir.exists() or not image_dir.is_dir():
            current_logger.error(f"'{split}' 图像目录不存在: {image_dir}")
            invalid_data_list.append(
                {'image_path': str(image_dir), 'label_path': None, 'error_message': 'Image directory not found'})
            continue

        image_files = list(image_dir.glob('*.*'))
        image_files = [f for f in image_files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]

        if not image_files:
            current_logger.warning(f"'{split}' 图像目录 '{image_dir}' 为空或不包含支持的图像格式。")
            continue

   # 3.如果处于 “SAMPLE” 模式，则为样本文件
        if mode.upper() == "SAMPLE":
            sample_size = max(MIN_SAMPLES, int(len(image_files) * SAMPLE_SIZE))
            files_to_check = random.sample(image_files, min(sample_size, len(image_files)))
            current_logger.info(f"抽样模式: 从 {len(image_files)} 个文件中随机抽取 {len(files_to_check)} 个进行验证。")
        else:
            files_to_check = image_files
            current_logger.info(f"完整模式: 将验证所有 {len(image_files)} 个文件。")

        # 4.检查每张图片及其对应的标签
        for image_path in files_to_check:
            if image_path.stem in failed_file_stems:
                continue

            label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"

            if not label_path.exists():
                error_msg = "缺少对应的标签文件。"
                current_logger.warning(f"文件无效: {image_path.name} -> {error_msg}")
                invalid_data_list.append(
                    {'image_path': str(image_path), 'label_path': str(label_path), 'error_message': error_msg})
                failed_file_stems.add(image_path.stem)
                continue

            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if not lines:
                    error_msg = "标签文件为空。"
                    current_logger.warning(f"文件无效: {label_path.name} -> {error_msg}")
                    invalid_data_list.append(
                        {'image_path': str(image_path), 'label_path': str(label_path), 'error_message': error_msg})
                    failed_file_stems.add(image_path.stem)
                    continue

                for i, line in enumerate(lines):
                    parts = line.strip().split()

                   # 根据任务类型验证 Part 数量
                    if task_type == 'detection':
                        if len(parts) != 5:
                            raise ValueError(f"行 {i + 1}: 检测任务需要5个值，但找到 {len(parts)} 个。")
                    elif task_type == 'segmentation':
                        if len(parts) < 7 or len(parts) % 2 == 0:
                            raise ValueError(f"行 {i + 1}: 分割任务需要奇数个值 (>=7)，但找到 {len(parts)} 个。")

                    values = [float(p) for p in parts]
                    class_id, coords = int(values[0]), values[1:]

                    # 验证 class_id
                    if not (0 <= class_id < nc):
                        raise ValueError(f"行 {i + 1}: 类别ID '{class_id}' 超出范围 [0, {nc - 1}]。")

                    # 验证坐标
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        raise ValueError(f"行 {i + 1}: 至少一个坐标值不在 [0.0, 1.0] 范围内。")

            except (ValueError, IndexError) as e:
                error_msg = f"标签文件内容错误: {e}"
                current_logger.warning(f"文件无效: {label_path.name} -> {error_msg}")
                invalid_data_list.append(
                    {'image_path': str(image_path), 'label_path': str(label_path), 'error_message': error_msg})
                failed_file_stems.add(image_path.stem)

    is_valid = len(invalid_data_list) == 0
    return is_valid, invalid_data_list

@time_it(name="数据集分割验证", logger_instance=logger)
def verify_split_uniqueness(yaml_path: Path, current_logger: logging.Logger) -> bool:
    """
    检查 train、val 和 test splits 之间的重叠图像文件。

    :param yaml_path：data.yaml 文件的路径。
    :param current_logger: 用于记录消息的 Logger 实例。
    :return: 如果未找到重叠，则为 True，否则为 False。
    """
    current_logger.info("--- 开始验证数据集分割唯一性 ---")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        current_logger.error("无法加载或解析 'data.yaml'，跳过唯一性验证。")
        return False

    sets = {}
    for split in ['train', 'val', 'test']:
        if split in data_cfg and data_cfg[split]:
            image_dir = Path(data_cfg[split])
            if not image_dir.is_absolute():
                image_dir = yaml_path.parent / image_dir

            if image_dir.exists():
                sets[split] = {p.stem for p in image_dir.glob('*.*') if
                               p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')}
            else:
                sets[split] = set()

    is_unique = True
   # 检查 train-val 重叠
    if 'train' in sets and 'val' in sets:
        train_val_overlap = sets['train'].intersection(sets['val'])
        if train_val_overlap:
            is_unique = False
            current_logger.error(f"发现 {len(train_val_overlap)} 个文件在 'train' 和 'val' 数据集中重复。")
            current_logger.debug(f"重复文件示例: {list(train_val_overlap)[:5]}")

    # 检查 train-test 重叠
    if 'train' in sets and 'test' in sets:
        train_test_overlap = sets['train'].intersection(sets['test'])
        if train_test_overlap:
            is_unique = False
            current_logger.error(f"发现 {len(train_test_overlap)} 个文件在 'train' 和 'test' 数据集中重复。")
            current_logger.debug(f"重复文件示例: {list(train_test_overlap)[:5]}")

    # 检查 val-test 重叠
    if 'val' in sets and 'test' in sets:
        val_test_overlap = sets['val'].intersection(sets['test'])
        if val_test_overlap:
            is_unique = False
            current_logger.error(f"发现 {len(val_test_overlap)} 个文件在 'val' 和 'test' 数据集中重复。")
            current_logger.debug(f"重复文件示例: {list(val_test_overlap)[:5]}")

    if is_unique:
        current_logger.info("数据集分割唯一性验证通过，各部分之间无重复文件。")

    return is_unique


def delete_invalid_files(invalid_data_list: list, current_logger: logging.Logger):
    """
   根据提供的列表删除无效的图像和标签文件。

    :param invalid_data_list:verify_dataset_config 中的词典列表。
    :param current_logger: 用于记录消息的 Logger 实例。
    """
    if not invalid_data_list:
        current_logger.info("没有需要删除的无效文件。")
        return

    current_logger.warning("开始删除无效的数据文件... 此操作不可逆！")
    deleted_count = 0
    for item in invalid_data_list:
        image_path_str = item.get('image_path')
        label_path_str = item.get('label_path')

        # 删除标签文件
        if label_path_str:
            label_path = Path(label_path_str)
            if label_path.exists():
                try:
                    label_path.unlink()
                    current_logger.info(f"已删除标签文件: {label_path}")
                    deleted_count += 1
                except OSError as e:
                    current_logger.error(f"删除标签文件失败: {label_path}, 错误: {e}")

         # 删除图片文件
        if image_path_str:
            image_path = Path(image_path_str)
            if image_path.exists():
                try:
                    image_path.unlink()
                    current_logger.info(f"已删除图像文件: {image_path}")
                    deleted_count += 1
                except OSError as e:
                    current_logger.error(f"删除图像文件失败: {image_path}, 错误: {e}")

    current_logger.info(f"删除操作完成。共删除 {deleted_count} 个文件（图像/标签）。")
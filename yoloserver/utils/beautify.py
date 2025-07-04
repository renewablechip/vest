#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName  : beautify_vectorized.py
# @Time      : 2025/6/29
# @Author    : 优化版本
# @Function  : YOLO 检测结果美化绘制（向量化优化版本）

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import OrderedDict
import os
import time

text_size_cache = OrderedDict()


# ======================= 参数计算函数（现在接收用户自定义的基准值） ======================
def calculate_beautify_params(img_width, img_height,
                              user_font_path,
                              user_base_font_size,
                              user_base_line_width,
                              user_base_label_padding_x,
                              user_base_label_padding_y,
                              user_base_radius,
                              ref_dim_for_scaling,
                              text_color,
                              label_mapping,
                              color_mapping, ):
    """
    根据输入图像的原始分辨率和用户提供的基准值，计算实际使用的美化参数。
    同时进行字体缓存的预加载。
    """
    current_short_dim = min(img_width, img_height)
    scale_factor = current_short_dim / ref_dim_for_scaling if ref_dim_for_scaling > 0 else 1.0
    # 使用用户提供的基准值进行缩放
    actual_font_size = max(10, int(user_base_font_size * scale_factor))

    actual_line_width = max(1, int(user_base_line_width * scale_factor))
    actual_label_padding_x = max(5, int(user_base_label_padding_x * scale_factor))
    actual_label_padding_y = max(5, int(user_base_label_padding_y * scale_factor))
    actual_radius = max(3, int(user_base_radius * scale_factor))

    # 预加载字体缓存
    font_sizes_to_preload = generate_preload_font_sizes(
        base_font_size=actual_font_size,
        ref_dim_for_base_font=ref_dim_for_scaling,
        current_display_short_dim=current_short_dim
    )
    preload_cache(user_font_path, font_sizes_to_preload, label_mapping)

    return {
        "font_path": user_font_path,
        "font_size": actual_font_size,
        "line_width": actual_line_width,
        "label_padding_x": actual_label_padding_x,
        "label_padding_y": actual_label_padding_y,
        "radius": actual_radius,
        "text_color_bgr": text_color,
        "label_mapping": label_mapping,
        "color_mapping": color_mapping,
    }


# ======================= 向量化辅助函数 ======================
def vectorized_text_processing(labels, confs, use_chinese_mapping, label_mapping, font_obj):
    """
    向量化处理所有标签文本，批量计算文本尺寸
    """
    # 批量构建显示标签
    if use_chinese_mapping:
        display_labels = [label_mapping.get(label, label) for label in labels]
    else:
        display_labels = labels

    # 批量构建完整文本
    label_texts_full = [f"{display_label} {conf * 100:.1f}%"
                        for display_label, conf in zip(display_labels, confs)]

    # 批量计算文本尺寸
    text_sizes = batch_get_text_sizes(label_texts_full, font_obj)

    return label_texts_full, text_sizes


def batch_get_text_sizes(texts, font_obj, max_cache_size=500):
    """
    批量计算文本尺寸，优化缓存使用
    """
    text_sizes = []
    uncached_texts = []
    uncached_indices = []

    # 第一遍：检查缓存
    for i, text in enumerate(texts):
        # 规范化文本用于缓存
        parts = text.split(" ")
        if len(parts) > 1 and parts[-1].endswith('%'):
            label_part = " ".join(parts[:-1])
            normalized_text = f"{label_part} 80.0%"
        else:
            normalized_text = text

        cache_key = f"{normalized_text}_{font_obj.size}"

        if cache_key in text_size_cache:
            text_sizes.append(text_size_cache[cache_key])
        else:
            text_sizes.append(None)  # 占位符
            uncached_texts.append(text)
            uncached_indices.append(i)

    # 第二遍：批量计算未缓存的文本
    if uncached_texts:
        temp_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_image)

        for text, idx in zip(uncached_texts, uncached_indices):
            bbox = draw.textbbox((0, 0), text, font=font_obj)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = (width, height)

            # 更新缓存
            parts = text.split(" ")
            if len(parts) > 1 and parts[-1].endswith('%'):
                label_part = " ".join(parts[:-1])
                normalized_text = f"{label_part} 80.0%"
            else:
                normalized_text = text
            cache_key = f"{normalized_text}_{font_obj.size}"
            text_size_cache[cache_key] = size

            # 更新结果
            text_sizes[idx] = size

        # 清理缓存
        if len(text_size_cache) > max_cache_size:
            # 删除最老的项目
            items_to_remove = len(text_size_cache) - max_cache_size
            for _ in range(items_to_remove):
                text_size_cache.popitem(last=False)

    return text_sizes


def vectorized_layout_calculation(boxes, text_sizes, label_padding_x, label_padding_y,
                                  radius, line_width, img_width, img_height):
    """
    向量化计算所有检测框和标签框的布局参数
    """
    boxes = np.array(boxes)
    text_sizes = np.array(text_sizes)

    # 基本参数计算
    text_widths = text_sizes[:, 0]
    text_heights = text_sizes[:, 1]

    label_box_widths = text_widths + 2 * label_padding_x
    label_box_heights = text_heights + 2 * label_padding_y

    # 确保标签框宽度至少能容纳圆角
    label_box_widths = np.maximum(label_box_widths, 2 * radius)

    # 检测框坐标
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 标签框位置计算
    label_box_x_mins = (x1 - line_width // 2).astype(int)

    # 标签位置决策（向量化）
    label_box_y_mins_above = y1 - label_box_heights
    can_place_above = label_box_y_mins_above >= 0

    det_box_heights = y2 - y1
    can_place_inside = det_box_heights >= (label_box_heights + line_width * 2)

    # 决定标签位置
    draw_label_inside = np.logical_and(~can_place_above, can_place_inside)
    draw_label_below = np.logical_and(~can_place_above, ~can_place_inside)

    # 计算最终的标签框位置
    label_box_y_mins = np.where(
        can_place_above,
        label_box_y_mins_above,
        np.where(
            draw_label_inside,
            (y1 - line_width / 2).astype(int),
            (y2 + line_width).astype(int)
        )
    )

    label_box_y_maxs = np.where(
        can_place_above,
        y1.astype(int),
        np.where(
            draw_label_inside,
            (y1 + label_box_heights).astype(int),
            np.minimum((y2 + line_width + label_box_heights).astype(int), img_height)
        )
    )

    # 处理超出右边界的情况
    label_box_x_maxs = label_box_x_mins + label_box_widths
    align_right = label_box_x_maxs > img_width

    # 重新计算需要右对齐的标签框位置
    label_box_x_mins = np.where(
        align_right,
        np.maximum((x2 + line_width // 2 - label_box_widths).astype(int), 0),
        label_box_x_mins
    )

    # 判断标签框是否比检测框宽
    det_box_widths = x2 - x1
    is_label_wider = label_box_widths > det_box_widths

    return {
        'label_box_x_mins': label_box_x_mins.astype(int),
        'label_box_y_mins': label_box_y_mins.astype(int),
        'label_box_y_maxs': label_box_y_maxs.astype(int),
        'label_box_widths': label_box_widths.astype(int),
        'label_box_heights': label_box_heights.astype(int),
        'text_widths': text_widths.astype(int),
        'text_heights': text_heights.astype(int),
        'draw_label_inside': draw_label_inside,
        'draw_label_below': draw_label_below,
        'align_right': align_right,
        'is_label_wider': is_label_wider
    }


def calculate_corner_states(layout_params, boxes):
    """
    向量化计算所有检测框和标签框的圆角状态
    """
    n_boxes = len(boxes)

    # 标签框圆角状态
    label_corners = {
        'top_left': np.ones(n_boxes, dtype=bool),
        'top_right': np.ones(n_boxes, dtype=bool),
        'bottom_left': np.ones(n_boxes, dtype=bool),
        'bottom_right': np.ones(n_boxes, dtype=bool)
    }

    # 检测框圆角状态
    det_corners = {
        'top_left': np.ones(n_boxes, dtype=bool),
        'top_right': np.ones(n_boxes, dtype=bool),
        'bottom_left': np.ones(n_boxes, dtype=bool),
        'bottom_right': np.ones(n_boxes, dtype=bool)
    }

    draw_label_inside = layout_params['draw_label_inside']
    draw_label_below = layout_params['draw_label_below']
    align_right = layout_params['align_right']
    is_label_wider = layout_params['is_label_wider']

    # 标签在上方的情况
    above_mask = ~draw_label_inside & ~draw_label_below

    # 标签在上方且右对齐
    mask = above_mask & align_right
    label_corners['bottom_left'][mask] = is_label_wider[mask]
    label_corners['bottom_right'][mask] = False
    det_corners['top_left'][mask] = is_label_wider[mask]
    det_corners['top_right'][mask] = False

    # 标签在上方且左对齐
    mask = above_mask & ~align_right
    label_corners['bottom_left'][mask] = False
    label_corners['bottom_right'][mask] = is_label_wider[mask]
    det_corners['top_left'][mask] = False
    det_corners['top_right'][mask] = ~is_label_wider[mask]

    # 标签在下方且右对齐
    mask = draw_label_below & align_right
    label_corners['top_left'][mask] = is_label_wider[mask]
    label_corners['top_right'][mask] = False
    det_corners['bottom_left'][mask] = is_label_wider[mask]
    det_corners['bottom_right'][mask] = False

    # 标签在下方且左对齐
    mask = draw_label_below & ~align_right
    label_corners['top_left'][mask] = False
    label_corners['top_right'][mask] = is_label_wider[mask]
    det_corners['bottom_left'][mask] = False
    det_corners['bottom_right'][mask] = is_label_wider[mask]

    # 标签在内部
    mask = draw_label_inside
    det_corners['top_left'][mask] = False
    det_corners['top_right'][mask] = False

    # 标签在内部且右对齐
    mask = draw_label_inside & align_right
    label_corners['bottom_left'][mask] = is_label_wider[mask]
    label_corners['bottom_right'][mask] = False

    # 标签在内部且左对齐
    mask = draw_label_inside & ~align_right
    label_corners['bottom_left'][mask] = False
    label_corners['bottom_right'][mask] = True

    return label_corners, det_corners


# ======================= 批量绘制函数 ======================
def batch_draw_rectangles(image_np, rectangles_data, radius):
    """
    批量绘制矩形，减少函数调用开销
    """
    for rect_data in rectangles_data:
        if rect_data['type'] == 'filled':
            draw_filled_rounded_rect(
                image_np, rect_data['pt1'], rect_data['pt2'],
                rect_data['color'], radius, **rect_data['corners']
            )
        elif rect_data['type'] == 'bordered':
            draw_bordered_rounded_rect(
                image_np, rect_data['pt1'], rect_data['pt2'],
                rect_data['color'], rect_data['thickness'], radius,
                **rect_data['corners']
            )


# ======================= 美化辅助函数（基于OpenCV） ======================
def draw_filled_rounded_rect(image_np, pt1, pt2, color_bgr, radius,
                             top_left_round=True, top_right_round=True,
                             bottom_left_round=True, bottom_right_round=True):
    """使用 OpenCV 绘制颜色填充的圆角矩形，可控制每个角的圆角状态"""
    x1, y1 = pt1
    x2, y2 = pt2
    thickness = -1

    # 绘制矩形部分
    cv2.rectangle(image_np,
                  (x1 + (radius if top_left_round else 0), y1),
                  (x2 - (radius if top_right_round else 0), y1 + radius),
                  color_bgr, thickness)
    cv2.rectangle(image_np,
                  (x1 + (radius if bottom_left_round else 0), y2 - radius),
                  (x2 - (radius if bottom_right_round else 0), y2),
                  color_bgr, thickness)
    cv2.rectangle(image_np,
                  (x1, y1 + (radius if top_left_round or top_right_round else 0)),
                  (x2, y2 - (radius if bottom_left_round or bottom_right_round else 0)),
                  color_bgr, thickness)

    # 绘制圆角
    if top_left_round:
        cv2.circle(image_np, (x1 + radius, y1 + radius), radius, color_bgr, thickness, cv2.LINE_AA)
    if top_right_round:
        cv2.circle(image_np, (x2 - radius, y1 + radius), radius, color_bgr, thickness, cv2.LINE_AA)
    if bottom_left_round:
        cv2.circle(image_np, (x1 + radius, y2 - radius), radius, color_bgr, thickness, cv2.LINE_AA)
    if bottom_right_round:
        cv2.circle(image_np, (x2 - radius, y2 - radius), radius, color_bgr, thickness, cv2.LINE_AA)


def draw_bordered_rounded_rect(image_np, pt1, pt2, color_bgr, thickness, radius,
                               top_left_round=True, top_right_round=True,
                               bottom_left_round=True, bottom_right_round=True):
    """使用 OpenCV 绘制带边框的圆角矩形，可控制每个角的圆角状态"""
    x1, y1 = pt1
    x2, y2 = pt2
    line_type = cv2.LINE_AA

    # 绘制直线部分
    cv2.line(image_np,
             (x1 + (radius if top_left_round else 0), y1),
             (x2 - (radius if top_right_round else 0), y1),
             color_bgr, thickness, line_type)
    cv2.line(image_np,
             (x1 + (radius if bottom_left_round else 0), y2),
             (x2 - (radius if bottom_right_round else 0), y2),
             color_bgr, thickness, line_type)
    cv2.line(image_np,
             (x1, y1 + (radius if top_left_round else 0)),
             (x1, y2 - (radius if bottom_left_round else 0)),
             color_bgr, thickness, line_type)
    cv2.line(image_np,
             (x2, y1 + (radius if top_right_round else 0)),
             (x2, y2 - (radius if bottom_right_round else 0)),
             color_bgr, thickness, line_type)

    # 绘制圆角
    if top_left_round:
        cv2.ellipse(image_np, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color_bgr, thickness, line_type)
    if top_right_round:
        cv2.ellipse(image_np, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color_bgr, thickness, line_type)
    if bottom_left_round:
        cv2.ellipse(image_np, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color_bgr, thickness, line_type)
    if bottom_right_round:
        cv2.ellipse(image_np, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color_bgr, thickness, line_type)


# ======================= 文本和缓存辅助函数（基于Pillow） ======================
def generate_preload_font_sizes(base_font_size, ref_dim_for_base_font, current_display_short_dim, buffer_range=2):
    """
    根据当前显示分辨率和基准字体大小，生成用于预加载的字体大小列表。
    """
    font_sizes_set = set()
    scale_factor = current_display_short_dim / ref_dim_for_base_font if ref_dim_for_base_font > 0 else 1.0
    scaled_base_font_size = int(base_font_size * scale_factor)

    for i in range(-buffer_range, buffer_range + 1):
        buffered_size = max(10, scaled_base_font_size + i)
        font_sizes_set.add(buffered_size)
    return sorted(list(font_sizes_set))


def preload_cache(font_path, font_sizes_list, label_mapping):
    """预缓存中英文标签尺寸"""
    global text_size_cache
    text_size_cache.clear()
    for size in font_sizes_list:
        try:
            font = ImageFont.truetype(font_path, size)
        except IOError:
            print(f"警告：无法加载字体文件 '{font_path}'。跳过字体大小 {size} 的预缓存。")
            continue

        for label_val in list(label_mapping.values()) + list(label_mapping.keys()):
            text = f"{label_val} 80.0%"
            cache_key = f"{text}_{size}"
            temp_image = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(temp_image)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_size_cache[cache_key] = (bbox[2] - bbox[0], bbox[3] - bbox[1])


# ======================= 核心绘制函数（向量化版本） ======================
def custom_plot(
        image,
        boxes,
        confs,
        labels,
        use_chinese_mapping,
        font_path,
        font_size,
        line_width,
        label_padding_x,
        label_padding_y,
        radius,
        text_color_bgr,
        label_mapping,
        color_mapping,
):
    """绘制检测框和标签 (向量化优化版本)"""
    if len(boxes) == 0:
        return image.copy()

    result_image_cv = image.copy()
    img_height, img_width = image.shape[:2]

    try:
        font_pil = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"错误：无法加载字体文件 '{font_path}'。将使用Pillow默认字体。")
        font_pil = ImageFont.load_default()

    # 1. 向量化处理文本
    label_texts_full, text_sizes = vectorized_text_processing(
        labels, confs, use_chinese_mapping, label_mapping, font_pil
    )

    # 2. 向量化计算布局
    layout_params = vectorized_layout_calculation(
        boxes, text_sizes, label_padding_x, label_padding_y,
        radius, line_width, img_width, img_height
    )

    # 3. 向量化计算圆角状态
    label_corners, det_corners = calculate_corner_states(layout_params, boxes)

    # 4. 准备批量绘制数据
    rectangles_to_draw = []
    texts_to_draw = []

    for i, (box, label_key) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        color_bgr = color_mapping.get(label_key, (0, 255, 0))

        # 检测框数据
        rectangles_to_draw.append({
            'type': 'bordered',
            'pt1': (x1, y1),
            'pt2': (x2, y2),
            'color': color_bgr,
            'thickness': line_width,
            'corners': {
                'top_left_round': det_corners['top_left'][i],
                'top_right_round': det_corners['top_right'][i],
                'bottom_left_round': det_corners['bottom_left'][i],
                'bottom_right_round': det_corners['bottom_right'][i]
            }
        })

        # 标签框数据
        label_x_min = layout_params['label_box_x_mins'][i]
        label_y_min = layout_params['label_box_y_mins'][i]
        label_x_max = label_x_min + layout_params['label_box_widths'][i]
        label_y_max = layout_params['label_box_y_maxs'][i]

        rectangles_to_draw.append({
            'type': 'filled',
            'pt1': (label_x_min, label_y_min),
            'pt2': (label_x_max, label_y_max),
            'color': color_bgr,
            'corners': {
                'top_left_round': label_corners['top_left'][i],
                'top_right_round': label_corners['top_right'][i],
                'bottom_left_round': label_corners['bottom_left'][i],
                'bottom_right_round': label_corners['bottom_right'][i]
            }
        })

        # 文本位置
        text_x = label_x_min + (layout_params['label_box_widths'][i] - layout_params['text_widths'][i]) // 2
        text_y = label_y_min + (layout_params['label_box_heights'][i] - layout_params['text_heights'][i]) // 2

        texts_to_draw.append({
            'text': label_texts_full[i],
            'position': (text_x, text_y),
            'fonts': font_pil,
            'fill_bgr': text_color_bgr
        })

    # 5. 批量绘制矩形
    batch_draw_rectangles(result_image_cv, rectangles_to_draw, radius)

    # 6. 批量绘制文本
    if texts_to_draw:
        image_pil = Image.fromarray(cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        for text_info in texts_to_draw:
            fill_rgb = (text_info['fill_bgr'][2], text_info['fill_bgr'][1], text_info['fill_bgr'][0])
            draw.text(text_info['position'], text_info['text'], font=text_info['fonts'], fill=fill_rgb)

        result_image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

    return result_image_cv
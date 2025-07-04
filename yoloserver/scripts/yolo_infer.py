#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_infer.py
# @Time      :2025/6/27 09:52:40
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :模型推理，支持视频，摄像头，文件，多参数保存，统一输出目录，动态美化参数
import argparse
import glob
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import cv2
import sys

# 将项目根目录（BTD）添加到系统搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloserver.utils.infer_frame import process_frame # 处理单帧图像的
from yoloserver.utils.logging_utils import setup_logger, log_parameters
from yoloserver.utils.config_utils import load_config,merge_config,parse_color_string
from yoloserver.utils.system_utils import log_device_info

from yoloserver.utils.beautify import (calculate_beautify_params)
from yoloserver.utils.paths import YOLOSERVER_ROOT,LOGS_DIR,CHECKPOINTS_DIR
from yoloserver.utils.multithread_inferer import MultiThreadInferer,process_single_image_mp,load_tensorrt_model


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference")
    parser.add_argument("--weights", type=str,
            default=r"C:\Users\Administrator\vest\yoloserver\models\checkpoints\best_2.pt", help="模型权重信息路径")
    parser.add_argument("--source", type=str,
            default=r"C:\Users\Administrator\vest\vest.mp4", help="推理数据源")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图片尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--save", type=bool, default=True, help="保存推理结果")
    parser.add_argument("--save_txt", type=bool, default=True, help="保存推理结果为txt")
    parser.add_argument("--save_conf", type=bool, default=True, help="保存推理结果为txt")
    parser.add_argument("--save_crop", type=bool, default=True, help="保存推理结果为图片")
    parser.add_argument("--save_frames", type=bool, default=True, help="保存推理结果为图片")
    parser.add_argument("--display-size", type=int, default=720,
                        choices=[360,480,720, 1280, 1440],help="摄像头/视频显示分辨率")
    parser.add_argument("--beautify", type=bool, default=True,help="启用美化绘制")
    parser.add_argument("--fonts-size", type=int, default=26, help="美化字体大小（覆盖自动调整）")
    parser.add_argument("--line-width", type=int, default=4, help="美化线宽（覆盖自动调整）")
    parser.add_argument("--label-padding-x", type=int, default=10, help="美化标签水平内边距（覆盖自动调整）")
    parser.add_argument("--label-padding-y", type=int, default=10, help="美化标签垂直内边距（覆盖自动调整）")
    parser.add_argument("--radius", type=int, default=4, help="美化圆角半径（覆盖自动调整）")
    parser.add_argument("--use-chinese-mapping", type=bool, default=True, help="启用中文映射")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")

    # 新增多线程相关参数
    parser.add_argument("--use-multithread", type=bool, default=False, help="启用多线程推理")
    parser.add_argument("--queue-maxsize", type=int, default=10, help="队列最大大小")
    parser.add_argument("--multithread-batch-size", type=int, default=8, help="多线程推理批次大小")
    # 增加这个新参数
    parser.add_argument("--num-threads", type=int, default=4,help="用于图片批处理的工作线程数")

    # 新增TensorRT相关参数
    parser.add_argument("--use-tensorrt", type=bool, default=False, help="启用TensorRT优化")
    parser.add_argument("--tensorrt-workspace", type=int, default=4, help="TensorRT工作空间大小(GB)")
    parser.add_argument("--tensorrt-precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "int8"], help="TensorRT精度模式")
    parser.add_argument("--tensorrt-device", type=int, default=0, help="TensorRT使用的GPU设备ID")

    return parser.parse_args()

def main():
    args = parse_args()
    resolution_map = {
    360: (640, 360),
    720: (1280, 720),
    1080: (1920, 1080),
    1280: (2560, 1440),
    1440: (3840, 2160)
    }
    display_width, display_height = resolution_map[args.display_size]

    # 加载YAML配置
    yaml_config = {}
    if args.use_yaml:
        yaml_config = load_config(config_type="infer")
    # 合并命令行和YAML参数
    yolo_args, project_args = merge_config(args, yaml_config, mode="infer")

    # 在这里，我们需要初始化 logger 才能在解析失败时打印日志
    # 提前初始化 logger
    model_name_for_log = Path(project_args.weights).stem
    logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="yolo_infer",
        model_name=model_name_for_log,
        log_level=logging.INFO,
        temp_log=False,
    )

    # 修正 project_args 中的颜色配置
    project_args.text_color = parse_color_string(project_args.text_color)
    if project_args.color_mapping:
        for key, value in project_args.color_mapping.items():
            project_args.color_mapping[key] = parse_color_string(value)

    # 根据是否使用中文映射，选择不同的字体路径
    # 注意：这里我们使用 project_args，因为它整合了命令行和YAML的最终配置
    if project_args.use_chinese_mapping:
        font_path_to_use = project_args.font_path_chinese
    else:
        font_path_to_use = project_args.font_path_english

    beautify_params = calculate_beautify_params(
        img_width=display_width,
        img_height=display_height,
        user_font_path=font_path_to_use,
        user_base_font_size=project_args.font_size,
        user_base_line_width=project_args.line_width,
        user_base_label_padding_x=project_args.label_padding_x,
        user_base_label_padding_y=project_args.label_padding_y,
        user_base_radius=project_args.radius,
        ref_dim_for_scaling=project_args.display_size,
        text_color=project_args.text_color,
        label_mapping=project_args.label_mapping,
        color_mapping=project_args.color_mapping,
    )
    beautify_params["use_chinese_mapping"] = args.use_chinese_mapping
    beautify_params["text_color_bgr"] = project_args.text_color

    # logger = setup_logger(
    #     base_path=LOGS_DIR,
    #     log_type="yolo_infer",
    #     model_name=model_name,
    #     log_level=logging.INFO,
    #     temp_log=False,
    # )

    # 检查模型文件
    model_path = Path(project_args.weights)
    if not model_path.is_absolute():
        model_path = CHECKPOINTS_DIR / project_args.weights
    if not model_path.exists():
        logger.error(f"模型文件 '{model_path}' 不存在")
        raise ValueError(f"模型文件 '{model_path}' 不存在")
    logger.info(f"加载推理模型: {project_args.weights}")

    # 记录参数，设备信息，数据集信息
    logger.info("========= 参数信息 =========")
    log_parameters(project_args, logger=logger)
    logger.info("========= 设备信息 =========")
    log_device_info(logger)
    logger.info("========= 数据集信息 =========")
    logger.info(f"此次使用的数据信息为: {project_args.source}")
    logger.info(f"加载推理模型: {project_args.weights}")
    # TensorRT配置
    tensorrt_config = {
        'use_tensorrt': project_args.use_tensorrt,
        'precision': project_args.tensorrt_precision,
        'workspace': project_args.tensorrt_workspace,
        'device': project_args.tensorrt_device
    } if project_args.use_tensorrt else None


    model = YOLO(str(model_path))

    # 核心推理
    # 确保 source 是一个字符串，这样后续的方法调用才不会出错
    source = str(project_args.source)

    # 流式推理（摄像头或视频）
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
        # 判断是否使用多线程推理
        if project_args.use_multithread:
            # === 多线程推理路径 ===
            logger.info("启用多线程推理模式")

            if tensorrt_config:
                # 为视频流指定批次大小
                tensorrt_config['batch'] = project_args.multithread_batch_size

                # 在这里加载为视频优化的模型
                model = load_tensorrt_model(
                    model_path=project_args.weights,
                    tensorrt_config=tensorrt_config,
                    logger=logger
                )

            # 创建多线程推理器
            inferer = MultiThreadInferer(
                model=model,
                source=source,
                display_size=(display_width, display_height),
                window_name="YOLO vest Detection - MultiThread",
                queue_maxsize=project_args.queue_maxsize,
                logger=logger
            )

            target_device = "cuda" if project_args.use_tensorrt else ("cuda" if torch.cuda.is_available() else "cpu")
            if tensorrt_config:
                target_device = f"cuda:{tensorrt_config.get('device', 0)}"

            inferer.configure_inference(
                conf=project_args.conf,
                iou=project_args.iou,
                batch_size=project_args.multithread_batch_size,
                device=target_device)

            # 配置保存参数
            if any([project_args.save, project_args.save_frames, project_args.save_txt, project_args.save_crop]):

                # 1. 定义基础目录
                base_dir = YOLOSERVER_ROOT / "runs" / "infer"
                base_dir.mkdir(parents=True, exist_ok=True)

                # 2. 查找下一个可用的 'exp' 目录
                #    这完美模拟了 ultralytics 的行为
                exp_num = 1
                while True:
                    # 如果 exp 目录不存在，就用它
                    candidate_dir = base_dir / (f"exp{exp_num}" if exp_num > 1 else "exp")
                    if not candidate_dir.exists():
                        save_dir = candidate_dir
                        break
                    exp_num += 1

                # 3. 创建最终的保存目录
                save_dir.mkdir(parents=True)
                logger.info(f"多线程推理结果将保存至: {save_dir}")

                # 4. 将所有保存指令传递给 inferer
                inferer.configure_save(
                    save_video=project_args.save,
                    save_frames=project_args.save_frames,
                    save_txt=project_args.save_txt,
                    save_conf=project_args.save_conf,
                    save_crop=project_args.save_crop,
                    save_dir=save_dir,
                    fps=30.0  # 可以从视频源动态获取
                )

                # 打印清晰的保存结构日志
                logger.info(f"目录结构: "
                            f"labels/({'✓' if project_args.save_txt else '✗'}), "
                            f"crops/({'✓' if project_args.save_crop else '✗'}), "
                            f"frames/({'✓' if project_args.save_frames else '✗'}), "
                            f"output.mp4({'✓' if project_args.save else '✗'})")

            # 运行多线程推理
            try:
                inferer.run(process_frame, project_args, beautify_params)

                # 输出统计信息
                stats = inferer.get_stats()
                logger.info("\n========= 多线程推理统计 =========")
                for thread_name, thread_stats in stats.items():
                    logger.info(f"{thread_name}: 处理帧数={thread_stats['frame_count']}, "
                                f"平均FPS={thread_stats['avg_fps']:.2f}")

            except Exception as e:
                logger.error(f"多线程推理出错: {e}")
                raise

        else:
            if tensorrt_config:
                tensorrt_config['batch'] = 1
                model = load_tensorrt_model(
                    model_path=project_args.weights,
                    tensorrt_config=tensorrt_config,
                    logger=logger
                )
            # 设置显示窗口
            window_name = "YOLO vest Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, display_width, display_height)

            # 初始化视频捕获
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            # 流式推理 - 使用与图片推理相同的保存逻辑
            results = model.predict(
                source=source,
                imgsz=project_args.imgsz,
                conf=project_args.conf,
                iou=project_args.iou,
                save=project_args.save_txt or project_args.save_crop,  # 只有需要txt或crop时才让YOLO保存
                save_txt=project_args.save_txt,
                save_conf=project_args.save_conf,
                save_crop=project_args.save_crop,
                show=False,
                project=YOLOSERVER_ROOT / "runs" / "infer",  # 指定为项目根下的runs/infer
                name="exp",  # 指定名称为exp，会自动递增
                stream=True
            )

            video_writer = None
            frames_dir = None
            save_dir = None

            for idx, result in enumerate(results):
                # 第一帧初始化保存路径
                if idx == 0:
                    # 使用与图片推理相同的保存目录逻辑
                    save_dir = Path(result.save_dir)
                    logger.info(f"此次推理结果保存路径为: {save_dir}")

                    # 创建frames目录（用于保存美化后的帧）
                    if project_args.save_frames:
                        frames_dir = save_dir / "frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"帧保存路径为: {frames_dir}")

                    # 创建视频保存路径
                    if project_args.save:
                        video_path = save_dir / "output.mp4"
                        logger.info(f"视频保存路径为: {video_path}")
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (display_width, display_height)
                        )
                        if video_writer and video_writer.isOpened():
                            logger.info("视频写入器已创建")
                        else:
                            logger.error("视频写入器创建失败")
                            video_writer = None

                # 处理帧
                frame = result.orig_img
                annotated_frame = process_frame(
                    frame,
                    result,
                    project_args,
                    beautify_params,
                )

                # 调整帧大小用于显示和保存
                resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

                # 保存美化后的帧图像到frames目录
                if project_args.save_frames and frames_dir:
                    frame_path = frames_dir / f"{idx}.jpg"
                    cv2.imwrite(str(frame_path), resized_frame)

                # 保存到输出视频
                if video_writer and video_writer.isOpened():
                    video_writer.write(resized_frame)

                # 显示帧
                cv2.imshow(window_name, resized_frame)

                # 按 q 或 Esc 退出
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            # 释放资源
            cap.release()
            if video_writer and video_writer.isOpened():
                logger.info("视频写入器已释放资源")
                video_writer.release()
            cv2.destroyAllWindows()
            logger.info(f"{'摄像头' if source.isdigit() else '视频'}推理完成，结果已保存至: {save_dir or '未保存'}")

    else:
        # --- 图片处理逻辑 ---
        if tensorrt_config:
            # 为图片处理指定批次大小为 1
            tensorrt_config['batch'] = 1

            # 在主进程中先生成 batch=1 的引擎，确保它存在
            logger.info("正在为图片处理准备 batch=1 的 TensorRT 引擎...")
            model = load_tensorrt_model(
                model_path=project_args.weights,
                tensorrt_config=tensorrt_config,
                logger=logger
            )
        if project_args.use_multithread:
            logger.info(f"启用多进程图片处理模式，使用 {project_args.num_threads} 个工作进程。")

            source_path = Path(source)
            if source_path.is_dir():
                image_patterns = [str(source_path / ext) for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]]
            else:
                image_patterns = [str(source_path)]

            image_paths = []
            for pattern in image_patterns:
                image_paths.extend(glob.glob(pattern))

            if not image_paths:
                logger.warning(f"在 '{source}' 中未找到可处理的图片。")
                return

            base_dir = YOLOSERVER_ROOT / "runs" / "infer"
            exp_num = 1
            while True:
                candidate_dir = base_dir / (f"exp{exp_num}" if exp_num > 1 else "exp")
                if not candidate_dir.exists():
                    save_dir = candidate_dir
                    break
                exp_num += 1
            save_dir.mkdir(parents=True, exist_ok=True)

            # 模型路径
            model_path_for_workers = Path(project_args.weights)
            if project_args.use_tensorrt:
                # 构造预期的 .engine 文件路径
                engine_path = model_path_for_workers.with_name(f"{model_path_for_workers.stem}_batch1.engine")
                # 检查 .engine 文件是否真实存在 (在之前的 load_tensorrt_model 步骤中已生成)
                if engine_path.exists():
                    logger.info(f"多进程模式：检测到 TensorRT 引擎，将使用 '{engine_path}'。")
                    model_path_for_workers = str(engine_path)
                else:
                    logger.warning(
                        f"TensorRT 已启用，但未找到引擎文件 '{engine_path}'。"
                        f"子进程将回退到使用原始模型 '{model_path_for_workers}'。"
                    )
            else:
                model_path_for_workers = str(model_path_for_workers)
            # 改成 Namespace 对象（multiprocessing 是可以序列化 argparse.Namespace 的）
            project_args_ns = argparse.Namespace(**{
                "beautify": project_args.beautify,
                "save": project_args.save,
                "save_txt": project_args.save_txt,
                "save_conf": project_args.save_conf,
                "save_crop": project_args.save_crop,
            })

            # 封装参数给子进程
            tasks = [
                (image_path, str(save_dir), model_path_for_workers, project_args_ns, beautify_params)
                for image_path in image_paths
            ]

            from multiprocessing import Pool
            with Pool(processes=project_args.num_threads) as pool:
                pool.map(process_single_image_mp, tasks)

            logger.info(f"多进程推理完成，结果已保存至: {save_dir}")
        else:
            results = model.predict(
                source=source,
                imgsz=project_args.imgsz,
                conf=project_args.conf,
                iou=project_args.iou,
                save=project_args.save,
                save_txt=project_args.save_txt,
                save_conf=project_args.save_conf,
                save_crop=project_args.save_crop,
                show=False,
                project= YOLOSERVER_ROOT / "runs" / "infer",
                name="exp",
            )
            save_dir = Path(results[0].save_dir)
            # 美化输出
            if project_args.beautify:
                bea_save_dir = save_dir / "beautified"
                bea_save_dir.mkdir(parents=True, exist_ok=True)
                for idx,result in enumerate(results):
                    annotated_frame = process_frame(
                        result.orig_img, result, project_args, beautify_params
                    )
                    if args.save:
                        cv2.imwrite(str(bea_save_dir/ f"{idx}.png"), annotated_frame)
            logger.info(f"推理完成，结果已保存至: {save_dir}")
        logger.info("推理结束".center(80, "="))


if __name__ == "__main__":
    main()

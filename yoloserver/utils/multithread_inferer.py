#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :multithread_inferer.py
# @Time      :2025/6/30
# @Author    :Assistant
# @Function  :封装三线程YOLO推理功能 - 支持批量推理
import shutil

import cv2
import time
import threading
import queue
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
import logging
import os
from yoloserver.utils.infer_frame import process_frame # 处理单帧图像的


def process_single_image_mp(image_path_and_args):
    image_path, base_save_dir, model_path, project_args, beautify_params = image_path_and_args

    # 必须每个进程里重新加载模型
    model = YOLO(model_path,task='detect')

    try:
        result = model.predict(source=image_path, stream=False, verbose=False)[0]

        if project_args.save:
            pred_save_dir = Path(base_save_dir)  # runs/infer/exp/
            pred_save_dir.mkdir(parents=True, exist_ok=True)
            pred_img = result.plot()  # 带框图像（未美化）
            filename = Path(image_path).name
            save_path = pred_save_dir / filename
            cv2.imwrite(str(save_path), pred_img)

        if project_args.beautify:
            bea_save_dir = Path(base_save_dir) / "beautified"
            bea_save_dir.mkdir(parents=True, exist_ok=True)
            annotated_frame = process_frame(result.orig_img, result, project_args, beautify_params)
            if project_args.save:
                original_filename = Path(result.path).name
                cv2.imwrite(str(bea_save_dir / original_filename), annotated_frame)

        if project_args.save_txt:
            labels_dir = Path(base_save_dir) / "labels"
            labels_dir.mkdir(exist_ok=True)
            txt_path = labels_dir / f"{Path(image_path).stem}.txt"
            result.save_txt(txt_file=txt_path, save_conf=project_args.save_conf)

        if project_args.save_crop:
            crops_dir = Path(base_save_dir) / "crops"
            crops_dir.mkdir(exist_ok=True)
            result.save_crop(save_dir=crops_dir, file_name=f"{Path(image_path).stem}_")

    except Exception as e:
        print(f"[PID {os.getpid()}] 处理图片 {image_path} 时出错: {e}")
    else:
        print(f"[PID {os.getpid()}] 处理完成: {Path(image_path).name}")

class MultiThreadInferer:
    """
    三线程YOLO推理器
    - Producer线程：读取帧 + 批量模型推理
    - Beautifier线程：结果绘制美化
    - Consumer线程：显示和保存
    """

    def __init__(
            self,
            model,
            source: str,
            display_size: Tuple[int, int] = (1280, 720),
            window_name: str = "YOLO Multi-Thread Inference",
            queue_maxsize: int = 10,
            logger: Optional[logging.Logger] = None,
            tensorrt_config: Optional[Dict] = None  # 新增参数
    ):
        """
        初始化多线程推理器

        Args:
            model: YOLO模型实例
            source: 视频源路径或摄像头索引
            display_size: 显示窗口大小 (width, height)
            window_name: 显示窗口名称
            queue_maxsize: 队列最大大小
            logger: 日志记录器
        """
        self.model = model
        self.source = source
        self.display_width, self.display_height = display_size
        self.window_name = window_name
        self.logger = logger or logging.getLogger(__name__)

        # 队列和控制信号
        self.inference_queue = queue.Queue(maxsize=queue_maxsize)
        self.annotated_frame_queue = queue.Queue(maxsize=queue_maxsize)
        self.stop_event = threading.Event()

        # 统计信息
        self.stats = {
            'producer': {
                'frame_count': 0,
                'start_time': 0,
                'total_read_time': 0,
                'total_inference_time': 0,
                'batch_count': 0,
                'total_batch_time': 0
            },
            'beautifier': {'frame_count': 0, 'start_time': 0, 'total_plot_time': 0},
            'consumer': {'frame_count': 0, 'start_time': 0, 'total_display_time': 0}
        }

        # 配置参数
        self.conf_threshold = 0.25
        self.iou_threshold = 0.7
        self.batch_size = 8  # 改为更合理的批次大小
        self.device = "cuda"

        # 保存相关
        self.save_video = False
        self.save_frames = False
        # 新增保存 txt 和 crop 的标志
        self.save_txt = False
        self.save_crop = False
        self.save_conf = False  # 保存置信度

        self.video_writer = None
        self.frames_save_dir = None
        # 新增TensorRT配置
        self.tensorrt_config = tensorrt_config or {}
        self.is_tensorrt = self._check_tensorrt_model(model)

        # 新增 labels 和 crops 的保存目录
        self.labels_save_dir = None
        self.crops_save_dir = None

        if self.is_tensorrt and logger:
            logger.info("检测到TensorRT优化模型，将使用优化推理")

    def _check_tensorrt_model(self, model) -> bool:
        """检查模型是否为TensorRT引擎（修复版）"""
        try:
            # 方法1: 检查模型文件路径
            if hasattr(model, 'ckpt_path') and model.ckpt_path:
                model_path = str(model.ckpt_path)
                if model_path.endswith('.engine'):
                    return True

            # 方法2: 检查模型的predictor属性
            if hasattr(model, 'predictor') and model.predictor:
                predictor_type = str(type(model.predictor))
                if 'tensorrt' in predictor_type.lower() or 'trt' in predictor_type.lower():
                    return True

            # 方法3: 检查模型的设备信息
            if hasattr(model, 'model') and hasattr(model.model, 'device'):
                device_info = str(model.model.device)
                if 'tensorrt' in device_info.lower():
                    return True

            # 方法4: 尝试获取引擎信息（如果是TensorRT模型）
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'engine'):
                    return True
            except:
                pass

            return False

        except Exception as e:
            if self.logger:
                self.logger.debug(f"TensorRT检测出错，假设为PyTorch模型: {e}")
            return False

    def configure_inference(self, conf: float = 0.25, iou: float = 0.7,
                           batch_size: int = 8, device: str = "cuda"):
        """配置推理参数"""
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.device = device

        if self.is_tensorrt:
            # TensorRT模型批次大小优化
            # 检查TensorRT引擎的最大批次大小
            try:
                if self.logger:
                    self.logger.info(f"TensorRT模型使用批次大小: {batch_size}")
                self.batch_size = batch_size
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"无法确定TensorRT批次限制，使用默认值1: {e}")
                self.batch_size = 1
        else:
            # PyTorch模型可以使用更大的批次
            self.batch_size =batch_size
            if self.logger:
                self.logger.info(f"PyTorch模型批次大小: {self.batch_size}")

    def configure_save(self, save_video: bool = False, save_frames: bool = False,
                       save_txt: bool = False, save_conf: bool = False, save_crop: bool = False,
                       save_dir: Optional[Path] = None, fps: float = 30.0):
        """配置所有保存参数"""
        self.save_video = save_video
        self.save_frames = save_frames
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop

        if save_dir and any([save_video, save_frames, save_txt, save_crop]):
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            if save_video:
                video_path = save_dir / "output.mp4"
                self.video_writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (self.display_width, self.display_height)
                )
                self.logger.info(f"视频将保存至: {video_path}")

            if save_frames:
                self.frames_save_dir = save_dir / "frames"
                self.frames_save_dir.mkdir(exist_ok=True)
                self.logger.info(f"帧图像将保存至: {self.frames_save_dir}")

            # 新增: 创建 labels 和 crops 目录
            if save_txt:
                self.labels_save_dir = save_dir / "labels"
                self.labels_save_dir.mkdir(exist_ok=True)
                self.logger.info(f"TXT标签将保存至: {self.labels_save_dir}")

            if save_crop:
                self.crops_save_dir = save_dir / "crops"
                self.crops_save_dir.mkdir(exist_ok=True)
                self.logger.info(f"裁剪图将保存至: {self.crops_save_dir}")

    def producer_thread(self, cap, process_frame_func, project_args, beautify_config):
        """生产者线程：读取帧 + 批量推理/针对TensorRT优化的批量推理"""
        self.logger.info(f"生产者线程启动 - 批量大小: {self.batch_size}")
        if self.is_tensorrt:
            self.logger.info("使用TensorRT优化推理")
        stats = self.stats['producer']
        stats['start_time'] = time.perf_counter()

        # 用于收集帧的缓冲区
        frame_buffer = []
        frame_indices = []  # 记录帧的索引，用于保持帧的顺序

        # --- 使用一个外层标志来控制主循环 ---
        is_running = True
        while is_running and not self.stop_event.is_set():
            try:
                # 读取帧直到达到batch_size或视频结束
                while len(frame_buffer) < self.batch_size:
                    read_start = time.perf_counter()
                    ret, frame = cap.read()
                    read_end = time.perf_counter()
                    stats['total_read_time'] += (read_end - read_start) * 1000

                    if not ret:
                        self.logger.info("生产者: 视频读取完毕")
                        # --- 关键：设置标志并跳出内层循环 ---
                        is_running = False
                        break

                    frame_buffer.append(frame.copy())
                    frame_indices.append(stats['frame_count'])
                    stats['frame_count'] += 1

                # 如果没有帧可处理，退出
                if not frame_buffer:
                    self.logger.info("生产者: 无更多帧可处理，发出停止信号")
                    self.stop_event.set()
                    break

                # 批量推理（TensorRT优化）
                if frame_buffer:
                    inference_start = time.perf_counter()

                    try:
                        # 对于TensorRT模型，使用更优化的推理参数
                        inference_kwargs = {
                            'conf': self.conf_threshold,
                            'iou': self.iou_threshold,
                            'verbose': False,
                            'device': self.device
                        }

                        # TensorRT特定优化
                        if self.is_tensorrt:
                            # TensorRT模型推理参数调整
                            inference_kwargs.update({
                                'augment': False,  # 关闭数据增强
                                'agnostic_nms': False,  # 关闭类别无关NMS
                            })
                            inference_kwargs.pop('half', None)

                            if self.logger:
                                self.logger.debug("使用TensorRT优化推理参数")

                        batch_results = self.model.predict(frame_buffer, **inference_kwargs)

                        inference_end = time.perf_counter()
                        batch_inference_time = (inference_end - inference_start) * 1000
                        stats['total_inference_time'] += batch_inference_time
                        stats['batch_count'] += 1
                        stats['total_batch_time'] += batch_inference_time

                        # 将结果放入队列
                        for i, (frame, result) in enumerate(zip(frame_buffer, batch_results)):
                            try:
                                # 传递 (frame, result, frame_index)
                                self.inference_queue.put((frame, result, frame_indices[i]), timeout=0.1)
                            except queue.Full:
                                self.logger.warning(f"推理队列已满，跳过帧 {frame_indices[i]}")
                                continue

                        # 输出批量推理统计（包含TensorRT信息）
                        avg_time_per_frame = batch_inference_time / len(frame_buffer)
                        engine_info = "TensorRT" if self.is_tensorrt else "PyTorch"
                        self.logger.debug(f"批量推理完成({engine_info}): {len(frame_buffer)} 帧, "
                                          f"总耗时: {batch_inference_time:.2f}ms, "
                                          f"平均每帧: {avg_time_per_frame:.2f}ms")

                    except Exception as e:
                        # TensorRT特定错误处理
                        if self.is_tensorrt and ("cuda" in str(e).lower() or "tensorrt" in str(e).lower()):
                            if self.logger:
                                self.logger.error(f"TensorRT推理失败: {e}")
                                self.logger.info("尝试单帧推理...")
                            # 对于TensorRT，尝试单帧推理
                            single_results = []
                            for frame in frame_buffer:
                                try:
                                    result = self.model.predict([frame], **inference_kwargs)
                                    single_results.extend(result)
                                except Exception as single_e:
                                    if self.logger:
                                        self.logger.error(f"单帧推理也失败: {single_e}")
                                    raise single_e
                            return single_results
                        else:
                            # 普通错误，直接抛出
                            raise e

                # 清空缓冲区
                frame_buffer.clear()
                frame_indices.clear()

                # 定期输出统计信息
                if stats['frame_count'] % 60 == 0:
                    elapsed = time.perf_counter() - stats['start_time']
                    fps = stats['frame_count'] / elapsed if elapsed > 0 else 0
                    avg_batch_time = stats['total_batch_time'] / stats['batch_count'] if stats[
                                                                                             'batch_count'] > 0 else 0
                    engine_info = "TensorRT" if self.is_tensorrt else "PyTorch"
                    self.logger.info(f"生产者({engine_info}): 已处理 {stats['frame_count']} 帧, "
                                     f"FPS: {fps:.1f}, 平均批次耗时: {avg_batch_time:.2f}ms")

            except Exception as e:
                self.logger.error(f"生产者线程错误: {e}")
                self.stop_event.set()
                break

        # 处理剩余的帧（如果有的话）
        if frame_buffer and not self.stop_event.is_set():
            self.logger.info(f"处理剩余的 {len(frame_buffer)} 帧")
            try:
                batch_results = self.model.predict(
                    frame_buffer,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device=self.device
                )

                for frame, result in zip(frame_buffer, batch_results):
                    try:
                        self.inference_queue.put((frame, result), timeout=0.1)
                    except queue.Full:
                        break

            except Exception as e:
                self.logger.error(f"处理剩余帧时出错: {e}")

        # --- 确保在所有情况下都设置停止事件 ---
        self.logger.info("生产者: 发出最终停止信号")
        self.stop_event.set()

        self._log_producer_stats()
        self.logger.info("生产者线程退出")

    def beautifier_thread(self, process_frame_func, project_args, beautify_config):
        """美化线程：绘制检测结果"""
        self.logger.info("绘制线程启动")
        stats = self.stats['beautifier']
        stats['start_time'] = time.perf_counter()
        while not (self.stop_event.is_set() and self.inference_queue.empty()):
            try:
                # 现在接收帧、结果和索引
                frame, result, frame_idx = self.inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"绘制线程获取队列数据错误: {e}")
                self.stop_event.set()
                break

            try:
                # 绘制美化
                plot_start = time.perf_counter()
                annotated_frame = process_frame_func(frame, result, project_args, beautify_config)
                plot_end = time.perf_counter()
                stats['total_plot_time'] += (plot_end - plot_start) * 1000

                # 新增: 保存 txt 和 crop
                if self.save_txt and self.labels_save_dir:
                    txt_path = self.labels_save_dir / f"{frame_idx}.txt"
                    result.save_txt(txt_file=txt_path, save_conf=self.save_conf)

                if self.save_crop and self.crops_save_dir:
                    result.save_crop(save_dir=self.crops_save_dir, file_name=f"{frame_idx}_")

                # 放入显示队列
                # 将帧和它的索引一起传递给消费者，以便保存时文件名一致
                self.annotated_frame_queue.put((annotated_frame, frame_idx), timeout=0.1)

                stats['frame_count'] += 1
                frame_idx += 1

                # (定期日志输出不变)
                if stats['frame_count'] % 30 == 0:
                    elapsed = time.perf_counter() - stats['start_time']
                    fps = stats['frame_count'] / elapsed if elapsed > 0 else 0
                    self.logger.info(f"绘制器: 已绘制 {stats['frame_count']} 帧, FPS: {fps:.1f}")

            except queue.Full:
                pass
            except Exception as e:
                self.logger.error(f"绘制线程处理帧错误: {e}")
                self.stop_event.set()
                break

        self._log_beautifier_stats()
        self.logger.info("绘制线程退出")

    def consumer_thread(self):
        """消费者线程：显示和保存"""
        self.logger.info("消费者线程启动")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        stats = self.stats['consumer']
        stats['start_time'] = time.perf_counter()
        last_fps_update = time.perf_counter()
        displayed_fps = 0.0

        while True:
            try:
                # 现在从队列中获取 annotated_frame 和它的索引
                annotated_frame, frame_idx = self.annotated_frame_queue.get(timeout=0.1)
            except queue.Empty:
                if self.stop_event.is_set() and self.annotated_frame_queue.empty():
                    self.logger.info("消费者: 停止信号已设置且显示队列已清空，准备退出")
                    break
                continue
            except Exception as e:
                self.logger.error(f"消费者线程获取队列数据错误: {e}")
                self.stop_event.set()
                break

            try:
                display_start = time.perf_counter()

                # 调整显示尺寸
                display_frame = cv2.resize(annotated_frame, (self.display_width, self.display_height))

                # 计算并显示FPS
                current_time = time.perf_counter()
                if current_time - last_fps_update >= 1.0:
                    elapsed = current_time - stats['start_time']
                    displayed_fps = stats['frame_count'] / elapsed if elapsed > 0 else 0
                    last_fps_update = current_time

                # 添加FPS文本
                self._add_fps_text(display_frame, displayed_fps)

                # 显示帧
                cv2.imshow(self.window_name, display_frame)

                # 保存视频
                if self.video_writer:
                    self.video_writer.write(display_frame)

                # 保存帧
                if self.save_frames and self.frames_save_dir:
                    frame_path = self.frames_save_dir / f"{frame_idx}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)  # 保存原始尺寸的美化帧

                display_end = time.perf_counter()
                stats['total_display_time'] += (display_end - display_start) * 1000
                stats['frame_count'] += 1

                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.logger.info("消费者: 检测到用户退出键，发出停止信号")
                    self.stop_event.set()
                    break

            except Exception as e:
                self.logger.error(f"消费者线程处理帧错误: {e}")
                self.stop_event.set()
                break

        self._log_consumer_stats()
        self._cleanup_consumer()
        self.logger.info("消费者线程退出")

    def _add_fps_text(self, frame, fps):
        """在帧上添加FPS文本"""
        fps_text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 255, 0)
        bg_color = (0, 0, 0)

        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
        text_x, text_y = 10, text_height + 10

        # 背景矩形
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5),
                      (text_x + text_width + 5, text_y + baseline + 5), bg_color, cv2.FILLED)
        # 文本
        cv2.putText(frame, fps_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def _cleanup_consumer(self):
        """清理消费者资源"""
        if self.video_writer:
            self.video_writer.release()
            self.logger.info("视频写入器已释放")
        cv2.destroyAllWindows()

    def run(self, process_frame_func: Callable, project_args, beautify_config):
        """
        运行多线程推理

        Args:
            process_frame_func: 帧处理函数
            project_args: 项目参数
            beautify_config: 美化配置
        """
        # 打开视频源
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        self.logger.info(f"开始多线程批量推理，视频源: {self.source}, 批次大小: {self.batch_size}")

        try:
            # 创建并启动线程
            producer = threading.Thread(
                target=self.producer_thread,
                args=(cap, process_frame_func, project_args, beautify_config)
            )
            beautifier = threading.Thread(
                target=self.beautifier_thread,
                args=(process_frame_func, project_args, beautify_config)
            )
            consumer = threading.Thread(target=self.consumer_thread)

            # 启动线程
            producer.start()
            beautifier.start()
            consumer.start()

            # 等待线程结束
            producer.join()
            beautifier.join()
            consumer.join()

        finally:
            cap.release()
            self.logger.info("多线程批量推理完成，资源已释放")

    def _log_producer_stats(self):
        """记录生产者统计信息"""
        stats = self.stats['producer']
        if stats['frame_count'] > 0:
            total_time = time.perf_counter() - stats['start_time']
            avg_batch_time = stats['total_batch_time'] / stats['batch_count'] if stats['batch_count'] > 0 else 0
            avg_frames_per_batch = stats['frame_count'] / stats['batch_count'] if stats['batch_count'] > 0 else 0

            self.logger.info("\n--- 生产者线程总结 ---")
            self.logger.info(f"总处理帧数: {stats['frame_count']}")
            self.logger.info(f"总批次数: {stats['batch_count']}")
            self.logger.info(f"平均每批帧数: {avg_frames_per_batch:.1f}")
            self.logger.info(f"总运行时间: {total_time:.2f} 秒")
            self.logger.info(f"平均FPS: {(stats['frame_count'] / total_time):.2f}")
            self.logger.info(f"平均读取耗时: {stats['total_read_time'] / stats['frame_count']:.2f}ms")
            self.logger.info(f"平均批次推理耗时: {avg_batch_time:.2f}ms")
            self.logger.info(f"平均单帧推理耗时: {stats['total_inference_time'] / stats['frame_count']:.2f}ms")

    def _log_beautifier_stats(self):
        """记录美化线程统计信息"""
        stats = self.stats['beautifier']
        if stats['frame_count'] > 0:
            total_time = time.perf_counter() - stats['start_time']
            self.logger.info("\n--- 绘制线程总结 ---")
            self.logger.info(f"总绘制帧数: {stats['frame_count']}")
            self.logger.info(f"总运行时间: {total_time:.2f} 秒")
            self.logger.info(f"平均绘制FPS: {(stats['frame_count'] / total_time):.2f}")
            self.logger.info(f"平均绘制耗时: {stats['total_plot_time'] / stats['frame_count']:.2f}ms")

    def _log_consumer_stats(self):
        """记录消费者统计信息"""
        stats = self.stats['consumer']
        if stats['frame_count'] > 0:
            total_time = time.perf_counter() - stats['start_time']
            self.logger.info("\n--- 消费者线程总结 ---")
            self.logger.info(f"总显示帧数: {stats['frame_count']}")
            self.logger.info(f"总运行时间: {total_time:.2f} 秒")
            self.logger.info(f"平均显示FPS: {(stats['frame_count'] / total_time):.2f}")
            self.logger.info(f"平均显示耗时: {stats['total_display_time'] / stats['frame_count']:.2f}ms")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        result = {}
        for thread_name, stats in self.stats.items():
            if stats['frame_count'] > 0:
                total_time = time.perf_counter() - stats['start_time']
                thread_result = {
                    'frame_count': stats['frame_count'],
                    'total_time': total_time,
                    'avg_fps': stats['frame_count'] / total_time if total_time > 0 else 0
                }

                # 为生产者线程添加额外的批量统计信息
                if thread_name == 'producer' and 'batch_count' in stats:
                    thread_result.update({
                        'batch_count': stats['batch_count'],
                        'avg_frames_per_batch': stats['frame_count'] / stats['batch_count'] if stats[
                                                                                                   'batch_count'] > 0 else 0,
                        'avg_batch_time_ms': stats['total_batch_time'] / stats['batch_count'] if stats[
                                                                                                     'batch_count'] > 0 else 0
                    })

                result[thread_name] = thread_result
            else:
                result[thread_name] = {'frame_count': 0, 'total_time': 0, 'avg_fps': 0}
        return result


def load_tensorrt_model(model_path, tensorrt_config=None, logger=None):
    """
    加载并优化YOLO模型为TensorRT格式（修复版）

    Args:
        model_path: 原始模型路径
        tensorrt_config: TensorRT配置字典
        logger: 日志记录器

    Returns:
        优化后的模型
    """
    if logger:
        logger.info("开始TensorRT模型加载/优化...")

    # 标准化路径
    model_path = Path(model_path)

    # 如果已经是TensorRT引擎文件，直接加载
    if model_path.suffix == '.engine':
        if logger:
            logger.info(f"检测到TensorRT引擎文件，直接加载: {model_path}")
        try:
            return YOLO(str(model_path),task = 'detect')
        except Exception as e:
            if logger:
                logger.error(f"加载TensorRT引擎失败: {e}")
            raise

    # 加载原始模型
    model = YOLO(str(model_path),task='detect')

    if tensorrt_config and tensorrt_config.get('use_tensorrt', False):
        try:
            # 生成TensorRT引擎文件路径
            # 从配置中获取批次大小，默认为1
            batch_size = tensorrt_config.get('batch', 1)

            # 根据批次大小生成不同的引擎文件名
            engine_path = model_path.with_name(f"{model_path.stem}_batch{batch_size}.engine")

            # 检查是否已存在TensorRT引擎文件
            if engine_path.exists():
                if logger:
                    logger.info(f"发现已存在的TensorRT引擎: {engine_path}")
                try:
                    # 尝试加载现有引擎
                    optimized_model = YOLO(str(engine_path),task = 'detect')
                    if logger:
                        logger.info("成功加载现有TensorRT引擎")
                    return optimized_model
                except Exception as e:
                    if logger:
                        logger.warning(f"加载现有引擎失败，将重新导出: {e}")
                    # 删除损坏的引擎文件
                    engine_path.unlink(missing_ok=True)

            if logger:
                logger.info(f"开始导出TensorRT引擎到: {engine_path}") # 日志中明确路径

            # 准备导出参数
            export_kwargs = {
                'format': 'engine',
                'device': tensorrt_config.get('device', 0),
                'workspace': tensorrt_config.get('workspace', 4),
                'verbose': True,
                'batch': batch_size,
            }

            # 根据精度设置参数
            precision = tensorrt_config.get('precision', 'fp16')
            if precision == 'fp16':
                export_kwargs['half'] = True
            elif precision == 'int8':
                export_kwargs['int8'] = True
                # INT8量化需要校准数据，这里使用默认
                if logger:
                    logger.warning("使用INT8精度，建议提供校准数据以获得最佳精度")

            # 导出TensorRT引擎
            default_exported_path_str = model.export(**export_kwargs)
            default_exported_path = Path(default_exported_path_str)

            if not default_exported_path.exists():
                raise FileNotFoundError(f"YOLO export 未能成功生成引擎文件 at {default_exported_path_str}")

            # --- 关键操作：将默认文件重命名为我们想要的文件 ---
            if logger:
                logger.info(f"引擎已生成于 {default_exported_path}, 准备将其重命名为 {engine_path}")

            # 使用 shutil.move 来重命名，更健壮
            shutil.move(str(default_exported_path), str(engine_path))

            if engine_path.exists():
                if logger:
                    logger.info(f"重命名成功。现在加载引擎: {engine_path}")
                # 加载我们刚刚重命名好的文件
                return YOLO(str(engine_path), task='detect')
            else:
                raise FileNotFoundError(f"重命名引擎文件失败，目标路径 {engine_path} 不存在。")

        except Exception as e:
            if logger:
                logger.error(f"TensorRT优化失败: {e}")
                logger.info("回退到原始PyTorch模型")
            return model

    return model

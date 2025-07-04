#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :extract_frame.py
# @Time      :2025/6/10 10:03
# @Author    :雨霓同学
# @Function  : 提取视频帧
import cv2  # opencv库用于提取视频
import os
import argparse

def extract_frame(video_path, output_dir, frame_interval=1):
    """
    从视频中按照指定的帧提取帧并保存为图像
    :param video_path: 输入的视频文件路径
    :param output_dir: 输入图像目录
    :param frame_interval: 帧提取间隔
    :return: None
    """
    # 创建一个输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误无法打开视频文件")
        return
    # 获取视频的属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    print(f"视频总帧数：{total_frames}, 帧率：{fps}")
    # 初始化计数器
    frame_count = 0
    save_count = 0

    # 逐帧提取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # 构建保存图像的文件名
            output_path = os.path.join(output_dir, f"frame_{save_count:06d}.png")
            # 保存图像
            cv2.imwrite(output_path, frame)
            save_count += 1
            print(f"已保存帧：{output_path}")
        frame_count += 1
    # 释放资源
    cap.release()
    print(f"已保存{save_count}帧")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中提取帧并保存为图像")
    parser.add_argument("--video", type=str, required=True, help="输入的视频文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出图像目录")
    parser.add_argument("--interval", type=int, default=10, help="帧提取间隔")
    args = parser.parse_args()
    extract_frame(args.video, args.output, args.interval)

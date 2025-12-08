#!/usr/bin/env python3
"""
GPU监控脚本 - 实时显示GPU使用情况和内存占用
"""

import time
import torch
import subprocess
import sys


def display_gpu_info():
    """显示GPU信息 - 简化版本"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n=== GPU信息 ===")
        print(f"GPU数量: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\n警告: 未检测到GPU，将使用CPU训练")


def get_gpu_memory_info():
    """获取GPU内存信息"""
    gpu_info = []
    gpu_count = torch.cuda.device_count()

    for i in range(gpu_count):
        gpu_info.append({
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'memory_total': torch.cuda.get_device_properties(i).total_memory,
            'memory_used': 0,
            'memory_free': 0,
            'gpu_util': 0,
            'mem_util': 0
        })

    return gpu_info


def get_gpu_memory_usage_string(max_gpus=3):
    """获取GPU负载信息字符串，用于进度条显示"""
    if not torch.cuda.is_available():
        return "CPU"

    import pynvml
    pynvml.nvmlInit()

    gpu_count = torch.cuda.device_count()
    gpu_loads = []

    for i in range(min(gpu_count, max_gpus)):
        # 获取GPU使用率
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu  # GPU使用率

        # 获取内存使用率
        memory_info = torch.cuda.mem_get_info(i)
        memory_used = memory_info[1] - memory_info[0]
        memory_total = memory_info[1]
        memory_util = (memory_used / memory_total) * 100

        gpu_loads.append(f"GPU{i}:{int(gpu_util):02d}%/{int(memory_util):02d}%M")

    return " | ".join(gpu_loads)


def print_gpu_status():
    """打印GPU状态"""
    if not torch.cuda.is_available():
        print("\n警告: 未检测到GPU，将使用CPU训练")
        return

    gpu_count = torch.cuda.device_count()
    print(f"\n=== GPU信息 ===")
    print(f"GPU数量: {gpu_count}")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")


def monitor_gpu(duration=60, interval=5):
    """
    持续监控GPU状态

    Args:
        duration: 监控总时长（秒）
        interval: 刷新间隔（秒）
    """
    print(f"\n开始GPU监控 ({duration}秒，每{interval}秒刷新一次)...")
    print("按 Ctrl+C 退出监控\n")

    start_time = time.time()

    while time.time() - start_time < duration:
        print_gpu_status()

        if time.time() - start_time < duration - interval:
            print(f"\n下次刷新在 {interval} 秒后...\n")
            time.sleep(interval)
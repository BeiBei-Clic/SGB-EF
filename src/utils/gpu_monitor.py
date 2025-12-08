#!/usr/bin/env python3
"""
GPU监控脚本 - 实时显示GPU使用情况和内存占用
"""

import time
import torch
import subprocess
import sys


def get_gpu_memory_info():
    """获取GPU内存信息"""
    gpu_info = []
    gpu_count = torch.cuda.device_count()

    for i in range(gpu_count):
        # 使用nvidia-smi获取详细信息
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if parts[0] == str(i):
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_free': int(parts[4]),
                        'gpu_util': int(parts[5]),
                        'mem_util': int(parts[6])
                    })
                    break
        except:
            # 如果nvidia-smi失败，使用torch获取基本信息
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
    """获取GPU内存使用率字符串，用于进度条显示"""
    if not torch.cuda.is_available():
        return "CPU"

    gpu_info = get_gpu_memory_info()
    gpu_memories = []
    for gpu in gpu_info[:max_gpus]:  # 最多显示指定数量的GPU
        memory_used = gpu['memory_used'] / 1024  # 转换为GB
        memory_total = gpu['memory_total'] / 1024  # 转换为GB
        gpu_memories.append(f"GPU{gpu['index']}:{memory_used:.1f}GB/{memory_total:.1f}GB")
    return " | ".join(gpu_memories)


def print_gpu_status():
    """打印GPU状态"""
    gpu_info = get_gpu_memory_info()

    print("\n" + "=" * 80)
    print(f"{'GPU':<5} {'名称':<30} {'内存使用':<20} {'利用率':<15}")
    print("=" * 80)

    for gpu in gpu_info:
        memory_str = f"{gpu['memory_used']/1024:.1f}/{gpu['memory_total']/1024:.1f} GB"
        util_str = f"GPU: {gpu['gpu_util']}% | MEM: {gpu['mem_util']}%"

        # 内存使用进度条
        bar_length = 20
        mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        filled_length = int(bar_length * mem_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"{gpu['index']:<5} {gpu['name']:<30} {memory_str} [{bar}] {util_str}")

    print("=" * 80)

    # 计算总内存
    total_memory = sum(gpu['memory_total'] for gpu in gpu_info)
    total_used = sum(gpu['memory_used'] for gpu in gpu_info)
    print(f"\n总计: {total_used/1024:.1f}/{total_memory/1024:.1f} GB "
          f"({(total_used/total_memory)*100:.1f}% 使用率)")


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

    try:
        while time.time() - start_time < duration:
            print_gpu_status()

            if time.time() - start_time < duration - interval:
                print(f"\n下次刷新在 {interval} 秒后...\n")
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == "__main__":
    # 默认显示一次当前状态
    print_gpu_status()

    # 如果指定了持续时间，则进行监控
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            monitor_gpu(duration, interval)
        except ValueError:
            print("用法: python gpu_monitor.py [duration_seconds] [interval_seconds]")
            sys.exit(1)
    else:
        print("\n提示: 运行 'python gpu_monitor.py 60 5' 进行60秒持续监控（每5秒刷新）")

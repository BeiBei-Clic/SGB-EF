"""
符号回归数据生成器，用于EditFlow预训练
"""

import numpy as np
import random
import os
import warnings
import time
import json
import datetime
import multiprocessing
from typing import List, Dict, Tuple
from tqdm import tqdm
from src.utils.timeout_utils import TimeoutError, with_timeout
from src.utils.logger import Logger
from src.symbolic.symbolic_utils import generate_random_expr, evaluate_expression_safe, expr_to_tree
from src.symbolic.corruption import corrupt_expression
from src.symbolic.sample_generator import generate_single_sample, set_logger

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 创建全局 Logger 实例用于样本生成日志
_sample_logger = Logger(enabled=True)

# 常量定义
MAX_RETRIES = 5  # 表达式生成和计算的最大重试次数

def generate_batch_worker(args: Tuple) -> Tuple[int, List[Dict], Dict[int, int]]:
    """单个进程处理一个批次的数据生成（用于多进程）

    Args:
        args: 包含所有必要参数的元组
            (batch_idx, current_batch_size, max_dim, n_points, max_depth,
             max_expr_length, batch_filename, verbose, process_id)

    Returns:
        Tuple[int, List[Dict], Dict[int, int]]:
            (批次索引, 生成的样本列表, 维度统计)
    """
    (batch_idx, current_batch_size, max_dim, n_points, max_depth,
     max_expr_length, batch_filename, verbose, process_id) = args

    # 设置进程特定的随机种子，确保真正的随机性
    # 使用高精度时间戳、进程ID、批次ID和额外随机因子
    current_time_ms = int(time.time() * 1000000)  # 微秒级时间戳
    # 使用多个因子组合，加上额外的随机性
    seed_base = current_time_ms + (process_id << 16) + (batch_idx << 8) + os.getpid()
    # 添加哈希操作增加随机性
    seed_val = hash(str(seed_base)) & 0x7fffffff  # 确保为正数

    random.seed(seed_val)
    np.random.seed(seed_val)

    # 设置日志进程标识
    process_prefix = f"[B{batch_idx+1}]"  # 使用批次号作为唯一标识

    # 批次开始信息通过进度条显示，避免额外print输出

    batch_samples = []
    dimension_count = {}
    sample_count = 0

    SAMPLE_TIMEOUT = 10.0  # 单个样本生成超时时间（秒）

    # 移除进度条，避免多进程显示混乱

    while sample_count < current_batch_size:
        # 生成更随机的样本ID，避免重复
        unique_factor = random.randint(0, 999999)
        sample_id = f"{process_prefix}_sample{sample_count}_{os.getpid()}_{unique_factor}"

        try:
            # 记录调用前的详细信息
            import traceback
            _sample_logger.log("DEBUG_CALL", f"准备调用 generate_single_sample", f"sample_id={sample_id}", level=1)
            _sample_logger.log("DEBUG_PARAMS", f"max_dim={max_dim}, n_points={n_points}, max_depth={max_depth}", f"max_expr_length={max_expr_length}, batch_idx={batch_idx}, current_batch_size={current_batch_size}, sample_count={sample_count}", level=1)
            _sample_logger.log("DEBUG_ARG_COUNT", f"传递参数数量=8 (不含func和timeout)", level=1)
            _sample_logger.log("DEBUG_STACK", f"调用栈:", "\n".join(traceback.format_stack()[-5:]), level=1)

            # 使用带超时保护的样本生成函数
            generated_samples = with_timeout(
                generate_single_sample,
                SAMPLE_TIMEOUT,
                sample_id,
                max_dim,
                n_points,
                max_depth,
                max_expr_length,
                batch_idx,
                current_batch_size,
                sample_count
            )

            _sample_logger.log("DEBUG_SUCCESS", f"调用成功, 返回 {len(generated_samples) if generated_samples else 0} 个样本", level=1)

            # 如果成功生成样本，添加到批次中
            if generated_samples:
                # 获取维度信息用于统计
                dim = generated_samples[0]["input_dimension"]

                # 添加生成的样本到批次
                for sample in generated_samples:
                    if sample_count >= current_batch_size:
                        break
                    batch_samples.append(sample)

                    sample_count += 1
                    dimension_count[dim] = dimension_count.get(dim, 0) + 1
            else:
                # 样本生成失败，记录并继续（不增加sample_count）
                _sample_logger.sample_failed(sample_id, "No samples generated")
                continue

        except TimeoutError:
            # 超时处理（不增加sample_count）
            _sample_logger.sample_timeout(sample_id, SAMPLE_TIMEOUT)
            continue

        except Exception as e:
            # 其他异常处理（不增加sample_count）
            import traceback
            _sample_logger.log("DEBUG_EXCEPTION", f"捕获异常: {type(e).__name__}", f"错误信息: {str(e)}", level=1)
            _sample_logger.log("DEBUG_TRACEBACK", f"完整堆栈跟踪:", f"\n{''.join(traceback.format_exc())}", level=1)
            _sample_logger.sample_error(sample_id, type(e).__name__, str(e))
            raise  # 直接抛出异常,不再继续

    # 批次完成，无需关闭进度条

    # 立即保存当前批次到文件
    if batch_samples:
        os.makedirs(os.path.dirname(batch_filename), exist_ok=True)
        with open(batch_filename, 'w', encoding='utf-8') as f:
            for sample in batch_samples:
                sample_line = json.dumps(sample, ensure_ascii=False)
                f.write(sample_line + '\n')

        if verbose:
            print(f"\n{process_prefix} 已保存 {len(batch_samples)} 个样本到 {batch_filename}")
            print(f"{process_prefix} 批次维度分布:")
            for dim, count in sorted(dimension_count.items()):
                print(f"  {dim}维: {count} 个样本")

    return batch_idx, len(batch_samples), dimension_count

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4, seed: int = None) -> Dict:
    """生成单个样本

    Args:
        input_dimension: 输入维度
        n_points: 数据点数量
        max_depth: 表达式最大深度
        seed: 随机种子，None表示使用系统随机源
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 生成更随机的样本ID，包含时间戳和随机因子
    timestamp_ms = int(time.time() * 1000000)
    random_factor = random.randint(0, 999999)
    sample_id = f"sample_{input_dimension}dim_{timestamp_ms}_{random_factor}"

    # 统一生成数据点，确保每个数据点是[x0, x1, x2, ...]的形式
    x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, input_dimension))
    x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式

    # 转换为numpy数组用于表达式计算
    x_array = np.array(x_values)

    # 生成目标表达式
    target_expr = generate_random_expr(input_dimension, max_depth)

    # 计算目标值
    success, y_values = evaluate_expression_safe(target_expr, x_array)
    if not success:
        # 如果计算失败，抛出异常让调用者处理
        raise ValueError(f"表达式计算失败: {target_expr}")

    # 生成当前表达式
    curr_expr = corrupt_expression(target_expr)

    return {
        "input_dimension": input_dimension,
        "x": x_values,  # 现在是[[x0, x1, x2], [x3, x4, x5], ...]的形式
        "y": y_values.tolist(),
        "tree_gt": expr_to_tree(target_expr),
        "exp_gt": str(target_expr),
        "tree_cur1": expr_to_tree(curr_expr),
        "exp_cur1": str(curr_expr)
    }

def load_dimension_index(filename: str, verbose: bool = True) -> Dict[int, List[int]]:
    """加载维度索引文件，如果不存在则扫描数据文件并保存索引"""
    index_filename = filename.replace('.txt', '_dimension_index.json')

    if os.path.exists(index_filename):
        if verbose:
            print(f"发现维度索引文件 {index_filename}，正在加载...")
        with open(index_filename, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        # 转换回原来的格式
        dimension_samples = {}
        for dim_str, positions in index_data.items():
            dimension_samples[int(dim_str)] = positions

        if verbose:
            print(f"维度索引加载完成，共发现 {len(dimension_samples)} 个维度")
        return dimension_samples

    # 维度索引不存在，需要扫描文件
    if verbose:
        print(f"维度索引不存在，正在扫描文件进行维度统计...")
    dimension_samples = {}  # 存储每个维度的样本位置索引

    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                sample = json.loads(line)
                dim = sample['input_dimension']
                if dim not in dimension_samples:
                    dimension_samples[dim] = []
                dimension_samples[dim].append(pos)

    if verbose:
        print(f"维度统计完成，共发现 {len(dimension_samples)} 个维度")

    # 保存维度索引到缓存文件
    index_dir = os.path.dirname(index_filename)
    if index_dir:  # 只有当目录不为空时才创建
        os.makedirs(index_dir, exist_ok=True)
    index_data = {}
    for dim, positions in dimension_samples.items():
        index_data[str(dim)] = [int(pos) for pos in positions]
    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)

    return dimension_samples

def generate_flow_samples(
    num_samples: int,
    max_dim: int = 5,
    n_points: int = 100,
    max_depth: int = 4,
    max_expr_length: int = 15,
    batch_size: int = 50000,
    verbose: bool = True,
    num_processes: int = None,
):
    """生成用于EditFlow连续流训练的数据文件，支持断点续传和多进程并行处理

    Args:
        num_samples: 总样本数
        max_dim: 最大维度
        n_points: 每个样本的数据点数
        max_depth: 表达式最大深度
        max_expr_length: 表达式最大token数量（前序遍历，默认15）
        batch_size: 批次大小
        verbose: 是否显示详细输出
        num_processes: 进程数，None表示使用所有可用CPU核心
    """

    # 设置样本生成器的 Logger 实例
    set_logger(_sample_logger)

    # 使用高精度时间戳和系统信息生成主随机种子
    main_time_ms = int(time.time() * 1000000)  # 微秒级时间戳
    # 组合多个因子确保唯一性
    main_seed_base = main_time_ms + os.getpid() + (num_samples & 0xffff)
    seed_val = hash(str(main_seed_base)) & 0x7fffffff
    random.seed(seed_val)
    np.random.seed(seed_val)

    # 检查是否存在缓存文件
    filename = f"data/flow_samples_{num_samples}_{max_dim}dim_{n_points}pts_{max_depth}depth_{max_expr_length}len.txt"

    # 检查批次文件状态
    num_batches = (num_samples + batch_size - 1) // batch_size
    # 批次文件保存在 data/temp 目录中
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    batch_filenames = [f"{temp_dir}/{os.path.basename(filename).replace('.txt', f'_batch_{i + 1}.txt')}" for i in range(num_batches)]

    # 1. 主文件存在且无批次文件 → 数据完整
    if os.path.exists(filename) and not any(os.path.exists(f) for f in batch_filenames):
        if verbose:
            print(f"发现完整数据文件 {filename}")
        return

    # 1.5. 主文件存在且有批次文件 → 直接进入合并阶段（跳过生成）
    if os.path.exists(filename) and any(os.path.exists(f) for f in batch_filenames):
        if verbose:
            print(f"发现主文件和批次文件，直接进入合并阶段...")
            print(f"将追加剩余批次文件到主文件...")
            merge_batches_to_main_file(filename, batch_filenames, num_batches, total_dimension_count=None)
        return

    # 设置多进程参数
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
        if verbose:
            print(f"使用所有 {num_processes} 个CPU核心进行并行处理")
    else:
        if verbose:
            print(f"使用 {num_processes} 个进程进行并行处理")

    # 开始数据生成
    if verbose:
        print(f"使用 {num_processes} 个进程并行生成 {num_samples} 个样本，共 {num_batches} 批...")

    # 2. 分批生成数据样本，支持断点续传
    total_dimension_count = {}

    # 收集需要处理的批次
    batch_tasks = []
    for batch_idx in range(num_batches):
        # 获取当前批次的文件名
        batch_filename = batch_filenames[batch_idx]
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # 如果这个批次文件已经存在，直接跳过
        if os.path.exists(batch_filename):
            if verbose:
                print(f"跳过已存在的批次 {batch_idx + 1}")
            continue

        # 添加到任务列表，为每个任务分配一个进程ID
        process_id = len(batch_tasks) % num_processes
        batch_tasks.append((
            batch_idx, current_batch_size, max_dim, n_points, max_depth,
            max_expr_length, batch_filename, verbose, process_id
        ))

    if not batch_tasks:
        if verbose:
            print("所有批次文件已存在，跳过生成阶段")
    else:
        if verbose:
            print(f"开始并行处理 {len(batch_tasks)} 个批次...")

        # 统一使用进程池处理（num_processes=1 时退化为单进程）
        if verbose:
            print(f"开始处理 {len(batch_tasks)} 个批次任务...")

        try:
            # 使用map方法确保所有任务同步完成，避免BrokenPipeError
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(generate_batch_worker, batch_tasks)

            # 处理所有结果（在pool已完全关闭后）
            for batch_idx, sample_count, dimension_count in results:
                if verbose:
                    print(f"\r批次 {batch_idx + 1} 完成 (生成{sample_count}个样本)", end='', flush=True)

                # 累积维度统计
                for dim, count in dimension_count.items():
                    total_dimension_count[dim] = total_dimension_count.get(dim, 0) + count

            if verbose:
                print(f"\n所有 {len(batch_tasks)} 个批次并行处理完成")

        except (BrokenPipeError, KeyboardInterrupt) as e:
            if verbose:
                print(f"\n检测到进程通信中断: {type(e).__name__}")
            raise

    if verbose and total_dimension_count:
        print(f"\n已完成批次的维度分布:")
        for dim, count in sorted(total_dimension_count.items()):
            print(f"{dim}维: {count} 个样本")

    # === 合并批次文件 ===
    if verbose:
        print(f"\n按批次顺序合并所有批次文件到主文件...")
    merge_batches_to_main_file(filename, batch_filenames, num_batches, total_dimension_count)
    if verbose:
        print(f"所有数据已保存到: {filename}")


def merge_batches_to_main_file(filename: str, batch_filenames: List[str], num_batches: int, total_dimension_count: Dict = None):
    """合并批次文件到主文件（用于中断恢复）

    Args:
        filename: 主文件名
        batch_filenames: 批次文件列表
        num_batches: 总批次数
        total_dimension_count: 维度计数（生成模式传入，合并模式为None）
    """
    if total_dimension_count is not None:
        print(f"\n=== 生成模式合并阶段 ===")
    else:
        print(f"\n=== 批次文件合并模式 ===")

    # 加载已有的维度索引（断点续传时）
    index_filename = filename.replace('.txt', '_dimension_index.json')
    dimension_samples = {}

    if os.path.exists(index_filename) and os.path.exists(filename):
        # 如果索引文件和主文件都存在，加载已有索引
        print(f"加载已有维度索引: {index_filename}")
        with open(index_filename, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        # 转换回原来的格式
        for dim_str, positions in index_data.items():
            dimension_samples[int(dim_str)] = positions
        print(f"已加载 {len(dimension_samples)} 个维度的索引信息")

    with open(filename, 'a', encoding='utf-8') as main_file:
        for batch_idx in range(num_batches):
            batch_filename = batch_filenames[batch_idx]
            if os.path.exists(batch_filename):
                # 读取批次样本并记录位置
                print(f"从 {batch_filename} 加载数据...")
                batch_samples = []
                with open(batch_filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            sample = json.loads(line)
                            batch_samples.append(sample)
                print(f"已加载 {len(batch_samples)} 个样本")
                for sample in batch_samples:
                    # 记录当前位置（即将写入的位置）
                    pos = main_file.tell()
                    dim = sample['input_dimension']
                    if dim not in dimension_samples:
                        dimension_samples[dim] = []
                    dimension_samples[dim].append(pos)

                    # 写入主文件
                    sample_line = json.dumps(sample, ensure_ascii=False)
                    main_file.write(sample_line + '\n')

                # 删除批次文件
                os.remove(batch_filename)
                print(f"已合并并删除批次文件: {batch_filename}")

    # 保存完整的维度索引文件（包含已有数据和新数据）
    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    index_data = {}
    for dim, positions in dimension_samples.items():
        index_data[str(dim)] = [int(pos) for pos in positions]
    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    print(f"完整维度索引已保存到 {index_filename}")

    # 根据模式打印维度分布
    if total_dimension_count is not None:
        print(f"\n总体样本维度分布:")
        for dim, count in sorted(total_dimension_count.items()):
            print(f"{dim}维: {count} 个样本")
    else:
        print(f"\n合并的样本维度分布:")
        for dim, count in sorted(dimension_samples.items()):
            print(f"{dim}维: {count} 个样本")

    print(f"所有批次文件已合并到主文件: {filename}")
    print(f"批次文件已清理完成")
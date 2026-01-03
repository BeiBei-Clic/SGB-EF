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
    """单个进程处理一个批次的数据生成

    Args:
        args: (batch_idx, current_batch_size, max_dim, n_points, max_depth,
               max_expr_length, batch_filename, verbose, process_id)

    Returns:
        (批次索引, 生成的样本列表, 维度统计)
    """
    (batch_idx, current_batch_size, max_dim, n_points, max_depth,
     max_expr_length, batch_filename, verbose, process_id) = args

    # 设置进程特定的随机种子
    current_time_ms = int(time.time() * 1000000)
    seed_base = current_time_ms + (process_id << 16) + (batch_idx << 8) + os.getpid()
    seed_val = hash(str(seed_base)) & 0x7fffffff

    random.seed(seed_val)
    np.random.seed(seed_val)

    process_prefix = f"[B{batch_idx+1}]"
    debug_log_path = f"logs/worker_batch_{batch_idx+1}_pid_{os.getpid()}.log"
    os.makedirs("logs", exist_ok=True)

    def debug_log(msg):
        """写入调试日志"""
        with open(debug_log_path, "a") as f:
            f.write(f"{msg}\n")

    batch_samples = []
    dimension_count = {}
    sample_count = 0
    attempt_count = 0
    fail_count = 0
    consecutive_fails = 0
    SAMPLE_TIMEOUT = 10.0

    while sample_count < current_batch_size:
        attempt_count += 1
        consecutive_fails += 1
        unique_factor = random.randint(0, 999999)
        sample_id = f"{process_prefix}_sample{sample_count}_{os.getpid()}_{unique_factor}"

        try:
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

            if generated_samples:
                dim = generated_samples[0]["input_dimension"]
                for sample in generated_samples:
                    if sample_count >= current_batch_size:
                        break
                    batch_samples.append(sample)
                    sample_count += 1
                    dimension_count[dim] = dimension_count.get(dim, 0) + 1
                consecutive_fails = 0
            else:
                fail_count += 1
                if fail_count % 100 == 0:
                    debug_log(f"样本失败 {fail_count}次 | 连续失败{consecutive_fails}次")
                _sample_logger.sample_failed(sample_id, "No samples generated")
                continue

        except TimeoutError:
            fail_count += 1
            debug_log(f"超时 | TIMEOUT={SAMPLE_TIMEOUT}s | fail_count={fail_count}")
            _sample_logger.sample_timeout(sample_id, SAMPLE_TIMEOUT)
            continue

        except Exception as e:
            debug_log(f"异常 | {type(e).__name__}: {str(e)[:100]}")
            _sample_logger.sample_error(sample_id, type(e).__name__, str(e))

            if batch_samples:
                try:
                    os.makedirs(os.path.dirname(batch_filename), exist_ok=True)
                    with open(batch_filename, 'w', encoding='utf-8') as f:
                        for sample in batch_samples:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    debug_log(f"异常保存 | 已保存{len(batch_samples)}个样本")
                except Exception as save_error:
                    debug_log(f"保存失败 | {save_error}")

            debug_log(f"批次失败 | {type(e).__name__}: {str(e)[:100]}")
            return batch_idx, -1, {}

    debug_log(f"批次完成 | 成功{sample_count}个样本 | 尝试{attempt_count}次 | 失败{fail_count}次")

    if batch_samples:
        os.makedirs(os.path.dirname(batch_filename), exist_ok=True)
        with open(batch_filename, 'w', encoding='utf-8') as f:
            for sample in batch_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        debug_log(f"保存成功 | 样本数={len(batch_samples)}")

    return batch_idx, len(batch_samples), dimension_count

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4, seed: int = None) -> Dict:
    """生成单个样本

    Args:
        input_dimension: 输入维度
        n_points: 数据点数量
        max_depth: 表达式最大深度
        seed: 随机种子，None表示使用系统随机源
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    x_values = [list(point) for point in np.random.uniform(-5.0, 5.0, (n_points, input_dimension))]
    x_array = np.array(x_values)
    target_expr = generate_random_expr(input_dimension, max_depth)

    success, y_values = evaluate_expression_safe(target_expr, x_array)
    if not success:
        raise ValueError(f"表达式计算失败: {target_expr}")

    curr_expr = corrupt_expression(target_expr)

    return {
        "input_dimension": input_dimension,
        "x": x_values,
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
        with open(index_filename, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        return {int(dim_str): positions for dim_str, positions in index_data.items()}

    # 扫描文件创建索引
    dimension_samples = {}
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

    # 保存索引
    index_dir = os.path.dirname(index_filename)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump({str(dim): [int(pos) for pos in positions] for dim, positions in dimension_samples.items()}, f, indent=2)

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
    alignment_method: str = 'randomized',
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
        alignment_method: 对齐方法，'levenshtein' (确定性) 或 'randomized' (随机化，来自Edit Flows论文)
    """
    set_logger(_sample_logger)

    from src.symbolic.sample_generator import set_alignment_method
    set_alignment_method(alignment_method)

    # 设置主随机种子
    main_time_ms = int(time.time() * 1000000)
    main_seed_base = main_time_ms + os.getpid() + (num_samples & 0xffff)
    seed_val = hash(str(main_seed_base)) & 0x7fffffff
    random.seed(seed_val)
    np.random.seed(seed_val)

    # 主文件使用parquet格式
    filename = f"data/flow_samples_{num_samples}_{max_dim}dim_{n_points}pts_{max_depth}depth_{max_expr_length}len.parquet"
    num_batches = (num_samples + batch_size - 1) // batch_size
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    batch_filenames = [f"{temp_dir}/{os.path.basename(filename).replace('.parquet', f'_batch_{i + 1}.txt')}" for i in range(num_batches)]

    # 断点续传检查逻辑（只检查parquet）
    # 情况1：parquet文件存在 → 数据完整，直接返回
    if os.path.exists(filename):
        if verbose:
            print(f"✓ Parquet文件已存在，跳过生成: {filename}")
        return

    # 情况2：parquet不存在，检查是否有中断的生成任务
    txt_filename = filename.replace('.parquet', '.txt')
    if os.path.exists(txt_filename) and any(os.path.exists(f) for f in batch_filenames):
        if verbose:
            print(f"检测到中断的生成任务，正在恢复...")
        merge_batches_to_main_file(txt_filename, batch_filenames, num_batches, verbose=verbose)
        return

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    total_dimension_count = {}
    retry_count = 0
    all_success = False

    while not all_success:
        batch_tasks = []
        for batch_idx in range(num_batches):
            batch_filename = batch_filenames[batch_idx]
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

            if os.path.exists(batch_filename):
                continue

            process_id = len(batch_tasks) % num_processes
            batch_tasks.append((
                batch_idx, current_batch_size, max_dim, n_points, max_depth,
                max_expr_length, batch_filename, verbose, process_id
            ))

        if not batch_tasks:
            all_success = True
        else:
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    chunksize = max(1, len(batch_tasks) // (num_processes * 4))
                    results_iter = pool.imap_unordered(
                        generate_batch_worker,
                        batch_tasks,
                        chunksize=chunksize
                    )

                    failed_batches = []
                    completed_count = 0

                    for result in results_iter:
                        batch_idx, sample_count, dimension_count = result
                        completed_count += 1

                        if sample_count == -1:
                            failed_batches.append(batch_idx)
                            batch_filename = batch_filenames[batch_idx]
                            if os.path.exists(batch_filename):
                                os.remove(batch_filename)
                        else:
                            for dim, count in dimension_count.items():
                                total_dimension_count[dim] = total_dimension_count.get(dim, 0) + count

                if verbose:
                    print(f"\n所有 {len(batch_tasks)} 个批次任务处理完成")

                if failed_batches:
                    retry_count += 1
                else:
                    all_success = True

            except (BrokenPipeError, KeyboardInterrupt, Exception) as e:
                if isinstance(e, (BrokenPipeError, KeyboardInterrupt)):
                    raise
                else:
                    retry_count += 1

        # 验证批次完整性
        missing_batches = [batch_idx for batch_idx, batch_filename in enumerate(batch_filenames) if not os.path.exists(batch_filename)]

        if missing_batches:
            retry_count += 1
        else:
            all_success = True
            break

    if verbose and total_dimension_count:
        dim_dist = ', '.join(f"{dim}维:{count}个" for dim, count in sorted(total_dimension_count.items()))
        print(f"\n已完成批次的维度分布: {dim_dist}")

    # 合并批次文件到txt，然后生成parquet
    txt_filename = filename.replace('.parquet', '.txt')
    merge_batches_to_main_file(txt_filename, batch_filenames, num_batches, verbose=verbose)


def merge_batches_to_main_file(filename: str, batch_filenames: List[str], num_batches: int, verbose: bool = True):
    """合并批次文件到主文件，并生成Parquet格式

    Args:
        filename: txt主文件名
        batch_filenames: 批次文件列表
        num_batches: 总批次数
        verbose: 是否显示详细输出
    """
    index_filename = filename.replace('.txt', '_dimension_index.json')
    parquet_filename = filename.replace('.txt', '.parquet')
    dimension_samples = {}

    if os.path.exists(index_filename) and os.path.exists(filename):
        with open(index_filename, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        dimension_samples = {int(dim_str): positions for dim_str, positions in index_data.items()}

    with open(filename, 'a', encoding='utf-8') as main_file:
        for batch_idx in range(num_batches):
            batch_filename = batch_filenames[batch_idx]
            if os.path.exists(batch_filename):
                batch_samples = []
                with open(batch_filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            batch_samples.append(json.loads(line))

                for sample in batch_samples:
                    pos = main_file.tell()
                    dim = sample['input_dimension']
                    if dim not in dimension_samples:
                        dimension_samples[dim] = []
                    dimension_samples[dim].append(pos)

                    main_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

                os.remove(batch_filename)

                batch_number = batch_idx + 1
                import glob
                worker_log_pattern = f"logs/worker_batch_{batch_number}_pid_*.log"
                for log_file in glob.glob(worker_log_pattern):
                    try:
                        os.remove(log_file)
                    except Exception:
                        pass

    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump({str(dim): [int(pos) for pos in positions] for dim, positions in dimension_samples.items()}, f, indent=2)

    # 生成Parquet文件（更高效的格式）
    if not os.path.exists(parquet_filename):
        if verbose:
            print(f"\n正在生成 Parquet 文件: {parquet_filename}")

        import pandas as pd
        from tqdm import tqdm

        # 读取txt文件中的所有样本
        samples = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取txt样本", unit="样本"):
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # 转换为DataFrame并保存为Parquet
        df = pd.DataFrame(samples)
        df.to_parquet(
            parquet_filename,
            engine='pyarrow',
            compression='snappy',  # 快速压缩
            index=False
        )

        if verbose:
            txt_size = os.path.getsize(filename) / (1024**3)
            parquet_size = os.path.getsize(parquet_filename) / (1024**3)
            compression_ratio = (1 - parquet_size / txt_size) * 100
            print(f"✓ Parquet 文件生成完成:")
            print(f"  TXT 大小:   {txt_size:.2f} GB")
            print(f"  Parquet 大小: {parquet_size:.2f} GB (压缩 {compression_ratio:.1f}%)")
            print(f"  样本数量:   {len(samples)}")
    elif verbose:
        print(f"✓ Parquet 文件已存在，跳过生成: {parquet_filename}")
"""
符号回归数据生成器，用于EditFlow预训练
"""

import numpy as np
import random
import os
import warnings
import time
import json
import multiprocessing
import subprocess
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
                    pass
                _sample_logger.sample_failed(sample_id, "No samples generated")
                continue

        except TimeoutError:
            fail_count += 1
            _sample_logger.sample_timeout(sample_id, SAMPLE_TIMEOUT)
            continue

        except Exception as e:
            _sample_logger.sample_error(sample_id, type(e).__name__, str(e))

            if batch_samples:
                try:
                    os.makedirs(os.path.dirname(batch_filename), exist_ok=True)
                    with open(batch_filename, 'w', encoding='utf-8') as f:
                        for sample in batch_samples:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                except Exception as save_error:
                    pass

            return batch_idx, -1, {}

    if batch_samples:
        os.makedirs(os.path.dirname(batch_filename), exist_ok=True)
        with open(batch_filename, 'w', encoding='utf-8') as f:
            for sample in batch_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return batch_idx, len(batch_samples), dimension_count

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

    # 情况3：txt文件存在但批次文件都不存在 → 数据生成已完成，直接生成parquet
    if os.path.exists(txt_filename) and not any(os.path.exists(f) for f in batch_filenames):
        if verbose:
            print(f"检测到已完成的数据生成(txt文件存在，批次文件已合并)，正在生成 Parquet 文件...")
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

    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump({str(dim): [int(pos) for pos in positions] for dim, positions in dimension_samples.items()}, f, indent=2)

    # 生成Parquet文件（更高效的格式）- 使用分批读取避免内存溢出
    if not os.path.exists(parquet_filename):
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔄 正在生成 Parquet 文件")
            print(f"{'='*70}")
            print(f"📁 源文件: {filename}")
            print(f"📁 目标文件: {parquet_filename}")

        import pandas as pd
        from tqdm import tqdm
        import pyarrow as pa
        import pyarrow.parquet as pq
        import time
        import psutil

        # 分批读取txt文件并写入parquet，避免一次性加载所有数据到内存
        BATCH_SIZE = 50000  # 每批处理5万个样本
        samples_batch = []
        total_samples = 0
        batch_num = 0

        # 记录开始时间
        start_time = time.time()

        # 获取总行数用于进度显示（使用wc命令快速统计）
        if verbose:
            print(f"\n⏳ 正在统计总样本数...")
        result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True)
        total_lines = int(result.stdout.split()[0])

        if verbose:
            print(f"📊 转换配置:")
            print(f"  • 总样本数: {total_lines:,}")
            print(f"  • 批次大小: {BATCH_SIZE:,} 样本/批")
            print(f"  • 预计批次数: {(total_lines + BATCH_SIZE - 1) // BATCH_SIZE}")
            print(f"\n{'='*70}\n")

        # 使用pyarrow.ParquetWriter进行追加写入
        writer = None
        schema = None

        # 创建增强的进度条
        with open(filename, 'r', encoding='utf-8') as f:
            pbar = tqdm(
                total=total_lines,
                desc="📦 转换进度",
                unit="样本",
                unit_scale=True,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            try:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        samples_batch.append(sample)

                        # 当批次达到指定大小时，写入parquet
                        if len(samples_batch) >= BATCH_SIZE:
                            df_batch = pd.DataFrame(samples_batch)

                            # 第一次写入时初始化writer和schema
                            if writer is None:
                                schema = pa.Table.from_pandas(df_batch).schema
                                writer = pq.ParquetWriter(
                                    parquet_filename,
                                    schema=schema,
                                    compression='snappy'
                                )

                            # 写入当前批次
                            table = pa.Table.from_pandas(df_batch, schema=schema)
                            writer.write_table(table)

                            total_samples += len(samples_batch)
                            batch_num += 1

                            # 更新进度条
                            pbar.update(BATCH_SIZE)

                            # 每批次更新详细统计
                            if verbose:
                                elapsed = time.time() - start_time
                                speed = total_samples / elapsed if elapsed > 0 else 0
                                progress_pct = 100 * total_samples / total_lines
                                eta = (total_lines - total_samples) / speed if speed > 0 else 0

                                # 获取内存使用情况
                                process = psutil.Process()
                                memory_mb = process.memory_info().rss / (1024**2)

                                # 每5个批次显示一次详细统计
                                if batch_num % 5 == 0:
                                    pbar.write(
                                        f"  📊 批次 #{batch_num:3d} | "
                                        f"进度: {progress_pct:6.2f}% | "
                                        f"速度: {speed:8.1f} 样本/秒 | "
                                        f"ETA: {eta/60:5.1f}分钟 | "
                                        f"内存: {memory_mb:6.1f}MB"
                                    )

                            samples_batch = []  # 清空批次，释放内存

                # 处理最后剩余的样本
                if samples_batch:
                    df_batch = pd.DataFrame(samples_batch)

                    if writer is None:
                        schema = pa.Table.from_pandas(df_batch).schema
                        writer = pq.ParquetWriter(
                            parquet_filename,
                            schema=schema,
                            compression='snappy'
                        )

                    table = pa.Table.from_pandas(df_batch, schema=schema)
                    writer.write_table(table)
                    total_samples += len(samples_batch)
                    pbar.update(len(samples_batch))

            finally:
                pbar.close()

        # 关闭writer
        if writer is not None:
            writer.close()

        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        avg_speed = total_samples / total_time if total_time > 0 else 0

        if verbose:
            txt_size = os.path.getsize(filename) / (1024**3)
            parquet_size = os.path.getsize(parquet_filename) / (1024**3)
            compression_ratio = (1 - parquet_size / txt_size) * 100

            print(f"\n{'='*70}")
            print(f"✅ Parquet 文件生成完成")
            print(f"{'='*70}")
            print(f"📁 文件信息:")
            print(f"  • TXT 大小:     {txt_size:.2f} GB")
            print(f"  • Parquet 大小:  {parquet_size:.2f} GB")
            print(f"  • 压缩率:       {compression_ratio:.1f}%")
            print(f"  • 样本数量:     {total_samples:,}")
            print(f"\n⏱️  性能统计:")
            print(f"  • 总耗时:       {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
            print(f"  • 平均速度:     {avg_speed:.1f} 样本/秒")
            print(f"  • 批次总数:     {batch_num} 批")
            print(f"  • 平均批次耗时: {total_time/batch_num if batch_num > 0 else 0:.2f} 秒/批")
            print(f"{'='*70}\n")
    elif verbose:
        print(f"✓ Parquet 文件已存在，跳过生成: {parquet_filename}")
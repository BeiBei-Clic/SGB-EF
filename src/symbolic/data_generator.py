"""
符号回归数据生成器，用于EditFlow预训练
"""

import datetime
import numpy as np
import random
import sympy as sp
import os
import warnings
import time
import json
import pickle
import logging
import threading
from typing import List, Dict, Tuple
from src.utils.timeout_utils import TimeoutError, with_timeout
from src.utils.log_utils import (
    _write_log, log_sample_step,
    log_expression_eval,
    log_batch_progress
)
from src.symbolic.symbolic_utils import (
    expr_to_tree, generate_random_expr,
    generate_reduction_sequence,
    evaluate_expression_safe,
    levenshtein_alignment_with_gap,
)
from src.symbolic.corruption import corrupt_expression
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 常量定义
MAX_RETRIES = 10  # 表达式生成和计算的最大重试次数

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4) -> Dict:
    """生成单个样本"""
    sample_id = f"sample_{input_dimension}dim_{int(time.time() * 1000) % 1000000}"

    log_sample_step(sample_id, f"开始生成 {input_dimension}维样本")

    # 统一生成数据点，确保每个数据点是[x0, x1, x2, ...]的形式
    log_sample_step(sample_id, "生成数据点")
    x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, input_dimension))
    x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式

    # 转换为numpy数组用于表达式计算
    x_array = np.array(x_values)

    # 生成目标表达式
    log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
    target_expr = generate_random_expr(input_dimension, max_depth)

    # 计算目标值
    log_sample_step(sample_id, "计算目标表达式值")
    success, y_values = evaluate_expression_safe(target_expr, x_array)
    if not success:
        # 如果计算失败，抛出异常让调用者处理
        raise ValueError(f"表达式计算失败: {target_expr}")

    # 生成当前表达式
    log_sample_step(sample_id, "生成当前表达式(破坏)")
    curr_expr = corrupt_expression(target_expr, 0.5)

    _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] SUCCESS")

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
    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
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
    max_expr_length: int = 24,
    batch_size: int = 50000,
    verbose: bool = True,
    # === 新增参数 ===
    process_rank: int = 0,
    world_size: int = 1,
):
    """生成用于EditFlow连续流训练的数据文件，支持断点续传和多进程并行生成

    Args:
        num_samples: 总样本数
        max_dim: 最大维度
        n_points: 每个样本的数据点数
        max_depth: 表达式最大深度
        max_expr_length: 表达式最大字符长度（默认24）
        batch_size: 批次大小
        verbose: 是否显示详细输出
        process_rank: 当前进程ID (0, 1, 2...)
        world_size: 总进程数
    """

    seed_val = int(time.time()) % (2**32 - 1)
    random.seed(seed_val)
    np.random.seed(seed_val)

    # 检查是否存在缓存文件
    filename = f"data/flow_samples_{num_samples}_{max_dim}dim_{n_points}pts_{max_depth}depth.txt"

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

    # === 单进程数据生成 ===
    if verbose:
        print(f"分批生成 {num_samples} 个样本，共 {num_batches} 批...")

    # 2. 分批生成数据样本，支持断点续传
    total_dimension_count = {}

    # 遍历所有批次
    for batch_idx in range(num_batches):
        # 获取当前批次的文件名
        batch_filename = batch_filenames[batch_idx]
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        batch_start_time = time.time()

        # 如果这个批次文件已经存在，直接跳过
        if os.path.exists(batch_filename):
            if verbose:
                print(f"跳过已存在的批次 {batch_idx + 1}")
            continue

        if verbose:
            print(f"开始生成第 {batch_idx + 1}/{num_batches} 批...")

        batch_samples, dimension_count, sample_count = [], {}, 0
        # 进度条
        pbar = tqdm(total=current_batch_size, desc=f"第{batch_idx + 1}批")
        SAMPLE_TIMEOUT = 5.0  # 样本最大处理时间

        # 记录批次开始
        log_batch_progress(batch_idx, num_batches, batch_idx * batch_size, num_samples)

        while sample_count < current_batch_size:
            sample_id = f"batch{batch_idx+1}_sample{sample_count}_{int(time.time() * 1000) % 1000000}"
            sample_start_time = time.time()

            try:
                dim = random.randint(1, max_dim)
                dimension_count[dim] = dimension_count.get(dim, 0) + 1

                log_sample_step(sample_id, "开始生成样本", f"批次{batch_idx+1}, 维度{dim}, 样本数{sample_count}/{current_batch_size}")

                # 生成数据点
                log_sample_step(sample_id, "生成数据点", f"{n_points}个点, {dim}维")
                x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, dim))
                x_values = [list(point) for point in x_values_raw]
                x_array = np.array(x_values)

                # 生成目标表达式，添加超时保护和重试机制
                log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
                target_expr = None
                expr_valid = False

                for retry_count in range(MAX_RETRIES):
                    retry_start = time.time()
                    try:
                        log_sample_step(sample_id, f"表达式生成尝试 {retry_count+1}/{MAX_RETRIES}", f"维度={dim}")
                        target_expr = with_timeout(generate_random_expr, 2.0, dim, max_depth)
                        expr_str = str(target_expr)
                        retry_time = (time.time() - retry_start) * 1000

                        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EXPR_GEN '{expr_str}' len={len(expr_str)}" + (f" | retry={retry_count+1} | time={retry_time:.1f}ms" if f"retry={retry_count+1} | time={retry_time:.1f}ms" else ""))

                        # 验证表达式是否符合要求
                        expr_tree = expr_to_tree(target_expr)
                        expr_tokens = expr_tree.split(',')

                        if len(expr_tokens) <= 1:
                            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] RETRY_EXPRESSION_TOKENS_TOO_FEW retry={retry_count+1} tokens={len(expr_tokens)} expr='{expr_str}'")
                            continue  # 继续下一次重试

                        if len(expr_str) > max_expr_length:
                            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] RETRY_EXPRESSION_TOO_LONG retry={retry_count+1} length={len(expr_str)} expr='{expr_str[:50]}{'...' if len(expr_str) > 50 else ''}'")
                            continue  # 继续下一次重试

                        if target_expr.has(sp.I) or 'I' in expr_str:
                            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] RETRY_EXPRESSION_COMPLEX retry={retry_count+1} expr='{expr_str}'")
                            continue  # 继续下一次重试

                        # 表达式验证通过
                        expr_valid = True
                        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EXPRESSION_ACCEPTED retry={retry_count+1} tokens={len(expr_tokens)} length={len(expr_str)} expr='{expr_str}'")
                        break  # 退出重试循环

                    except TimeoutError as te:
                        retry_time = time.time() - retry_start
                        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] TIMEOUT generate_random_expr >2.0s | dim={dim}, attempt={retry_count+1}, duration={retry_time:.1f}s")
                        if retry_count == MAX_RETRIES - 1:
                            log_sample_step(sample_id, "跳过生成超时的表达式", f"已重试{MAX_RETRIES}次")
                            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] ERROR generate_random_expr_timeout: {type(te).__name__}: {te}")
                    except Exception as expr_error:
                        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] RETRY {retry_count + 1}/{MAX_RETRIES} error: {str(expr_error)[:30]}")
                        if retry_count == MAX_RETRIES - 1:
                            log_sample_step(sample_id, "跳过生成失败的表达式", f"已重试{MAX_RETRIES}次")
                            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] ERROR generate_random_expr_failed: {type(expr_error).__name__}: {expr_error}")

                if not expr_valid:
                    # 表达式验证失败，重新生成样本
                    continue

                expr_str = str(target_expr)

                # 尝试计算目标值
                log_sample_step(sample_id, "计算目标表达式值")
                eval_start = time.time()
                success, y_target = evaluate_expression_safe(
                    target_expr, x_array,
                    error_callback=lambda err: log_expression_eval(sample_id, expr_str, (time.time() - eval_start) * 1000, False, err)
                )
                eval_time = (time.time() - eval_start) * 1000
                if success:
                    log_expression_eval(sample_id, expr_str, eval_time, True)
                if not success:
                    # 不增加sample_count，直接重新生成
                    continue

                # 生成删减序列
                log_sample_step(sample_id, "生成删减序列")
                reduction_start = time.time()
                reduction_sequence = generate_reduction_sequence(target_expr)
                reduction_time = (time.time() - reduction_start) * 1000
                _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] REDUCE {len(reduction_sequence)} steps | time={reduction_time:.1f}ms")

                # 为删减序列中的每个表达式创建样本
                for i, reduced_expr in enumerate(reduction_sequence):
                    if sample_count >= current_batch_size:
                        break

                    log_sample_step(sample_id, f"处理删减表达式 {i+1}/{len(reduction_sequence)}", f"表达式: {str(reduced_expr)}")

                    # 对删减后的表达式应用额外的随机破坏
                    corruption_start = time.time()
                    curr_expr = corrupt_expression(reduced_expr)
                    corruption_time = (time.time() - corruption_start) * 1000
                    _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] CORRUPT step{i+1} | time={corruption_time:.1f}ms")

                    # 检查删减后的表达式是否与目标表达式相同，如果相同则无学习意义
                    if curr_expr == target_expr:
                        log_sample_step(sample_id, f"跳过相同的删减表达式 {i+1}", f"破坏后表达式与目标表达式相同")
                        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] SKIP_DUPLICATE step{i+1}")
                        continue

                    # 检查删减后的表达式是否包含复数单位
                    if curr_expr.has(sp.I) or 'I' in str(curr_expr):
                        log_sample_step(sample_id, f"跳过复数删减表达式 {i+1}", f"表达式包含复数单位")
                        continue

                    # 尝试计算当前值
                    eval_curr_start = time.time()
                    success, y_curr = evaluate_expression_safe(
                        curr_expr, x_array,
                        error_callback=lambda err: log_sample_step(sample_id, f"跳过计算失败的删减表达式 {i+1}", f"计算错误: {err}")
                    )
                    eval_curr_time = (time.time() - eval_curr_start) * 1000
                    _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EVAL_CURR step{i+1} success={success} | time={eval_curr_time:.1f}ms")
                    if not success:
                        continue

                    # 转换为token序列
                    tree_start = time.time()
                    target_tree = expr_to_tree(target_expr)
                    curr_tree = expr_to_tree(curr_expr)
                    target_tokens = target_tree.split(',')
                    curr_tokens = curr_tree.split(',')
                    tree_time = (time.time() - tree_start) * 1000

                    # 对齐到Z空间，包含gap token
                    align_start = time.time()
                    z0_tokens, z1_tokens = levenshtein_alignment_with_gap(curr_tokens, target_tokens)
                    align_time = (time.time() - align_start) * 1000
                    _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] ALIGN step{i+1} | tree_time={tree_time:.1f}ms align_time={align_time:.1f}ms tokens={len(curr_tokens)}→{len(target_tokens)}")

                    batch_samples.append({
                        "input_dimension": dim,
                        "x_values": x_values,  # 保持与原代码一致的字段名
                        "y_target": y_target.tolist(),
                        "y_curr": y_curr.tolist(),
                        "residuals": (y_target - y_curr).tolist(),
                        "tree_gt": target_tree,
                        "exp_gt": str(target_expr),
                        "tree_cur1": curr_tree,
                        "exp_cur1": str(curr_expr),
                        "curr_tokens": curr_tokens,
                        "target_tokens": target_tokens,
                        "z0_tokens": z0_tokens,
                        "z1_tokens": z1_tokens
                    })

                    sample_count += 1
                    pbar.update(1)

                _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] SUCCESS")

            except Exception as e:
                duration = time.time() - sample_start_time
                steps = [f"错误: {str(e)}"]
                _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] ERROR sample_generation: {type(e).__name__}: {e}")
                _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] STUCK {duration:.1f}s steps={len(steps)}")
                print(f"警告: 生成样本时出错，跳过该样本: {e}")
                continue

        pbar.close()

        # 计算批次统计
        batch_duration = time.time() - batch_start_time
        avg_time = batch_duration / current_batch_size if current_batch_size > 0 else 0
        success_rate = len(batch_samples) / current_batch_size if current_batch_size > 0 else 0

        # 记录批次完成统计
        log_batch_progress(batch_idx, num_batches, (batch_idx + 1) * batch_size, num_samples,
                          avg_time, success_rate)

        # 累积维度统计
        for dim, count in dimension_count.items():
            total_dimension_count[dim] = total_dimension_count.get(dim, 0) + count

        # 立即保存当前批次
        batch_filename = batch_filenames[batch_idx]
        if verbose:
            print(f"保存数据到 {batch_filename}...")
        os.makedirs(os.path.dirname(batch_filename), exist_ok=True)
        with open(batch_filename, 'w', encoding='utf-8') as f:
            for sample in batch_samples:
                sample_line = json.dumps(sample, ensure_ascii=False)
                f.write(sample_line + '\n')
        print(f"已保存 {len(batch_samples)} 个样本到 {batch_filename}")
        if verbose:
            print(f"第 {batch_idx + 1} 批完成并已保存到 {batch_filename}")
            print(f"当前批次维度分布:")
            for dim, count in sorted(dimension_count.items()):
                print(f"  {dim}维: {count} 个样本")

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

    dimension_samples = {}  # 存储每个维度的样本位置索引

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
                    # 记录当前位置
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

    # 保存维度索引文件
    index_filename = filename.replace('.txt', '_dimension_index.json')
    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    index_data = {}
    for dim, positions in dimension_samples.items():
        index_data[str(dim)] = [int(pos) for pos in positions]
    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    print(f"维度索引已保存到 {index_filename}")

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
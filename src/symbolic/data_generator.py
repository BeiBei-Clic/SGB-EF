"""
符号回归数据生成器，用于EditFlow预训练
"""

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
from src.utils.log_utils import log_sample_step, log_sample_success, log_sample_stuck, cleanup_successful_logs
from src.symbolic.symbolic_utils import (
    timed_simplify, expr_to_tree, generate_random_expr,
    apply_unary_op, apply_binary_op, corrupt_expression,
    reduce_expression, generate_reduction_sequence,
    preprocess_expression, evaluate_expr, levenshtein_alignment_with_gap,
    UNARY_OPS, BINARY_OPS
)
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4) -> Dict:
    """生成单个样本"""
    sample_id = f"sample_{input_dimension}dim_{int(time.time() * 1000) % 1000000}"
    start_time = time.time()
    steps = []

    log_sample_step(sample_id, f"开始生成 {input_dimension}维样本")

    # 统一生成数据点，确保每个数据点是[x0, x1, x2, ...]的形式
    log_sample_step(sample_id, "生成数据点")
    x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, input_dimension))
    x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式
    steps.append("数据点生成完成")

    # 转换为numpy数组用于表达式计算
    x_array = np.array(x_values)

    # 生成目标表达式
    log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
    target_expr = generate_random_expr(input_dimension, max_depth)
    steps.append(f"目标表达式: {str(target_expr)}")

    # 计算目标值
    log_sample_step(sample_id, "计算目标表达式值")
    y_values = evaluate_expr(target_expr, x_array)
    steps.append("目标值计算完成")

    # 生成当前表达式
    log_sample_step(sample_id, "生成当前表达式(破坏)")
    curr_expr = corrupt_expression(target_expr, 0.5)
    steps.append(f"当前表达式: {str(curr_expr)}")

    log_sample_success(sample_id)

    return {
        "input_dimension": input_dimension,
        "x": x_values,  # 现在是[[x0, x1, x2], [x3, x4, x5], ...]的形式
        "y": y_values.tolist(),
        "tree_gt": expr_to_tree(target_expr),
        "exp_gt": str(target_expr),
        "tree_cur1": expr_to_tree(curr_expr),
        "exp_cur1": str(curr_expr)
    }

def save_samples_to_txt(samples: List[Dict], filename: str):
    """将样本保存到txt文件，每行一个样本"""
    print(f"保存数据到 {filename}...")

    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            # 将样本转换为JSON格式并写入一行
            sample_line = json.dumps(sample, ensure_ascii=False)
            f.write(sample_line + '\n')

    print(f"已保存 {len(samples)} 个样本到 {filename}")

def get_dimension_index_filename(data_filename: str) -> str:
    """生成维度索引文件名"""
    return data_filename.replace('.txt', '_dimension_index.json')

def save_dimension_index(filename: str, dimension_samples: Dict[int, List[int]]):
    """保存维度索引文件

    Args:
        filename: 数据文件名
        dimension_samples: 维度到位置索引的映射
    """
    index_filename = get_dimension_index_filename(filename)

    # 将位置索引转换为普通列表（可能包含numpy int）
    index_data = {}
    for dim, positions in dimension_samples.items():
        index_data[str(dim)] = [int(pos) for pos in positions]

    with open(index_filename, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)

    print(f"维度索引已保存到 {index_filename}")

def load_dimension_index(filename: str) -> Dict[int, List[int]]:
    """加载维度索引文件"""
    index_filename = get_dimension_index_filename(filename)

    if not os.path.exists(index_filename):
        return None

    print(f"发现维度索引文件 {index_filename}，正在加载...")
    with open(index_filename, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    # 转换回原来的格式
    dimension_samples = {}
    for dim_str, positions in index_data.items():
        dimension_samples[int(dim_str)] = positions

    print(f"维度索引加载完成")
    return dimension_samples

def load_samples_from_txt(filename: str) -> List[Dict]:
    """从txt文件加载样本"""
    print(f"从 {filename} 加载数据...")

    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                samples.append(sample)

    print(f"已加载 {len(samples)} 个样本")
    return samples

def generate_flow_samples(num_samples: int, max_dim: int = 5, n_points: int = 100, max_depth: int = 4, batch_size: int = 50000):
    """生成用于EditFlow连续流训练的数据文件，支持断点续传"""

    # 设置真正的随机种子，确保每次运行生成不同的数据
    current_time = int(time.time()) % (2**32 - 1)  # 确保种子在有效范围内
    random.seed(current_time)
    np.random.seed(current_time)

    # 检查是否存在缓存文件
    filename = f"data/flow_samples_{num_samples}_{max_dim}dim_{n_points}pts_{max_depth}depth.txt"

    # 检查批次文件状态
    num_batches = (num_samples + batch_size - 1) // batch_size
    existing_batches = [
        filename.replace('.txt', f'_batch_{i + 1}.txt')
        for i in range(num_batches)
        if os.path.exists(filename.replace('.txt', f'_batch_{i + 1}.txt'))
    ]

    # 1. 主文件存在且无批次文件 → 数据完整
    if os.path.exists(filename) and not existing_batches:
        print(f"发现完整数据文件 {filename}")
        return

    # 2. 分批生成数据样本，支持断点续传
    all_samples = []
    total_dimension_count = {}

    print(f"分批生成 {num_samples} 个连续流训练样本，每批 {batch_size} 个...")

    # 检查已完成的批次
    completed_batches = []
    for batch_idx in range(num_batches):
        batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
        if os.path.exists(batch_filename):
            completed_batches.append(batch_idx)

    if completed_batches:
        print(f"发现已完成 {len(completed_batches)} 个批次，将从第 {len(completed_batches) + 1} 批开始继续生成...")

    # 按批次顺序生成
    for batch_idx in range(len(completed_batches), num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        print(f"\n生成第 {batch_idx + 1}/{num_batches} 批数据 ({current_batch_size} 个样本)...")

        batch_samples = []
        dimension_count = {}
        sample_count = 0
        pbar = tqdm(total=current_batch_size, desc=f"第{batch_idx + 1}批")

        while sample_count < current_batch_size:
            sample_start_time = time.time()
            sample_id = f"batch{batch_idx+1}_sample{sample_count}_{int(time.time() * 1000) % 1000000}"
            steps = []

            # 每个样本的最大处理时间（秒）
            SAMPLE_TIMEOUT = 5.0

            try:
                # 检查样本生成是否超时
                if time.time() - sample_start_time > SAMPLE_TIMEOUT:
                    log_sample_step(sample_id, "样本生成超时", f"超过{SAMPLE_TIMEOUT}秒")
                    sample_count += 1
                    pbar.update(1)
                    continue

                dim = random.randint(1, max_dim)
                dimension_count[dim] = dimension_count.get(dim, 0) + 1

                log_sample_step(sample_id, f"开始生成样本", f"批次{batch_idx+1}, 维度{dim}, 样本数{sample_count}/{current_batch_size}")

                # 生成数据点，统一处理确保每个数据点是[x0, x1, x2, ...]的形式
                log_sample_step(sample_id, "生成数据点", f"{n_points}个点, {dim}维")
                x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, dim))
                x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式
                x_array = np.array(x_values)  # 用于表达式计算
                steps.append("数据点生成完成")

                # 生成目标表达式，添加超时保护
                log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
                try:
                    target_expr = with_timeout(generate_random_expr, 2.0, dim, max_depth)
                    expr_str = str(target_expr)
                    steps.append(f"目标表达式: {expr_str}")
                except TimeoutError:
                    log_sample_step(sample_id, "跳过生成超时的表达式", "生成超时2秒")
                    sample_count += 1
                    pbar.update(1)
                    continue
                except Exception as expr_error:
                    log_sample_step(sample_id, "跳过生成失败的表达式", f"生成错误: {str(expr_error)}")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 如果表达式太复杂，跳过
                if len(expr_str) > 100:
                    log_sample_step(sample_id, "跳过复杂表达式", f"长度{len(expr_str)} > 100")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 预先检查表达式是否可能导致复数问题
                if target_expr.has(sp.I) or 'I' in expr_str:
                    log_sample_step(sample_id, "跳过复数表达式", f"包含复数单位I")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 尝试计算目标值，添加超时保护
                try:
                    log_sample_step(sample_id, "计算目标表达式值")
                    y_target = with_timeout(evaluate_expr, 2.0, target_expr, x_array)
                    steps.append("目标值计算完成")
                except TimeoutError:
                    log_sample_step(sample_id, "跳过计算超时的表达式", "计算超时2秒")
                    sample_count += 1
                    pbar.update(1)
                    continue
                except Exception as eval_error:
                    log_sample_step(sample_id, "跳过计算失败的表达式", f"计算错误: {str(eval_error)}")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 生成删减序列
                log_sample_step(sample_id, "生成删减序列")
                reduction_sequence = generate_reduction_sequence(target_expr)
                steps.append(f"删减序列长度: {len(reduction_sequence)}")

                # 为删减序列中的每个表达式创建样本
                for i, reduced_expr in enumerate(reduction_sequence):
                    if sample_count >= current_batch_size:
                        break

                    log_sample_step(sample_id, f"处理删减表达式 {i+1}/{len(reduction_sequence)}", f"表达式: {str(reduced_expr)}")

                    # 对删减后的表达式应用额外的随机破坏
                    curr_expr = corrupt_expression(reduced_expr, corruption_prob=0.3)

                    # 检查删减后的表达式是否包含复数单位
                    if curr_expr.has(sp.I) or 'I' in str(curr_expr):
                        log_sample_step(sample_id, f"跳过复数删减表达式 {i+1}", f"表达式包含复数单位")
                        continue

                    # 尝试计算当前值，添加超时保护
                    try:
                        y_curr = with_timeout(evaluate_expr, 1.0, curr_expr, x_array)
                    except TimeoutError:
                        log_sample_step(sample_id, f"跳过计算超时的删减表达式 {i+1}", "计算超时1秒")
                        continue
                    except Exception as eval_error:
                        log_sample_step(sample_id, f"跳过计算失败的删减表达式 {i+1}", f"计算错误: {str(eval_error)}")
                        continue

                    # 转换为token序列
                    target_tokens = expr_to_tree(target_expr).split(',')
                    curr_tokens = expr_to_tree(curr_expr).split(',')

                    # 对齐到Z空间，包含gap token
                    z0_tokens, z1_tokens = levenshtein_alignment_with_gap(curr_tokens, target_tokens)

                    batch_samples.append({
                        "input_dimension": dim,
                        "x_values": x_values,  # 保持与原代码一致的字段名
                        "y_target": y_target.tolist(),
                        "y_curr": y_curr.tolist(),
                        "residuals": (y_target - y_curr).tolist(),
                        "tree_gt": expr_to_tree(target_expr),
                        "exp_gt": str(target_expr),
                        "tree_cur1": expr_to_tree(curr_expr),
                        "exp_cur1": str(curr_expr),
                        "curr_tokens": curr_tokens,
                        "target_tokens": target_tokens,
                        "z0_tokens": z0_tokens,
                        "z1_tokens": z1_tokens
                    })

                    sample_count += 1
                    pbar.update(1)

                log_sample_success(sample_id)

            except Exception as e:
                # 记录卡住的样本
                duration = time.time() - sample_start_time
                steps.append(f"错误: {str(e)}")
                log_sample_stuck(sample_id, duration, steps)
                print(f"警告: 生成样本时出错，跳过该样本: {e}")
                continue

            if sample_count >= current_batch_size:
                break

        pbar.close()

        # 累积维度统计
        for dim, count in dimension_count.items():
            total_dimension_count[dim] = total_dimension_count.get(dim, 0) + count

        # 立即保存当前批次
        batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
        save_samples_to_txt(batch_samples, batch_filename)

        # 将批次样本添加到总样本中
        all_samples.extend(batch_samples)

        print(f"第 {batch_idx + 1} 批完成并已保存到 {batch_filename}")
        print(f"当前批次维度分布:")
        for dim, count in sorted(dimension_count.items()):
            print(f"  {dim}维: {count} 个样本")

        # 每20个批次清理一次日志，保留卡住的记录（最后一批不清理）
        if (batch_idx + 1) % 20 == 0 and batch_idx + 1 < num_batches:
            print("清理成功样本的详细日志...")
            cleanup_successful_logs()

    # 3. 按批次顺序合并所有剩余批次文件到主文件，并记录维度索引
    print(f"\n按批次顺序合并所有剩余批次文件到主文件...")
    dimension_samples = {}  # 存储每个维度的样本位置索引

    with open(filename, 'a', encoding='utf-8') as main_file:
        for batch_idx in range(num_batches):
            batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
            if os.path.exists(batch_filename):
                # 读取批次样本并记录位置
                batch_samples = load_samples_from_txt(batch_filename)
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
    save_dimension_index(filename, dimension_samples)

    print(f"\n总体样本维度分布:")
    for dim, count in sorted(total_dimension_count.items()):
        print(f"{dim}维: {count} 个样本")

    print(f"所有数据已保存到: {filename}")
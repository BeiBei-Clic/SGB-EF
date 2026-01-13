#!/usr/bin/env python3
"""
Feynman 数据集转换工具

将 Feynman 方程数据集转换为项目训练格式。
采用"直接破坏"策略：替换操作 + 新增冗余节点（不使用删减序列）。

核心策略：
- 直接使用数据文件的 y 值（高效）
- 表达式计算仅用于验证（准确）
- 残差控制：max(abs(residuals)) > 100 直接丢弃
- 只使用原项目支持的运算符（过滤不支持的方程）
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.symbolic.corruption import corrupt_expression
from src.symbolic.symbolic_utils import (
    expr_to_tree,
    randomized_alignment_with_gap,
)
from src.utils.special_tokens import SymbolicRegressionTokenizer
from src.symbolic.sample_generator import pad_sequence_with_bos

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 常量定义（与原项目保持一致）====================
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']
UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
SUPPORTED_OPS = set(BINARY_OPS + UNARY_OPS)

# ==================== 辅助函数：表达式验证 ====================

def is_supported_expression(expr_str: str) -> bool:
    """
    检查表达式是否只包含支持的运算符

    Args:
        expr_str: 表达式字符串，如 "x1*x2/((x4-x5)**2)"

    Returns:
        True 如果表达式只包含支持的运算符
    """
    try:
        # 尝试解析表达式
        expr = sp.sympify(expr_str)

        # 检查所有函数调用
        for func in expr.atoms(sp.Function):
            func_name = str(func.func).lower()
            # 精确匹配函数名（不支持的部分匹配，如 asin 不能匹配 sin）
            # 需要完全匹配或包含支持的运算符作为独立单词
            is_supported = False
            for op in SUPPORTED_OPS:
                # 精确匹配或 op 后面跟着括号（避免 asin 匹配 sin）
                if func_name == op or func_name.startswith(op + '('):
                    is_supported = True
                    break

            if not is_supported:
                logger.debug(f"不支持的函数: {func_name} 在表达式中: {expr_str}")
                return False

        return True

    except Exception as e:
        logger.error(f"表达式解析失败: {expr_str} - {e}")
        return False


def extract_variables(formula: str) -> List[str]:
    """
    从公式字符串中提取变量名

    Args:
        formula: 公式字符串，如 "G*m1*m2/((x2-x1)**2)"

    Returns:
        变量名列表，如 ['G', 'm1', 'm2', 'x1', 'x2']
    """
    # 移除已知函数名
    for func in SUPPORTED_OPS:
        formula = formula.replace(func, '')

    # 移除数学常数
    math_constants = ['pi', 'e', 'inf', 'nan']
    for const in math_constants:
        formula = formula.replace(const, '')

    # 提取所有标识符（字母开头的单词）
    import re
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)

    # 过滤掉 Python 关键字
    keywords = {'pi', 'e', 'inf', 'nan', 'True', 'False', 'None'}
    variables = list(set(identifiers) - keywords)

    return sorted(variables)


def map_formula_variables(formula: str, var_mapping: Dict[str, str]) -> str:
    """
    将原始变量名映射为 x0, x1, x2...

    Args:
        formula: 原始公式，如 "G*m1*m2/((x2-x1)**2)"
        var_mapping: 变量映射，如 {'G': 'x0', 'm1': 'x1', ...}

    Returns:
        转换后的公式，如 "x0*x1*x2/((x3-x4)**2)"
    """
    result = formula
    # 按长度降序排序，避免部分替换（如 m1 和 m）
    for original_var in sorted(var_mapping.keys(), key=len, reverse=True):
        result = result.replace(original_var, var_mapping[original_var])

    return result


# ==================== 模块A: 表达式加载与变量映射 ====================

def parse_feynman_equations(csv_path: str, encoding: str = 'utf-8-sig') -> Tuple[Dict[str, Dict], Dict]:
    """
    解析 FeynmanEquations.csv

    Args:
        csv_path: CSV文件路径

    Returns:
        (equations, stats)
        - equations: {filename: {formula, var_mapping, n_vars, filename}}
        - stats: 统计信息
    """
    equations = {}
    stats = {
        'total': 0,
        'supported': 0,
        'unsupported': 0,
        'too_many_vars': 0,
        'no_vars': 0,
        'parse_error': 0
    }

    unsupported_ops_examples = []

    with open(csv_path, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Filename'].strip()
            formula_original = row['Formula'].strip()

            stats['total'] += 1

            # 跳过空公式
            if not formula_original:
                stats['parse_error'] += 1
                continue

            # 提取变量名
            original_vars = extract_variables(formula_original)
            n_vars = len(original_vars)

            if n_vars == 0:
                stats['no_vars'] += 1
                continue

            if n_vars > 10:
                stats['too_many_vars'] += 1
                continue

            # 创建变量映射（注意：从 x0 开始，与原项目一致）
            var_mapping = {
                var: f"x{i}"
                for i, var in enumerate(sorted(original_vars))
            }

            # 转换公式
            formula_mapped = map_formula_variables(formula_original, var_mapping)

            # 验证表达式是否只包含支持的运算符
            if not is_supported_expression(formula_mapped):
                stats['unsupported'] += 1

                # 记录不支持的运算符示例
                try:
                    expr = sp.sympify(formula_mapped)
                    for func in expr.atoms(sp.Function):
                        func_name = str(func.func).lower()
                        if not any(op in func_name for op in SUPPORTED_OPS):
                            example = f"{filename}: {func_name}"
                            if example not in unsupported_ops_examples:
                                unsupported_ops_examples.append(example)
                                if len(unsupported_ops_examples) >= 5:  # 只保留前5个
                                    break
                except:
                    pass

                continue

            # 通过所有检查，添加到有效方程列表
            equations[filename] = {
                'formula': formula_mapped,
                'original_formula': formula_original,
                'var_mapping': var_mapping,
                'n_vars': n_vars,
                'filename': filename
            }
            stats['supported'] += 1

    logger.info(f"解析完成:")
    logger.info(f"  总方程数: {stats['total']}")
    logger.info(f"  支持的方程: {stats['supported']}")
    logger.info(f"  不支持的运算符: {stats['unsupported']}")
    logger.info(f"  变量过多(>10): {stats['too_many_vars']}")
    logger.info(f"  无变量: {stats['no_vars']}")
    logger.info(f"  解析错误: {stats['parse_error']}")

    if unsupported_ops_examples:
        logger.info(f"\n不支持的运算符示例:")
        for example in unsupported_ops_examples:
            logger.info(f"  - {example}")

    return equations, stats


# ==================== 模块B: 数据文件读取与验证 ====================

def read_and_validate_data(
    filename: str,
    var_mapping: Dict[str, str],
    formula: str,
    data_dir: str,
    n_verify: int = 100
) -> Optional[np.ndarray]:
    """
    读取数据文件并验证表达式

    Args:
        filename: 数据文件名（不含路径）
        var_mapping: 变量映射
        formula: 转换后的公式
        data_dir: 数据文件目录
        n_verify: 验证前n个点

    Returns:
        数据数组 (n_samples, n_vars + 1)，最后一列为y值
        如果验证失败，返回 None
    """
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        logger.error(f"数据文件不存在: {filepath}")
        return None

    try:
        # 读取数据文件
        data = np.loadtxt(filepath)
        n_samples = data.shape[0]

        if n_samples < n_verify:
            logger.warning(f"数据点不足: {filename} 只有 {n_samples} 个点")
            return None

        # 验证前 n_verify 个点
        verify_data = data[:n_verify]
        n_vars = len(var_mapping)

        # 提取变量列（假设前n列是变量）
        x_values = verify_data[:, :n_vars]
        y_file = verify_data[:, -1]  # 最后一列是y值

        # 用表达式计算y值
        try:
            y_calc = compute_expression(formula, x_values, var_mapping)

            # 验证误差
            max_error = np.max(np.abs(y_file - y_calc))
            mean_error = np.mean(np.abs(y_file - y_calc))

            if max_error > 1e-6:
                logger.warning(
                    f"验证失败 {filename}: max_error={max_error:.2e}, "
                    f"mean_error={mean_error:.2e}"
                )
                return None

            logger.debug(f"验证通过 {filename}: max_error={max_error:.2e}")

        except Exception as e:
            logger.error(f"表达式计算错误 {filename}: {e}")
            return None

        # 验证通过，返回完整数据
        return data

    except Exception as e:
        logger.error(f"读取数据文件失败 {filepath}: {e}")
        return None


def compute_expression(
    formula: str,
    x_values: np.ndarray,
    var_mapping: Dict[str, str]
) -> np.ndarray:
    """
    计算表达式值

    Args:
        formula: 公式字符串，如 "x0*x1/((x3-x4)**2)"
        x_values: 变量值数组 (n_samples, n_vars)
        var_mapping: 变量映射

    Returns:
        计算结果 (n_samples,)
    """
    # 构建符号表达式
    n_vars = len(var_mapping)
    symbols = [sp.symbols(f"x{i}") for i in range(n_vars)]
    symbol_dict = {f"x{i}": symbols[i] for i in range(n_vars)}

    try:
        expr = sp.sympify(formula, locals=symbol_dict)

        # 转换为数值函数
        f = sp.lambdify(symbols, expr, modules=['numpy', {'sqrt': np.sqrt}])

        # 计算
        result = f(*[x_values[:, i] for i in range(n_vars)])

        return np.asarray(result).flatten()

    except Exception as e:
        raise ValueError(f"表达式计算失败: {e}")


# ==================== 模块C: 增强的破坏函数 ====================

def add_redundant_node(expr: sp.Expr, max_dim: int) -> sp.Expr:
    """
    新增冗余节点操作：添加额外的不必要节点，使表达式变复杂且计算错误

    策略：
    1. 乘法冗余：x -> x * random(0.5, 2.0)
    2. 加法冗余：x -> x + random(-1, 1)
    3. 在子树中插入运算：sin(x) -> sin(x + 0.5)
    4. 幂运算冗余：x -> x ** random(0.8, 1.2)

    Args:
        expr: 原始表达式
        max_dim: 最大维度

    Returns:
        添加冗余节点后的表达式（值会改变）
    """
    strategies = [
        _add_extra_mul,
        _add_extra_add,
        _add_extra_pow,
        _insert_extra_in_subtree,
    ]

    # 随机选择策略
    strategy = random.choice(strategies)

    try:
        result = strategy(expr, max_dim)
        return result
    except Exception:
        # 如果失败，返回原表达式
        return expr


def _add_extra_mul(expr: sp.Expr, max_dim: int = None) -> sp.Expr:
    """添加额外乘法节点：expr * random(0.5, 2.0)，使值改变"""
    factor = random.uniform(0.5, 2.0)
    return expr * sp.Float(factor)


def _add_extra_add(expr: sp.Expr, max_dim: int = None) -> sp.Expr:
    """添加额外加法节点：expr + random(-1, 1)，使值改变"""
    offset = random.uniform(-1.0, 1.0)
    return expr + sp.Float(offset)


def _add_extra_pow(expr: sp.Expr, max_dim: int = None) -> sp.Expr:
    """添加额外幂运算：expr ** random(0.8, 1.2)，使值改变"""
    exponent = random.uniform(0.8, 1.2)
    return expr ** sp.Float(exponent)


def _insert_extra_in_subtree(expr: sp.Expr, max_dim: int) -> sp.Expr:
    """在子树中插入额外运算"""
    if not hasattr(expr, 'args') or not expr.args:
        # 叶子节点，添加简单运算
        return _add_extra_mul(expr, max_dim)

    # 随机选择一个子节点进行修改
    arg_idx = random.randint(0, len(expr.args) - 1)
    modified_args = list(expr.args)

    # 对选中的子节点添加额外运算
    original_arg = modified_args[arg_idx]
    operation = random.choice(['mul', 'add', 'pow'])

    if operation == 'mul':
        factor = random.uniform(0.5, 2.0)
        modified_args[arg_idx] = original_arg * sp.Float(factor)
    elif operation == 'add':
        offset = random.uniform(-1.0, 1.0)
        modified_args[arg_idx] = original_arg + sp.Float(offset)
    elif operation == 'pow':
        exponent = random.uniform(0.8, 1.2)
        modified_args[arg_idx] = original_arg ** sp.Float(exponent)

    # 构建新表达式
    return expr.func(*modified_args)


def corrupt_expression_enhanced(expr: sp.Expr, max_dim: int) -> sp.Expr:
    """
    增强版破坏函数

    随机选择：
    - 50% 原项目的替换操作（corrupt_expression）
    - 35% 新增冗余节点（让表达式变复杂）
    - 15% 混合操作

    Args:
        expr: 原始表达式
        max_dim: 最大维度

    Returns:
        破坏后的表达式
    """
    operation_type = random.random()

    if operation_type < 0.5:
        # 使用原项目的替换操作
        return corrupt_expression(expr)
    elif operation_type < 0.85:
        # 新增冗余节点（添加额外的不必要节点，使表达式变复杂且计算错误）
        return add_redundant_node(expr, max_dim)
    else:
        # 混合操作
        tmp = corrupt_expression(expr)
        return add_redundant_node(tmp, max_dim)


# ==================== 模块D: 样本生成（核心） ====================

def generate_single_feynman_sample(
    filename: str,
    formula: str,
    var_mapping: Dict[str, str],
    data: np.ndarray,
    max_dim: int = 10,
    n_points: int = 100,
    max_corruptions: int = 3,
    max_expr_length: int = 32
) -> Optional[Dict]:
    """
    生成单个Feynman样本

    流程：
    1. 从数据文件随机采样 n_points 行
    2. 使用数据文件的 y 值作为 y_target
    3. 对表达式应用 1-3 次破坏操作
    4. 计算 y_curr
    5. 计算 residuals
    6. 检查 max(abs(residuals)) < 100，否则丢弃
    7. 使用原项目的 expr_to_tree 转换为前缀表示
    8. 返回 JSON 格式样本

    Args:
        filename: 文件名
        formula: 转换后的公式
        var_mapping: 变量映射
        data: 完整数据数组
        max_dim: 最大维度
        n_points: 每个样本的数据点数
        max_corruptions: 最大破坏次数
        max_expr_length: 最大表达式长度

    Returns:
        样本字典或None（如果不满足条件）
    """
    n_vars = len(var_mapping)
    n_samples_total = data.shape[0]

    # 1. 随机采样
    if n_samples_total < n_points:
        return None

    indices = np.random.choice(n_samples_total, n_points, replace=False)
    sampled_data = data[indices]
    x_values = sampled_data[:, :n_vars]
    y_target = sampled_data[:, -1]  # 直接使用数据文件的y值

    # 2. 构建符号表达式
    symbols = [sp.symbols(f"x{i}") for i in range(n_vars)]
    symbol_dict = {f"x{i}": symbols[i] for i in range(n_vars)}
    expr_gt = sp.sympify(formula, locals=symbol_dict)

    # 3. 应用破坏操作
    n_corruptions = random.randint(1, max_corruptions)
    expr_curr = expr_gt

    for _ in range(n_corruptions):
        expr_curr = corrupt_expression_enhanced(expr_curr, max_dim)

    # 4. 计算 y_curr
    try:
        f_curr = sp.lambdify(symbols, expr_curr, modules=['numpy'])
        y_curr = f_curr(*[x_values[:, i] for i in range(n_vars)])
        y_curr = np.asarray(y_curr).flatten()
    except Exception:
        return None

    # 检查数值稳定性
    if np.any(np.isnan(y_curr)) or np.any(np.isinf(y_curr)):
        return None

    # 检查复数
    if np.iscomplexobj(y_curr):
        return None

    # 5. 计算残差
    residuals = y_target - y_curr

    # 6. 残差阈值检查（关键！）
    max_residual = np.max(np.abs(residuals))
    if max_residual > 100.0:
        # 直接丢弃，不裁剪
        return None

    # 7. 使用原项目的函数转换为前缀表示
    tree_gt = expr_to_tree(expr_gt)
    tree_cur = expr_to_tree(expr_curr)

    # 检查表达式长度
    target_tokens = tree_gt.split(',')
    curr_tokens = tree_cur.split(',')

    if len(target_tokens) > max_expr_length or len(curr_tokens) > max_expr_length:
        return None

    # 8. 对齐到Z空间（使用随机化对齐）
    z0_tokens, z1_tokens = randomized_alignment_with_gap(curr_tokens, target_tokens)

    # 9. 初始化tokenizer并转换为token IDs
    tokenizer = SymbolicRegressionTokenizer(max_dim=max_dim)
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

    z0_token_ids = pad_sequence_with_bos(
        tokenizer.convert_tokens_to_ids(z0_tokens),
        max_expr_length, bos_token_id, pad_token_id
    )
    z1_token_ids = pad_sequence_with_bos(
        tokenizer.convert_tokens_to_ids(z1_tokens),
        max_expr_length, bos_token_id, pad_token_id
    )

    # 10. 生成样本
    sample = {
        'input_dimension': n_vars,
        'x_values': x_values.tolist(),
        'y_target': y_target.tolist(),
        'y_curr': y_curr.tolist(),
        'residuals': residuals.tolist(),
        'tree_gt': tree_gt,
        'exp_gt': str(expr_gt),
        'tree_cur1': tree_cur,
        'exp_cur1': str(expr_curr),
        'curr_tokens': curr_tokens,
        'target_tokens': target_tokens,
        'z0_tokens': z0_tokens,
        'z1_tokens': z1_tokens,
        'z0_token_ids': z0_token_ids,
        'z1_token_ids': z1_token_ids,
    }

    return sample


# ==================== 模块E: 批量处理与Parquet生成 ====================

def process_single_equation(args: Tuple) -> List[Dict]:
    """
    处理单个方程（用于多进程）

    Args:
        args: (filename, formula, var_mapping, data, config)

    Returns:
        样本列表（包含原始样本和对称样本）
    """
    filename, formula, var_mapping, data, config = args

    samples = []
    num_samples = config['num_samples_per_eq']

    for _ in range(num_samples):
        sample = generate_single_feynman_sample(
            filename=filename,
            formula=formula,
            var_mapping=var_mapping,
            data=data,
            max_dim=config['max_dim'],
            n_points=config['n_points'],
            max_corruptions=config['max_corruptions'],
            max_expr_length=config['max_expr_length']
        )

        if sample is not None:
            samples.append(sample)

            # 生成对称样本（交换z0和z1）
            sym_sample = {
                'input_dimension': sample['input_dimension'],
                'x_values': sample['x_values'],
                'y_target': sample['y_curr'],  # 交换
                'y_curr': sample['y_target'],   # 交换
                'residuals': [-r for r in sample['residuals']],  # 取负
                'tree_gt': sample['tree_cur1'],  # 交换
                'exp_gt': sample['exp_cur1'],    # 交换
                'tree_cur1': sample['tree_gt'],  # 交换
                'exp_cur1': sample['exp_gt'],    # 交换
                'curr_tokens': sample['target_tokens'],  # 交换
                'target_tokens': sample['curr_tokens'],   # 交换
                'z0_tokens': sample['z1_tokens'],  # 交换
                'z1_tokens': sample['z0_tokens'],  # 交换
                'z0_token_ids': sample['z1_token_ids'],  # 交换
                'z1_token_ids': sample['z0_token_ids'],  # 交换
            }
            samples.append(sym_sample)

    return samples


def convert_feynman_dataset(
    csv_path: str,
    data_dir: str,
    output_dir: str,
    num_samples_per_eq: int = 10000,
    n_points: int = 100,
    max_dim: int = 10,
    max_corruptions: int = 3,
    max_expr_length: int = 32,
    batch_size: int = 10000,
    num_workers: int = 4
):
    """
    转换完整数据集

    Args:
        csv_path: CSV文件路径
        data_dir: 数据文件目录
        output_dir: 输出目录
        num_samples_per_eq: 每个方程生成样本数
        n_points: 每个样本的数据点数
        max_dim: 最大维度
        max_corruptions: 最大破坏次数
        max_expr_length: 最大表达式长度
        batch_size: 批处理大小
        num_workers: 并行进程数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 解析方程
    logger.info("=" * 60)
    logger.info("步骤 1: 解析方程并过滤不支持的运算符")
    logger.info("=" * 60)
    equations, stats = parse_feynman_equations(csv_path)

    if not equations:
        logger.error("未找到有效方程")
        return

    # 2. 读取和验证数据
    logger.info("\n" + "=" * 60)
    logger.info("步骤 2: 读取和验证数据文件")
    logger.info("=" * 60)
    valid_equations = []

    for filename, eq_info in tqdm(equations.items(), desc="验证数据"):
        data = read_and_validate_data(
            filename=filename,
            var_mapping=eq_info['var_mapping'],
            formula=eq_info['formula'],
            data_dir=data_dir
        )

        if data is not None:
            valid_equations.append((filename, eq_info, data))

    logger.info(f"\n有效方程数: {len(valid_equations)}/{len(equations)}")

    if not valid_equations:
        logger.error("没有有效方程")
        return

    # 3. 准备处理参数
    config = {
        'num_samples_per_eq': num_samples_per_eq,
        'max_dim': max_dim,
        'n_points': n_points,
        'max_corruptions': max_corruptions,
        'max_expr_length': max_expr_length
    }

    process_args = [
        (filename, eq_info['formula'], eq_info['var_mapping'], data, config)
        for filename, eq_info, data in valid_equations
    ]

    # 4. 多进程处理
    logger.info("\n" + "=" * 60)
    logger.info(f"步骤 3: 生成样本（{num_workers} 个进程）")
    logger.info("=" * 60)

    all_samples = []
    total_attempts = 0

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_equation, process_args),
            total=len(process_args),
            desc="生成样本"
        ))

    for samples in results:
        all_samples.extend(samples)
        total_attempts += num_samples_per_eq

    logger.info(f"\n生成样本数: {len(all_samples)}")
    logger.info(f"总尝试次数: {total_attempts}")
    logger.info(f"通过率: {len(all_samples)/total_attempts*100:.1f}%")

    if not all_samples:
        logger.error("未生成任何样本")
        return

    # 5. 转换为DataFrame并保存
    logger.info("\n" + "=" * 60)
    logger.info("步骤 4: 保存 Parquet 文件")
    logger.info("=" * 60)
    df = pd.DataFrame(all_samples)

    # 生成输出文件名
    output_file = os.path.join(
        output_dir,
        f"feynman_samples_{len(all_samples)}_{max_dim}dim_{n_points}pts_{max_expr_length}len.parquet"
    )

    df.to_parquet(output_file, index=False)
    logger.info(f"保存到: {output_file}")

    # 6. 生成维度索引
    logger.info("\n生成维度索引...")
    dimension_counts = df['input_dimension'].value_counts().sort_index()

    index_file = output_file.replace('.parquet', '_index.json')
    index_data = {
        'total_samples': len(all_samples),
        'dimension_counts': dimension_counts.to_dict(),
        'dimensions': [int(d) for d in dimension_counts.index],
        'num_equations': len(valid_equations),
        'samples_per_equation': num_samples_per_eq,
        'pass_rate': f"{len(all_samples)/total_attempts*100:.2f}%",
        'config': {
            'max_dim': max_dim,
            'n_points': n_points,
            'max_corruptions': max_corruptions,
            'max_expr_length': max_expr_length
        },
        'parsing_stats': stats
    }

    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"索引保存到: {index_file}")

    # 7. 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("数据集统计")
    logger.info("=" * 60)
    logger.info(f"总样本数: {len(all_samples)}")
    logger.info(f"维度分布:")
    for dim, count in dimension_counts.items():
        logger.info(f"  {int(dim)}维: {count} 样本 ({count/len(all_samples)*100:.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ 转换完成！")
    logger.info("=" * 60)


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='转换Feynman数据集')

    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='FeynmanEquations.csv 文件路径'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Feynman数据文件目录'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='输出目录'
    )

    parser.add_argument(
        '--num_samples_per_eq',
        type=int,
        default=10000,
        help='每个方程生成样本数'
    )

    parser.add_argument(
        '--n_points',
        type=int,
        default=100,
        help='每个样本的数据点数'
    )

    parser.add_argument(
        '--max_dim',
        type=int,
        default=10,
        help='最大输入维度'
    )

    parser.add_argument(
        '--max_corruptions',
        type=int,
        default=3,
        help='每个样本的破坏次数（1-3，随机）'
    )

    parser.add_argument(
        '--max_expr_length',
        type=int,
        default=32,
        help='最大表达式长度'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=10000,
        help='批处理大小'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='并行进程数'
    )

    args = parser.parse_args()

    # 转换数据集
    convert_feynman_dataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples_per_eq=args.num_samples_per_eq,
        n_points=args.n_points,
        max_dim=args.max_dim,
        max_corruptions=args.max_corruptions,
        max_expr_length=args.max_expr_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()

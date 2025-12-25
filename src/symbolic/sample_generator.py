"""
单个样本生成器模块
包含单个样本生成的核心逻辑
"""

import random
import time
import numpy as np
import sympy as sp
from typing import List, Dict
from src.utils.logger import Logger
from src.symbolic.symbolic_utils import (
    expr_to_tree, generate_random_expr,
    generate_reduction_sequence,
    evaluate_expression_safe,
    levenshtein_alignment_with_gap,
)
from src.symbolic.corruption import corrupt_expression

# 全局变量存储 Logger 实例
_sample_logger = Logger(enabled=True)

def set_logger(logger: Logger):
    """设置 Logger 实例"""
    global _sample_logger
    _sample_logger = logger

def generate_single_sample(
    sample_id: str,
    max_dim: int = 3,
    n_points: int = 100,
    max_depth: int = 4,
    max_expr_length: int = 6,
    batch_idx: int = 0,
    current_batch_size: int = 0,
    current_sample_count: int = 0
) -> List[Dict]:
    """
    生成单个样本的完整数据（可能包含多个删减表达式）

    返回:
        List[Dict]: 生成的样本列表，每个样本对应一个删减表达式
    """
    sample_start_time = time.time()

    try:
        # 随机选择维度
        dim = random.randint(1, max_dim)

        _sample_logger.sample_step(sample_id, "开始生成样本",
                                   f"批次{batch_idx+1}, 维度{dim}, 样本数{current_sample_count}/{current_batch_size}",
                                   info_only=True)

        # 生成数据点
        _sample_logger.sample_step(sample_id, "生成数据点", f"{n_points}个点, {dim}维", info_only=True)
        x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, dim))
        x_values = [list(point) for point in x_values_raw]
        x_array = np.array(x_values)

        # 生成目标表达式
        _sample_logger.sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}", info_only=True)
        expr_gen_start = time.time()
        target_expr = generate_random_expr(dim, max_depth)
        expr_gen_time = (time.time() - expr_gen_start) * 1000
        expr_str = str(target_expr)
        _sample_logger.expression_generate(sample_id, expr_str, expr_gen_time)

        # 转换为表达式树
        _sample_logger.sample_step(sample_id, "转换为表达式树", info_only=True)
        tree_convert_start = time.time()
        expr_tree = expr_to_tree(target_expr)
        tree_convert_time = (time.time() - tree_convert_start) * 1000
        expr_tokens = expr_tree.split(',')
        _sample_logger.expression_convert(sample_id, len(expr_tokens), tree_convert_time)

        # 验证表达式是否符合要求
        _sample_logger.sample_step(sample_id, "验证表达式", info_only=True)
        if len(expr_tokens) <= 1:
            _sample_logger.sample_failed(sample_id, f"表达式token太少: {len(expr_tokens)}")
            return []

        if len(expr_str) > max_expr_length:
            _sample_logger.sample_failed(sample_id, f"表达式过长: {len(expr_str)}")
            return []

        if target_expr.has(sp.I) or 'I' in expr_str:
            _sample_logger.sample_failed(sample_id, "表达式包含复数")
            return []

        _sample_logger.expression_validation(sample_id, expr_str, len(expr_str), len(expr_tokens))

        # 尝试计算目标值
        _sample_logger.sample_step(sample_id, "计算目标表达式值", info_only=True)
        eval_start = time.time()
        success, y_target = evaluate_expression_safe(
            target_expr, x_array,
            error_callback=lambda err: _sample_logger.expression_eval(sample_id, expr_str, (time.time() - eval_start) * 1000, False, err)
        )
        eval_time = (time.time() - eval_start) * 1000
        if success:
            _sample_logger.expression_eval(sample_id, expr_str, eval_time, True)
        else:
            # 计算失败，返回空列表
            _sample_logger.sample_failed(sample_id, "目标表达式计算失败")
            return []

        # 生成删减序列
        _sample_logger.sample_step(sample_id, "生成删减序列", info_only=True)
        reduction_start = time.time()
        reduction_sequence = generate_reduction_sequence(target_expr)
        reduction_time = (time.time() - reduction_start) * 1000
        _sample_logger.reduction_sequence(sample_id, len(reduction_sequence), reduction_time)

        # 为删减序列中的每个表达式创建样本
        batch_samples = []

        for i, reduced_expr in enumerate(reduction_sequence):
            reduced_expr_str = str(reduced_expr)
            _sample_logger.sample_step(sample_id, f"处理删减表达式 {i+1}/{len(reduction_sequence)}", f"表达式: {reduced_expr_str}")

            # 对删减后的表达式应用额外的随机破坏
            _sample_logger.sample_step(sample_id, f"表达式破坏 {i+1}")
            corruption_start = time.time()
            curr_expr = corrupt_expression(reduced_expr)
            corruption_time = (time.time() - corruption_start) * 1000
            curr_expr_str = str(curr_expr)
            _sample_logger.corrupt_expression(sample_id, i+1, reduced_expr_str, curr_expr_str, corruption_time)

            # 检查删减后的表达式是否与目标表达式相同，如果相同则无学习意义
            _sample_logger.sample_step(sample_id, f"检查表达式相同性 {i+1}")
            if curr_expr == target_expr:
                _sample_logger.sample_step(sample_id, f"跳过相同的删减表达式 {i+1}", f"破坏后表达式与目标表达式相同")
                _sample_logger.skip_duplicate(sample_id, i+1)
                continue

            # 检查删减后的表达式是否包含复数单位
            _sample_logger.sample_step(sample_id, f"检查复数 {i+1}")
            if curr_expr.has(sp.I) or 'I' in curr_expr_str:
                _sample_logger.sample_step(sample_id, f"跳过复数删减表达式 {i+1}", f"表达式包含复数单位")
                _sample_logger.skip_complex(sample_id, i+1, curr_expr_str)
                continue

            # 尝试计算当前值
            _sample_logger.sample_step(sample_id, f"计算当前表达式值 {i+1}")
            eval_curr_start = time.time()
            success, y_curr = evaluate_expression_safe(
                curr_expr, x_array,
                error_callback=lambda err: _sample_logger.sample_step(sample_id, f"跳过计算失败的删减表达式 {i+1}", f"计算错误: {err}")
            )
            eval_curr_time = (time.time() - eval_curr_start) * 1000
            _sample_logger.eval_curr_expression(sample_id, i+1, success, eval_curr_time, curr_expr_str)
            if not success:
                continue

            # 转换为token序列
            _sample_logger.sample_step(sample_id, f"转换为token序列 {i+1}")
            tree_start = time.time()
            target_tree = expr_to_tree(target_expr)
            curr_tree = expr_to_tree(curr_expr)
            target_tokens = target_tree.split(',')
            curr_tokens = curr_tree.split(',')
            tree_time = (time.time() - tree_start) * 1000
            _sample_logger.convert_to_trees(sample_id, i+1, len(target_tokens), len(curr_tokens), tree_time)

            # 对齐到Z空间，包含gap token
            _sample_logger.sample_step(sample_id, f"对齐到Z空间 {i+1}")
            align_start = time.time()
            z0_tokens, z1_tokens = levenshtein_alignment_with_gap(curr_tokens, target_tokens)
            align_time = (time.time() - align_start) * 1000
            _sample_logger.levenshtein_alignment(sample_id, i+1, len(z0_tokens), len(z1_tokens), align_time)

            # 计算 residuals 并进行数值裁剪，防止爆炸
            residuals = y_target - y_curr
            # 记录原始 residuals 统计
            _sample_logger.residuals_before_clip(sample_id, i+1, residuals.min(), residuals.max(), residuals.mean())

            # 数值裁剪：防止训练时的梯度爆炸
            # 设置合理的阈值，超过阈值的值会被裁剪
            CLIP_THRESHOLD = 1e6  # 100万
            residuals_clipped = np.clip(residuals, -CLIP_THRESHOLD, CLIP_THRESHOLD)

            # 检查是否有值被裁剪，如果有则跳过样本
            if np.any(residuals != residuals_clipped):
                clip_count = np.sum(residuals != residuals_clipped)
                _sample_logger.skip_clipped(sample_id, i+1, clip_count, len(residuals), CLIP_THRESHOLD)
                continue

            residuals = residuals_clipped

            batch_samples.append({
                "input_dimension": dim,
                "x_values": x_values,  # 保持与原代码一致的字段名
                "y_target": y_target.tolist(),
                "y_curr": y_curr.tolist(),
                "residuals": residuals.tolist(),  # 使用裁剪后的 residuals
                "tree_gt": target_tree,
                "exp_gt": str(target_expr),
                "tree_cur1": curr_tree,
                "exp_cur1": str(curr_expr),
                "curr_tokens": curr_tokens,
                "target_tokens": target_tokens,
                "z0_tokens": z0_tokens,
                "z1_tokens": z1_tokens
            })

        _sample_logger.sample_success(sample_id)
        return batch_samples

    except Exception as e:
        duration = time.time() - sample_start_time
        _sample_logger.sample_error(sample_id, "sample_generation", f"{type(e).__name__}: {e}", duration)
        return []
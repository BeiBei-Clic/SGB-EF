"""
单个样本生成器模块
包含单个样本生成的核心逻辑
"""

import random
import time
import datetime
import numpy as np
import sympy as sp
from typing import List, Dict
from src.utils.log_utils import log_expression_eval
from src.symbolic.symbolic_utils import (
    expr_to_tree, generate_random_expr,
    generate_reduction_sequence,
    evaluate_expression_safe,
    levenshtein_alignment_with_gap,
)
from src.symbolic.corruption import corrupt_expression

# 全局变量存储日志写入函数
_write_log = None

def set_log_writer(log_writer_func):
    """设置日志写入函数"""
    global _write_log
    _write_log = log_writer_func

def log_sample_step(sample_id: str, step: str, details: str = ""):
    """记录样本生成步骤"""
    if _write_log:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        message = f"[{sample_id}] {step}"
        if details:
            message += f" | {details}"
        _write_log(f"{timestamp} {message}")

def generate_single_sample(
    sample_id: str,
    max_dim: int = 3,
    n_points: int = 100,
    max_depth: int = 4,
    max_expr_length: int = 6,
    max_retries: int = 10,
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

        log_sample_step(sample_id, "开始生成样本",
                       f"批次{batch_idx+1}, 维度{dim}, 样本数{current_sample_count}/{current_batch_size}")

        # 生成数据点
        log_sample_step(sample_id, "生成数据点", f"{n_points}个点, {dim}维")
        x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, dim))
        x_values = [list(point) for point in x_values_raw]
        x_array = np.array(x_values)

        # 生成目标表达式
        log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
        expr_gen_start = time.time()
        target_expr = generate_random_expr(dim, max_depth)
        expr_gen_time = (time.time() - expr_gen_start) * 1000
        expr_str = str(target_expr)
        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] GENERATE_RANDOM_EXPR '{expr_str}' | time={expr_gen_time:.1f}ms")

        # 转换为表达式树
        log_sample_step(sample_id, "转换为表达式树")
        tree_convert_start = time.time()
        expr_tree = expr_to_tree(target_expr)
        tree_convert_time = (time.time() - tree_convert_start) * 1000
        expr_tokens = expr_tree.split(',')
        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EXPR_TO_TREE tokens={len(expr_tokens)} | time={tree_convert_time:.1f}ms")

        # 验证表达式是否符合要求
        log_sample_step(sample_id, "验证表达式")
        if len(expr_tokens) <= 1:
            log_sample_step(sample_id, "样本生成失败", f"表达式token太少: {len(expr_tokens)}")
            return []

        if len(expr_str) > max_expr_length:
            log_sample_step(sample_id, "样本生成失败", f"表达式过长: {len(expr_str)}")
            return []

        if target_expr.has(sp.I) or 'I' in expr_str:
            log_sample_step(sample_id, "样本生成失败", "表达式包含复数")
            return []

        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EXPR_VALIDATION_PASSED '{expr_str}' len={len(expr_str)} tokens={len(expr_tokens)}")

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
        else:
            # 计算失败，返回空列表
            log_sample_step(sample_id, "样本生成失败", "目标表达式计算失败")
            return []

        # 生成删减序列
        log_sample_step(sample_id, "生成删减序列")
        reduction_start = time.time()
        reduction_sequence = generate_reduction_sequence(target_expr)
        reduction_time = (time.time() - reduction_start) * 1000
        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] REDUCE_SEQUENCE {len(reduction_sequence)} steps | time={reduction_time:.1f}ms")

        # 为删减序列中的每个表达式创建样本
        batch_samples = []

        for i, reduced_expr in enumerate(reduction_sequence):
            reduced_expr_str = str(reduced_expr)
            log_sample_step(sample_id, f"处理删减表达式 {i+1}/{len(reduction_sequence)}", f"表达式: {reduced_expr_str}")

            # 对删减后的表达式应用额外的随机破坏
            log_sample_step(sample_id, f"表达式破坏 {i+1}")
            corruption_start = time.time()
            curr_expr = corrupt_expression(reduced_expr)
            corruption_time = (time.time() - corruption_start) * 1000
            curr_expr_str = str(curr_expr)
            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] CORRUPT_EXPRESSION step{i+1} | time={corruption_time:.1f}ms")
            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] CORRUPT_RESULT step{i+1} '{reduced_expr_str}' → '{curr_expr_str}'")

            # 检查删减后的表达式是否与目标表达式相同，如果相同则无学习意义
            log_sample_step(sample_id, f"检查表达式相同性 {i+1}")
            if curr_expr == target_expr:
                log_sample_step(sample_id, f"跳过相同的删减表达式 {i+1}", f"破坏后表达式与目标表达式相同")
                _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] SKIP_DUPLICATE step{i+1}")
                continue

            # 检查删减后的表达式是否包含复数单位
            log_sample_step(sample_id, f"检查复数 {i+1}")
            if curr_expr.has(sp.I) or 'I' in curr_expr_str:
                log_sample_step(sample_id, f"跳过复数删减表达式 {i+1}", f"表达式包含复数单位")
                _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] SKIP_COMPLEX step{i+1} expr='{curr_expr_str}'")
                continue

            # 尝试计算当前值
            log_sample_step(sample_id, f"计算当前表达式值 {i+1}")
            eval_curr_start = time.time()
            success, y_curr = evaluate_expression_safe(
                curr_expr, x_array,
                error_callback=lambda err: log_sample_step(sample_id, f"跳过计算失败的删减表达式 {i+1}", f"计算错误: {err}")
            )
            eval_curr_time = (time.time() - eval_curr_start) * 1000
            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EVAL_CURR_EXPRESSION step{i+1} success={success} | time={eval_curr_time:.1f}ms expr='{curr_expr_str}'")
            if not success:
                continue

            # 转换为token序列
            log_sample_step(sample_id, f"转换为token序列 {i+1}")
            tree_start = time.time()
            target_tree = expr_to_tree(target_expr)
            curr_tree = expr_to_tree(curr_expr)
            target_tokens = target_tree.split(',')
            curr_tokens = curr_tree.split(',')
            tree_time = (time.time() - tree_start) * 1000
            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] CONVERT_TO_TREES step{i+1} | time={tree_time:.1f}ms target_tokens={len(target_tokens)} curr_tokens={len(curr_tokens)}")

            # 对齐到Z空间，包含gap token
            log_sample_step(sample_id, f"对齐到Z空间 {i+1}")
            align_start = time.time()
            z0_tokens, z1_tokens = levenshtein_alignment_with_gap(curr_tokens, target_tokens)
            align_time = (time.time() - align_start) * 1000
            _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] LEVENSHTEIN_ALIGNMENT step{i+1} | time={align_time:.1f}ms z0_len={len(z0_tokens)} z1_len={len(z1_tokens)}")

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

        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] SUCCESS")
        return batch_samples

    except Exception as e:
        duration = time.time() - sample_start_time
        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] ERROR sample_generation: {type(e).__name__}: {e}")
        _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] STUCK {duration:.1f}s")
        print(f"警告: 生成样本时出错，跳过该样本: {e}")
        return []
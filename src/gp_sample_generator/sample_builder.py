"""
构建GP样本的核心逻辑
"""
import numpy as np
import sympy as sp
from pathlib import Path
from src.symbolic.symbolic_utils import (
    expr_to_tree,
    evaluate_expression_safe,
    randomized_alignment_with_gap,
)
from src.utils.special_tokens import SymbolicRegressionTokenizer
from src.pysr_solver.data_loader import load_equation_data


def parse_gp_expression(expr_str: str) -> sp.Expr:
    """解析GP表达式字符串为SymPy表达式"""
    # 处理 ^ 符号 (PySR使用 ^ 表示幂运算)
    expr_str = expr_str.replace('^', '**')
    return sp.sympify(expr_str)


def build_single_sample(
    curr_expr_str: str,
    target_expr_str: str,
    x_data: np.ndarray,
    y_true: np.ndarray,
    tokenizer: SymbolicRegressionTokenizer,
    max_expr_length: int = 24
) -> dict:
    """
    构建单个样本

    数据流:
    1. 解析表达式字符串 -> SymPy表达式
    2. 计算表达式值 (y_curr, y_target)
    3. 转换为树结构
    4. 生成token对齐
    5. 构造样本字典
    """
    # 解析表达式
    curr_expr = parse_gp_expression(curr_expr_str)
    target_expr = parse_gp_expression(target_expr_str)

    # 计算值
    _, y_curr = evaluate_expression_safe(curr_expr, x_data)
    _, y_target = evaluate_expression_safe(target_expr, x_data)

    # 计算残差
    residuals = y_target - y_curr

    # 转换为树
    curr_tree = expr_to_tree(curr_expr)
    target_tree = expr_to_tree(target_expr)
    curr_tokens = curr_tree.split(',')
    target_tokens = target_tree.split(',')

    # 对齐
    z0_tokens, z1_tokens = randomized_alignment_with_gap(curr_tokens, target_tokens)

    # Token IDs
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

    def pad_sequence(token_ids):
        tokens = [bos_token_id] + token_ids[:max_expr_length-1]
        tokens.extend([pad_token_id] * (max_expr_length - len(tokens)))
        return tokens

    return {
        "input_dimension": x_data.shape[1],
        "x_values": x_data.tolist(),
        "y_target": y_target.tolist(),
        "y_curr": y_curr.tolist(),
        "residuals": residuals.tolist(),
        "tree_gt": target_tree,
        "exp_gt": target_expr_str,
        "tree_cur1": curr_tree,
        "exp_cur1": curr_expr_str,
        "curr_tokens": curr_tokens,
        "target_tokens": target_tokens,
        "z0_tokens": z0_tokens,
        "z1_tokens": z1_tokens,
        "z0_token_ids": pad_sequence(tokenizer.convert_tokens_to_ids(z0_tokens)),
        "z1_token_ids": pad_sequence(tokenizer.convert_tokens_to_ids(z1_tokens)),
    }

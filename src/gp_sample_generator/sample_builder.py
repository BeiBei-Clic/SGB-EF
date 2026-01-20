"""
构建GP样本的核心逻辑
"""
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from pathlib import Path
from src.symbolic.symbolic_utils import randomized_alignment_with_gap
from src.utils.special_tokens import SymbolicRegressionTokenizer
from src.pysr_solver.data_loader import load_equation_data


def expr_to_tree(expr: sp.Expr) -> str:
    """将SymPy表达式转换为前序遍历字符串"""
    if expr.is_Symbol:
        return str(expr)
    elif expr.is_Number:
        return "constant"
    elif expr == sp.pi:
        return "pi"

    if not hasattr(expr, 'args') or not expr.args:
        return str(expr)

    func_name = str(expr.func).lower()
    args = expr.args

    if 'add' in func_name:
        if len(args) == 0:
            return "0"
        elif len(args) == 1:
            return expr_to_tree(args[0])
        else:
            result = expr_to_tree(args[-1])
            for arg in reversed(args[:-1]):
                result = f'add,{expr_to_tree(arg)},{result}'
            return result
    elif 'sub' in func_name:
        return f'sub,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "0"}'
    elif 'mul' in func_name:
        if len(args) == 0:
            return "1"
        elif len(args) == 1:
            return expr_to_tree(args[0])
        else:
            result = expr_to_tree(args[-1])
            for arg in reversed(args[:-1]):
                result = f'mul,{expr_to_tree(arg)},{result}'
            return result
    elif 'div' in func_name or 'truediv' in func_name:
        return f'div,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "1"}'
    elif 'pow' in func_name:
        return f'pow,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "1"}'
    elif 'asin' in func_name:
        return f'arcsin,{expr_to_tree(args[0])}'
    elif 'sin' in func_name:
        return f'sin,{expr_to_tree(args[0])}'
    elif 'cos' in func_name:
        return f'cos,{expr_to_tree(args[0])}'
    elif 'tanh' in func_name:
        return f'tanh,{expr_to_tree(args[0])}'
    elif 'tan' in func_name:
        return f'tan,{expr_to_tree(args[0])}'
    elif 'exp' in func_name:
        return f'exp,{expr_to_tree(args[0])}'
    elif 'log' in func_name:
        return f'ln,{expr_to_tree(args[0])}'
    elif 'sqrt' in func_name:
        return f'sqrt,{expr_to_tree(args[0])}'
    elif 'abs' in func_name:
        return f'abs,{expr_to_tree(args[0])}'
    else:
        return str(expr)


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
    curr_expr = parse_expr(curr_expr_str.replace('^', '**'))
    target_expr = parse_expr(target_expr_str.replace('^', '**'))

    # 计算值
    def evaluate_expr(expr, x_vals):
        symbols = [sp.Symbol(f'x{i}') for i in range(x_vals.shape[1])]
        f = sp.lambdify(symbols, expr, 'numpy')
        return f(*[x_vals[:, i] for i in range(len(symbols))])

    y_curr = evaluate_expr(curr_expr, x_data)
    y_target = evaluate_expr(target_expr, x_data)

    # 计算残差
    residuals = y_target - y_curr

    # 转换为树
    curr_tree = expr_to_tree(curr_expr)
    target_tree = expr_to_tree(target_expr)
    curr_tokens = curr_tree.split(',')
    target_tokens = target_tree.split(',')

    # 长度检查
    if len(curr_tokens) > max_expr_length or len(target_tokens) > max_expr_length:
        raise ValueError(f"表达式长度超过限制: curr={len(curr_tokens)}, target={len(target_tokens)}, max={max_expr_length}")

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

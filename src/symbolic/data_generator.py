"""
符号回归数据生成器，用于EditFlow预训练
"""

import numpy as np
import random
import sympy as sp
import os
import warnings
import time
from typing import List, Dict, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)

UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']

def timed_simplify(expr: sp.Expr, max_time: float = 1) -> sp.Expr:
    """带时间限制的化简函数"""
    start_time = time.time()
    simplified = expr

    if time.time() - start_time < max_time:
        simplified = sp.together(simplified)
    if time.time() - start_time < max_time:
        simplified = sp.radsimp(simplified)

    return simplified

def expr_to_tree(expr: sp.Expr) -> str:
    """将表达式转换为前序遍历字符串"""
    if expr.is_Symbol:
        return str(expr)
    elif expr.is_Number:
        val = float(expr)
        return str(int(round(val))) if abs(val - round(val)) < 1e-6 else str(round(val, 6))

    if not hasattr(expr, 'args') or not expr.args:
        return str(expr)

    func_name = str(expr.func).lower()
    args = expr.args

    if 'add' in func_name:
        return f'add,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) > 1 else "0"}'
    elif 'sub' in func_name:
        return f'sub,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) > 1 else "0"}'
    elif 'mul' in func_name:
        return f'mul,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) > 1 else "1"}'
    elif 'div' in func_name or 'truediv' in func_name:
        return f'div,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) > 1 else "1"}'
    elif 'pow' in func_name:
        return f'pow,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) > 1 else "1"}'
    elif 'sin' in func_name:
        return f'sin,{expr_to_tree(args[0])}'
    elif 'cos' in func_name:
        return f'cos,{expr_to_tree(args[0])}'
    elif 'tan' in func_name:
        return f'tan,{expr_to_tree(args[0])}'
    elif 'exp' in func_name:
        return f'exp,{expr_to_tree(args[0])}'
    elif 'log' in func_name:
        return f'log,{expr_to_tree(args[0])}'
    elif 'sqrt' in func_name:
        return f'sqrt,{expr_to_tree(args[0])}'
    else:
        return str(expr)

def generate_random_expr(input_dimension: int, max_depth: int = 4) -> sp.Expr:
    """生成随机表达式"""
    symbols = [sp.Symbol(f'x{i+1}') for i in range(input_dimension)]

    def generate_expr(depth: int) -> sp.Expr:
        if depth >= max_depth:
            return random.choice(symbols + [sp.Rational(random.randint(-5, 5))])

        if random.random() < 0.7:  # 70%概率创建操作节点
            if random.choice(['unary', 'binary']) == 'unary':
                op = random.choice(UNARY_OPS)
                return apply_unary_op(op, generate_expr(depth + 1))
            else:
                op = random.choice(BINARY_OPS)
                return apply_binary_op(op, generate_expr(depth + 1), generate_expr(depth + 1))
        else:
            return random.choice(symbols + [sp.Rational(random.randint(-5, 5))])

    # 使用带时间限制的化简
    return timed_simplify(generate_expr(0), max_time=1)

def apply_unary_op(op: str, operand: sp.Expr) -> sp.Expr:
    """应用一元运算符"""
    if op == 'sin': return sp.sin(operand)
    elif op == 'cos': return sp.cos(operand)
    elif op == 'tan': return sp.tan(operand)
    elif op == 'exp': return sp.exp(operand) / (1 + sp.exp(operand-20))
    elif op == 'log': return sp.log(abs(operand) + sp.Rational(1, 1000))
    elif op == 'sqrt': return sp.sqrt(abs(operand))
    return operand

def apply_binary_op(op: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """应用二元运算符"""
    if op == 'add': return left + right
    elif op == 'sub': return left - right
    elif op == 'mul': return left * right
    elif op == 'div': return left / (right + sp.Rational(1, 100))
    elif op == 'pow': return left ** right
    return left + right

def corrupt_expression(expr: sp.Expr, corruption_prob: float = 0.5) -> sp.Expr:
    """对表达式应用随机破坏"""
    if random.random() >= corruption_prob:
        return expr

    corruption_type = random.choice(['simplify', 'replace_constant', 'mutate_operator'])

    if corruption_type == 'simplify':
        scaled_expr = expr * random.uniform(0.5, 2.0)
        return timed_simplify(scaled_expr, max_time=1)
    elif corruption_type == 'replace_constant' and expr.is_Number:
        return sp.Rational(random.randint(-5, 5))
    elif corruption_type == 'mutate_operator' and hasattr(expr, 'args') and len(expr.args) >= 2:
        func_name = str(expr.func).lower()
        if 'add' in func_name:
            return expr.args[0] - expr.args[1]
        elif 'mul' in func_name:
            return expr.args[0] / (expr.args[1] + sp.Rational(1, 100))

    return expr

def evaluate_expr(expr: sp.Expr, x_values: np.ndarray) -> np.ndarray:
    """在给定x值上计算表达式"""
    n_dims = x_values.shape[1] if x_values.ndim > 1 else 1
    variables = [sp.Symbol(f'x{i+1}') for i in range(n_dims)]
    expr_vars = [var for var in variables if var in expr.free_symbols]

    if not expr_vars:
        return np.full(x_values.shape[0] if x_values.ndim > 1 else len(x_values), float(expr))

    f = sp.lambdify(expr_vars, expr, 'numpy')

    if x_values.ndim > 1:
        result = f(*[x_values[:, i] for i in range(len(expr_vars))])
    else:
        result = f(x_values)

    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(result, -1e6, 1e6)

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4) -> Dict:
    """生成单个样本"""
    x_values = np.linspace(-5.0, 5.0, n_points).reshape(-1, 1) if input_dimension == 1 else np.random.uniform(-5.0, 5.0, (n_points, input_dimension))

    target_expr = generate_random_expr(input_dimension, max_depth)
    y_values = evaluate_expr(target_expr, x_values)
    curr_expr = corrupt_expression(target_expr, 0.5)

    return {
        "input_dimension": input_dimension,
        "x": x_values.tolist(),
        "y": y_values.tolist(),
        "tree_gt": expr_to_tree(target_expr),
        "exp_gt": str(target_expr),
        "tree_cur1": expr_to_tree(curr_expr),
        "exp_cur1": str(curr_expr)
    }

def generate_samples(num_samples: int, max_dim: int = 5, n_points: int = 100, max_depth: int = 4) -> List[Dict]:
    """生成多样维度的样本"""
    samples = []
    dimension_count = {}

    for i in range(num_samples):
        dim = random.randint(1, max_dim)
        dimension_count[dim] = dimension_count.get(dim, 0) + 1
        samples.append(generate_sample(dim, n_points, max_depth))

        if (i + 1) % 10 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本")

    print(f"\n样本维度分布:")
    for dim, count in sorted(dimension_count.items()):
        print(f"{dim}维: {count} 个样本")

    return samples

def generate_triplet_samples(num_samples: int, max_dim: int = 5, n_points: int = 100, max_depth: int = 4) -> List[Dict]:
    """生成三元组样本 (E_curr, E_target, r, z) 用于EditFlow预训练"""
    samples = []
    dimension_count = {}

    print(f"生成 {num_samples} 个三元组样本用于EditFlow预训练...")

    for i in tqdm(range(num_samples), desc="生成三元组数据"):
        dim = random.randint(1, max_dim)
        dimension_count[dim] = dimension_count.get(dim, 0) + 1

        x_values = np.linspace(-5.0, 5.0, n_points).reshape(-1, 1) if dim == 1 else np.random.uniform(-5.0, 5.0, (n_points, dim))

        target_expr = generate_random_expr(dim, max_depth)
        y_target = evaluate_expr(target_expr, x_values)
        curr_expr = corrupt_expression(target_expr, corruption_prob=0.7)
        y_curr = evaluate_expr(curr_expr, x_values)

        samples.append({
            "input_dimension": dim,
            "x_values": x_values.tolist(),
            "y_target": y_target.tolist(),
            "y_curr": y_curr.tolist(),
            "residuals": (y_target - y_curr).tolist(),
            "tree_gt": expr_to_tree(target_expr),
            "exp_gt": str(target_expr),
            "tree_cur1": expr_to_tree(curr_expr),
            "exp_cur1": str(curr_expr),
            "alignment_vector": compute_expression_alignment(curr_expr, target_expr)
        })

    print(f"\n样本维度分布:")
    for dim, count in sorted(dimension_count.items()):
        print(f"{dim}维: {count} 个样本")

    return samples


def compute_expression_alignment(curr_expr: sp.Expr, target_expr: sp.Expr) -> Dict:
    """计算两个表达式之间的对齐路径"""
    curr_tokens = expr_to_tree(curr_expr).split(',')
    target_tokens = expr_to_tree(target_expr).split(',')
    alignment = levenshtein_alignment(curr_tokens, target_tokens)

    return {
        'curr_tokens': curr_tokens,
        'target_tokens': target_tokens,
        'alignment': alignment,
        'edit_distance': len([op for op, _, _ in alignment if op != 'keep'])
    }

def levenshtein_alignment(tokens1: List[str], tokens2: List[str]) -> List[Tuple[str, str, str]]:
    """计算两个token序列的Levenshtein对齐路径"""
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and tokens1[i-1] == tokens2[j-1]:
            alignment.append(('keep', tokens1[i-1], tokens2[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append(('substitute', tokens1[i-1], tokens2[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append(('delete', tokens1[i-1], ''))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append(('insert', '', tokens2[j-1]))
            j -= 1

    return list(reversed(alignment))


def save_to_txt(samples: List[Dict], filename: str):
    """将样本保存到txt文件"""
    os.makedirs("/home/xyh/SGB-EF/data", exist_ok=True)
    filepath = os.path.join("/home/xyh/SGB-EF/data", filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            line = f'{{x:{sample["x"]},y:{sample["y"]},tree_gt:"{sample["tree_gt"]}",exp_gt:"{sample["exp_gt"]}",tree_cur1:"{sample["tree_cur1"]}",exp_cur1:"{sample["exp_cur1"]}"}}\n'
            f.write(line)

    print(f"已保存 {len(samples)} 个样本到 {filepath}")

def save_triplets_to_txt(samples: List[Dict], filename: str):
    """将三元组样本保存到txt文件"""
    os.makedirs("/home/xyh/SGB-EF/data", exist_ok=True)
    filepath = os.path.join("/home/xyh/SGB-EF/data", filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            line = f'{{"x_values":{sample["x_values"]},"residuals":{sample["residuals"]},"curr_tree":"{sample["tree_cur1"]}","target_tree":"{sample["tree_gt"]}","alignment":{sample["alignment_vector"]}}}\n'
            f.write(line)

    print(f"已保存 {len(samples)} 个三元组样本到 {filepath}")


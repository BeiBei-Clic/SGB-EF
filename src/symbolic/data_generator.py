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
from typing import List, Dict, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)

UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
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
        # 处理 NaN 值，使用 0 作为默认值（与其他代码保持一致）
        if np.isnan(val):
            return "0"
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
    elif 'abs' in func_name:
        return f'abs,{expr_to_tree(args[0])}'  # Abs(x) -> abs,x
    else:
        return str(expr)

def generate_random_expr(input_dimension: int, max_depth: int = 4) -> sp.Expr:
    """生成随机表达式"""
    symbols = [sp.Symbol(f'x{i}') for i in range(input_dimension)]

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
    elif op == 'exp': return sp.exp(operand)
    elif op == 'log': return sp.log(abs(operand) + sp.Rational(1, 1000))
    elif op == 'sqrt': return sp.sqrt(abs(operand))
    elif op == 'abs': return abs(operand)
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

def reduce_expression(expr: sp.Expr) -> sp.Expr:
    """对表达式进行一次删减操作，返回简化后的表达式"""
    # 如果表达式已经是常数或符号，无法进一步删减
    if expr.is_Number or expr.is_Symbol:
        return expr

    # 如果表达式没有参数（参数为空），返回常数1
    if not hasattr(expr, 'args') or not expr.args:
        return sp.Integer(1)

    func_name = str(expr.func).lower()
    args = expr.args

    # 处理一元操作
    if func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']:
        if len(args) >= 1:
            # 一元操作删减：直接返回操作数
            return args[0]
        else:
            return sp.Integer(1)

    # 处理二元操作
    elif 'add' in func_name or 'sub' in func_name:
        if len(args) >= 2:
            # 加减法删减：随机选择其中一个操作数
            return random.choice(args)
        else:
            return args[0] if args else sp.Integer(0)

    elif 'mul' in func_name or 'div' in func_name or 'truediv' in func_name:
        if len(args) >= 2:
            # 乘除法删减：随机选择其中一个操作数
            return random.choice(args)
        else:
            return args[0] if args else sp.Integer(1)

    elif 'pow' in func_name:
        if len(args) >= 2:
            # 幂运算删减：随机选择底数或指数
            return random.choice(args)
        else:
            return args[0] if args else sp.Integer(1)

    # 其他情况：随机选择一个参数
    elif args:
        return random.choice(args)
    else:
        return sp.Integer(1)

def generate_reduction_sequence(target_expr: sp.Expr) -> List[sp.Expr]:
    """生成从目标表达式逐步删减的序列，直到得到最简表达式"""
    reduction_sequence = []
    current_expr = target_expr

    # 持续删减直到表达式无法进一步简化
    max_iterations = 20  # 防止无限循环
    iterations = 0

    while iterations < max_iterations:
        reduction_sequence.append(current_expr)

        # 检查是否已经是最简形式（常数或符号）
        if current_expr.is_Number or current_expr.is_Symbol:
            break

        # 应用一次删减操作
        reduced_expr = reduce_expression(current_expr)

        # 如果删减后的表达式与原表达式相同，说明无法进一步删减
        if reduced_expr == current_expr:
            # 强制简化为常数
            reduced_expr = sp.Integer(1)

        current_expr = timed_simplify(reduced_expr, max_time=0.5)
        iterations += 1

    # 确保序列中包含最简形式
    if not reduction_sequence or (reduction_sequence[-1].is_Number or reduction_sequence[-1].is_Symbol):
        if reduction_sequence and reduction_sequence[-1] != current_expr:
            reduction_sequence.append(current_expr)

    return reduction_sequence

def preprocess_expression(expr: sp.Expr) -> sp.Expr:
    """预处理表达式，替换有问题的值"""
    # 替换 zoo (复无穷大，SymPy中ComplexInfinity的实际名称)
    expr = expr.replace(sp.zoo, sp.Integer(1))

    # 替换其他无穷大形式
    expr = expr.replace(sp.oo, sp.Integer(1000000))
    expr = expr.replace(-sp.oo, sp.Integer(-1000000))

    # 处理可能的 NaN 值
    expr = expr.replace(sp.nan, sp.Integer(0))

    return expr

def evaluate_expr(expr: sp.Expr, x_values: np.ndarray) -> np.ndarray:
    """在给定x值上计算表达式"""
    # 预处理表达式，替换 ComplexInfinity 和其他有问题的值
    expr = preprocess_expression(expr)

    n_dims = x_values.shape[1] if x_values.ndim > 1 else 1
    variables = [sp.Symbol(f'x{i}') for i in range(n_dims)]
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
    # 统一生成数据点，确保每个数据点是[x0, x1, x2, ...]的形式
    x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, input_dimension))
    x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式

    # 转换为numpy数组用于表达式计算
    x_array = np.array(x_values)

    target_expr = generate_random_expr(input_dimension, max_depth)
    y_values = evaluate_expr(target_expr, x_array)
    curr_expr = corrupt_expression(target_expr, 0.5)

    return {
        "input_dimension": input_dimension,
        "x": x_values,  # 现在是[[x0, x1, x2], [x3, x4, x5], ...]的形式
        "y": y_values.tolist(),
        "tree_gt": expr_to_tree(target_expr),
        "exp_gt": str(target_expr),
        "tree_cur1": expr_to_tree(curr_expr),
        "exp_cur1": str(curr_expr)
    }

def levenshtein_alignment_with_gap(tokens1: List[str], tokens2: List[str]) -> Tuple[List[str], List[str]]:
    """使用Levenshtein距离计算两个token序列的对齐，返回包含gap token的等长序列"""
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # 匹配，无代价
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  # 插入、删除、替换

    # 回溯构建对齐序列
    z1, z2 = [], []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and tokens1[i-1] == tokens2[j-1]:
            # 匹配
            z1.append(tokens1[i-1])
            z2.append(tokens2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # 替换
            z1.append(tokens1[i-1])
            z2.append(tokens2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # 删除（从tokens1中删除）
            z1.append(tokens1[i-1])
            z2.append("<gap>")  # gap token标记
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # 插入（向tokens1中插入）
            z1.append("<gap>")  # gap token标记
            z2.append(tokens2[j-1])
            j -= 1

    # 反转序列，因为我们是反向构建的
    return list(reversed(z1)), list(reversed(z2))

def get_data_filename(num_samples: int, max_dim: int, n_points: int, max_depth: int) -> str:
    """生成数据文件名"""
    return f"data/flow_samples_{num_samples}_{max_dim}dim_{n_points}pts_{max_depth}depth.txt"

def save_samples_to_txt(samples: List[Dict], filename: str):
    """将样本保存到txt文件，每行一个样本"""
    print(f"保存数据到 {filename}...")

    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            # 将样本转换为JSON格式并写入一行
            sample_line = json.dumps(sample, ensure_ascii=False)
            f.write(sample_line + '\n')

    print(f"已保存 {len(samples)} 个样本到 {filename}")

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

def generate_flow_samples(num_samples: int, max_dim: int = 5, n_points: int = 100, max_depth: int = 4, use_cache: bool = True) -> List[Dict]:
    """生成用于EditFlow连续流训练的样本"""

    # 检查是否存在缓存文件
    filename = get_data_filename(num_samples, max_dim, n_points, max_depth)

    if use_cache and os.path.exists(filename):
        print(f"发现缓存文件 {filename}，直接加载数据...")
        return load_samples_from_txt(filename)

    samples = []
    dimension_count = {}

    print(f"生成 {num_samples} 个连续流训练样本...")

    sample_count = 0
    pbar = tqdm(total=num_samples, desc="生成流数据")

    while sample_count < num_samples:
        dim = random.randint(1, max_dim)
        dimension_count[dim] = dimension_count.get(dim, 0) + 1

        # 生成数据点，统一处理确保每个数据点是[x0, x1, x2, ...]的形式
        x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, dim))
        x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式
        x_array = np.array(x_values)  # 用于表达式计算

        # 生成目标表达式
        target_expr = generate_random_expr(dim, max_depth)
        y_target = evaluate_expr(target_expr, x_array)

        # 生成删减序列
        reduction_sequence = generate_reduction_sequence(target_expr)

        # 为删减序列中的每个表达式创建样本
        for reduced_expr in reduction_sequence:
            if sample_count >= num_samples:
                break

            # 对删减后的表达式应用额外的随机破坏
            curr_expr = corrupt_expression(reduced_expr, corruption_prob=0.3)
            y_curr = evaluate_expr(curr_expr, x_array)

            # 转换为token序列
            target_tokens = expr_to_tree(target_expr).split(',')
            curr_tokens = expr_to_tree(curr_expr).split(',')

            # 对齐到Z空间，包含gap token
            z0_tokens, z1_tokens = levenshtein_alignment_with_gap(curr_tokens, target_tokens)

            samples.append({
                "input_dimension": dim,
                "x_values": x_values,  # 已经是列表格式
                "y_target": y_target.tolist(),
                "y_curr": y_curr.tolist(),
                "residuals": (y_target - y_curr).tolist(),
                "tree_gt": expr_to_tree(target_expr),
                "exp_gt": str(target_expr),
                "tree_cur1": expr_to_tree(curr_expr),
                "exp_cur1": str(curr_expr),
                "curr_tokens": curr_tokens,
                "target_tokens": target_tokens,
                "z0_tokens": z0_tokens,  # 对齐后的当前序列（包含gap）
                "z1_tokens": z1_tokens,  # 对齐后的目标序列（包含gap）
                "edit_distance": len([op for op in range(len(z0_tokens)) if z0_tokens[op] != z1_tokens[op]])
            })

            sample_count += 1
            pbar.update(1)

        # 如果还没有达到目标样本数，继续生成新的目标表达式
        if sample_count >= num_samples:
            break

    pbar.close()

    print(f"\n样本维度分布:")
    for dim, count in sorted(dimension_count.items()):
        print(f"{dim}维: {count} 个样本")

    # 保存到文件
    save_samples_to_txt(samples, filename)

    return samples




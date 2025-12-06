"""
符号回归数据生成器，用于EditFlow预训练
"""

import numpy as np
import random
import sympy as sp
import os
from typing import List, Dict

# 运算符定义
UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']

def expr_to_tree(expr: sp.Expr) -> str:
    """将表达式转换为前序遍历字符串"""
    if expr.is_Symbol:
        return str(expr)
    elif expr.is_Number:
        val = float(expr)
        return str(int(round(val))) if abs(val - round(val)) < 1e-6 else str(round(val, 6))

    func_name = str(expr.func).lower()

    if 'add' in func_name:
        return 'add,' + expr_to_tree(expr.args[0]) + ',' + expr_to_tree(expr.args[1])
    elif 'sub' in func_name:
        return 'sub,' + expr_to_tree(expr.args[0]) + ',' + expr_to_tree(expr.args[1])
    elif 'mul' in func_name:
        return 'mul,' + expr_to_tree(expr.args[0]) + ',' + expr_to_tree(expr.args[1])
    elif 'div' in func_name or 'truediv' in func_name:
        return 'div,' + expr_to_tree(expr.args[0]) + ',' + expr_to_tree(expr.args[1])
    elif 'pow' in func_name:
        return 'pow,' + expr_to_tree(expr.args[0]) + ',' + expr_to_tree(expr.args[1])
    elif 'sin' in func_name:
        return 'sin,' + expr_to_tree(expr.args[0])
    elif 'cos' in func_name:
        return 'cos,' + expr_to_tree(expr.args[0])
    elif 'tan' in func_name:
        return 'tan,' + expr_to_tree(expr.args[0])
    elif 'exp' in func_name:
        return 'exp,' + expr_to_tree(expr.args[0])
    elif 'log' in func_name:
        return 'log,' + expr_to_tree(expr.args[0])
    elif 'sqrt' in func_name:
        return 'sqrt,' + expr_to_tree(expr.args[0])
    else:
        # 对于未识别的表达式，返回其字符串表示
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

    return sp.simplify(generate_expr(0))

def apply_unary_op(op: str, operand: sp.Expr) -> sp.Expr:
    """应用一元运算符"""
    op_map = {
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': lambda x: sp.log(abs(x) + 1e-8),
        'sqrt': lambda x: sp.sqrt(abs(x)),
    }
    return op_map.get(op, lambda x: x)(operand)

def apply_binary_op(op: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """应用二元运算符"""
    op_map = {
        'add': lambda l, r: l + r,
        'sub': lambda l, r: l - r,
        'mul': lambda l, r: l * r,
        'div': lambda l, r: l / (r + sp.Rational(1, 100)),
        'pow': lambda l, r: l ** r
    }
    return op_map.get(op, lambda l, r: l + r)(left, right)

def corrupt_expression(expr: sp.Expr, corruption_prob: float = 0.5) -> sp.Expr:
    """对表达式应用随机破坏"""
    if random.random() < corruption_prob:
        corruption_type = random.choice(['simplify', 'replace_constant', 'mutate_operator'])

        if corruption_type == 'simplify':
            # 简化表达式
            return sp.simplify(expr * random.uniform(0.5, 2.0))

        elif corruption_type == 'replace_constant':
            # 替换常数为随机值
            if expr.is_Number:
                return sp.Rational(random.randint(-5, 5))

        elif corruption_type == 'mutate_operator':
            # 变异运算符（简化版）
            if hasattr(expr, 'args') and len(expr.args) >= 1:
                func_name = str(expr.func).lower()
                if 'add' in func_name and len(expr.args) >= 2:
                    return expr.args[0] - expr.args[1]
                elif 'mul' in func_name and len(expr.args) >= 2:
                    return expr.args[0] / (expr.args[1] + sp.Rational(1, 100))

    return expr

def evaluate_expr(expr: sp.Expr, x_values: np.ndarray) -> np.ndarray:
    """在给定x值上计算表达式"""
    variables = [sp.Symbol(f'x{i+1}') for i in range(x_values.shape[1] if x_values.ndim > 1 else 1)]
    expr_vars = [var for var in variables if var in expr.free_symbols]

    if not expr_vars:
        # 如果表达式没有变量，返回常数
        return np.full(x_values.shape[0] if x_values.ndim > 1 else len(x_values), float(expr))

    f = sp.lambdify(expr_vars, expr, 'numpy')

    if x_values.ndim > 1:
        result = f(*[x_values[:, i] for i in range(len(expr_vars))])
    else:
        result = f(x_values)

    # 处理无穷大和NaN值
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(result, -1e6, 1e6)

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4) -> Dict:
    """生成单个样本"""
    # 生成x值
    if input_dimension == 1:
        x_values = np.linspace(-5.0, 5.0, n_points)
    else:
        x_values = np.random.uniform(-5.0, 5.0, (n_points, input_dimension))

    # 生成目标表达式
    target_expr = generate_random_expr(input_dimension, max_depth)
    y_values = evaluate_expr(target_expr, x_values)

    # 生成破坏的当前表达式
    curr_expr = corrupt_expression(target_expr, 0.5)

    return {
        "input_dimension": input_dimension,
        "x": x_values.tolist() if input_dimension == 1 else x_values.tolist(),
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
        # 随机选择输入维度
        dim = random.randint(1, max_dim)
        dimension_count[dim] = dimension_count.get(dim, 0) + 1

        # 生成样本
        sample = generate_sample(dim, n_points, max_depth)
        samples.append(sample)

        if (i + 1) % 10 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本")

    # 打印维度分布
    print(f"\n样本维度分布:")
    for dim, count in sorted(dimension_count.items()):
        print(f"{dim}维: {count} 个样本")

    return samples

def save_to_txt(samples: List[Dict], filename: str):
    """将样本保存到txt文件"""
    # 确保data目录存在
    data_dir = "/home/xyh/SGB-EF/data"
    os.makedirs(data_dir, exist_ok=True)

    # 完整文件路径
    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            # 转换为要求的格式
            line = f'{{x:{sample["x"]},y:{sample["y"]},tree_gt:"{sample["tree_gt"]}",exp_gt:"{sample["exp_gt"]}",tree_cur1:"{sample["tree_cur1"]}",exp_cur1:"{sample["exp_cur1"]}"}}\n'
            f.write(line)

    print(f"已保存 {len(samples)} 个样本到 {filepath}")

if __name__ == "__main__":
    # 测试数据生成器
    print("开始生成并保存样本...")
    samples = generate_samples(
        num_samples=50,
        max_dim=5,
        n_points=100,
        max_depth=4
    )

    save_to_txt(samples, "symbolic_samples_simplified.txt")
    print(f"成功生成并保存了 {len(samples)} 个样本")
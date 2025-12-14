"""
符号表达式处理和计算工具函数
"""

import random
import time
import numpy as np
import sympy as sp
from typing import List, Tuple
from src.utils.timeout_utils import TimeoutError


# 运算符定义
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
        # 所有数值常数都返回统一的 "constant" token
        return "constant"

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
    """生成随机表达式 - 增加超时保护和详细日志"""
    start_time = time.time()
    symbols = [sp.Symbol(f'x{i}') for i in range(input_dimension)]

    def generate_expr(depth: int) -> sp.Expr:
        # 超时检查
        if time.time() - start_time > 1.5:
            raise TimeoutError("表达式生成超时")

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

    # 生成表达式
    expr = generate_expr(0)

    # 简化表达式（带超时保护）
    try:
        simplified = timed_simplify(expr, max_time=0.5)
        expr = simplified
    except Exception as e:
        pass  # 简化失败就使用原始表达式

    # 检查并移除复数单位I
    if expr.has(sp.I):
        expr = expr.replace(sp.I, sp.Integer(1))
        expr = timed_simplify(expr, max_time=0.5)

    return expr


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


def evaluate_expr(expr: sp.Expr, x_values: np.ndarray) -> np.ndarray:
    """在给定x值上计算表达式

    如果表达式包含有问题的值，将抛出异常而不是修复
    """
    # 检查表达式是否包含有问题的值
    if expr.has(sp.I):
        raise ValueError(f"表达式包含复数单位I: {expr}")
    if expr.has(sp.zoo):
        raise ValueError(f"表达式包含复无穷大(zoo): {expr}")
    if expr.has(sp.oo) or expr.has(-sp.oo):
        raise ValueError(f"表达式包含无穷大: {expr}")
    if expr.has(sp.nan):
        raise ValueError(f"表达式包含NaN: {expr}")

    # 获取表达式中的变量
    n_dims = x_values.shape[1] if x_values.ndim > 1 else 1
    symbols = [sp.Symbol(f'x{i}') for i in range(n_dims)]
    expr_vars = [s for s in symbols if s in expr.free_symbols]

    # 常数表达式直接计算
    if not expr_vars:
        value = float(sp.re(expr)) if expr.is_complex else float(expr)
        return np.full(x_values.shape[0] if x_values.ndim > 1 else len(x_values), value)

    # 转换为可计算函数
    f = sp.lambdify(expr_vars, expr, 'numpy')

    # 计算结果
    if x_values.ndim > 1:
        result = f(*[x_values[:, i] for i in range(len(expr_vars))])
    else:
        result = f(x_values)

    # 检查结果是否为复数
    if np.iscomplexobj(result):
        raise ValueError(f"计算结果包含复数: {expr}")

    # 检查结果中是否包含NaN或无穷大
    if np.any(np.isnan(result)):
        raise ValueError(f"计算结果包含NaN，表达式: {expr}")
    if np.any(np.isinf(result)):
        raise ValueError(f"计算结果包含无穷大，表达式: {expr}")

    return result


def evaluate_expression_safe(expr: sp.Expr, x_values: np.ndarray,
                             error_callback=None) -> Tuple[bool, np.ndarray]:
    """
    安全地计算表达式值，带异常处理

    Args:
        expr: 要计算的表达式
        x_values: x值数组
        error_callback: 可选的回调函数，用于记录错误信息

    Returns:
        (是否成功, 计算结果)
        - 成功: (True, 结果数组)
        - 失败: (False, None)
    """
    from src.utils.timeout_utils import TimeoutError, with_timeout

    try:
        # 为evaluate_expr添加3秒超时保护，防止复杂表达式卡死
        result = with_timeout(evaluate_expr, 3.0, expr, x_values)
        return True, result
    except TimeoutError as e:
        if error_callback:
            error_callback(f"计算超时: {str(e)}")
        return False, None
    except Exception as e:
        if error_callback:
            error_callback(str(e))
        return False, None

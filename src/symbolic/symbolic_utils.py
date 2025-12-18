"""
符号表达式处理和计算工具函数
"""

import random
import time
import numpy as np
import sympy as sp
from typing import List, Tuple
from src.symbolic.corruption import corrupt_expression, replace_variables


# 运算符定义
UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']


def simplify_expr(expr: sp.Expr) -> sp.Expr:
    """化简表达式"""
    simplified = expr
    simplified = sp.together(simplified)
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
        return f'add,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "0"}'
    elif 'sub' in func_name:
        return f'sub,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "0"}'
    elif 'mul' in func_name:
        return f'mul,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "1"}'
    elif 'div' in func_name or 'truediv' in func_name:
        return f'div,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "1"}'
    elif 'pow' in func_name:
        return f'pow,{expr_to_tree(args[0])},{expr_to_tree(args[1]) if len(args) >= 2 else "1"}'
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
    import time
    import datetime
    start_time = time.time()

    def log_debug(step: str, details: str = ""):
        """记录调试日志"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        message = f"[GENERATE_RANDOM_EXPR] {step}"
        if details:
            message += f" | {details}"
        try:
            from src.utils.log_utils import _write_log
            if _write_log:
                _write_log(f"{timestamp} {message}")
        except:
            pass

    log_debug("开始", f"input_dimension={input_dimension}, max_depth={max_depth}")

    # 创建符号
    symbol_start = time.time()
    symbols = [sp.Symbol(f'x{i}') for i in range(input_dimension)]
    symbol_time = (time.time() - symbol_start) * 1000
    log_debug("符号创建完成", f"{symbol_time:.1f}ms, symbols={[str(s) for s in symbols]}")

    # 递归生成表达式的计数器
    recursion_count = 0
    max_recursions = 1000  # 防止无限递归

    def generate_expr(depth: int) -> sp.Expr:
        nonlocal recursion_count
        recursion_count += 1

        if recursion_count > max_recursions:
            raise Exception(f"递归次数过多: {recursion_count} > {max_recursions}")

        if depth >= max_depth:
            result = random.choice(symbols + [sp.Rational(random.randint(-5, 5))])
            log_debug(f"递归底层(depth={depth})", str(result))
            return result

        node_type = "operation" if random.random() < 0.5 else "leaf"
        log_debug(f"递归层(depth={depth})", f"选择{node_type}")

        if node_type == "operation":
            op_type = random.choice(['unary', 'binary'])
            log_debug(f"递归层(depth={depth})", f"选择{op_type}操作")

            if op_type == 'unary':
                op = random.choice(UNARY_OPS)
                log_debug(f"递归层(depth={depth})", f"应用一元操作{op}")
                operand = generate_expr(depth + 1)
                result = apply_unary_op(op, operand)
                log_debug(f"递归层(depth={depth})", f"{op}({operand}) = {result}")
                return result
            else:
                op = random.choice(BINARY_OPS)
                log_debug(f"递归层(depth={depth})", f"应用二元操作{op}")
                left = generate_expr(depth + 1)
                right = generate_expr(depth + 1)
                result = apply_binary_op(op, left, right)
                log_debug(f"递归层(depth={depth})", f"{left} {op} {right} = {result}")
                return result
        else:
            result = random.choice(symbols + [sp.Rational(random.randint(-5, 5))])
            log_debug(f"递归层(depth={depth})", f"叶子节点 {result}")
            return result

    # 生成表达式
    log_debug("开始生成表达式")
    gen_start = time.time()
    expr = generate_expr(0)
    gen_time = (time.time() - gen_start) * 1000
    log_debug("表达式生成完成", f"{gen_time:.1f}ms, 递归次数={recursion_count}, 表达式={expr}")

    # 简化表达式
    log_debug("开始简化表达式")
    simplify_start = time.time()
    try:
        expr = simplify_expr(expr)
        simplify_time = (time.time() - simplify_start) * 1000
        log_debug("表达式简化成功", f"{simplify_time:.1f}ms, 简化后={expr}")
    except Exception as e:
        simplify_time = (time.time() - simplify_start) * 1000
        log_debug("表达式简化失败", f"{simplify_time:.1f}ms, 错误={e}, 使用原始表达式")
        # 简化失败就使用原始表达式

    total_time = (time.time() - start_time) * 1000
    log_debug("generate_random_expr完成", f"总时间={total_time:.1f}ms, 最终表达式={expr}")

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
        # 一元操作删减：直接返回操作数
        return args[0] if args else sp.Integer(1)

    # 处理二元操作
    default_map = {
        'add': 0, 'sub': 0,
        'mul': 1, 'div': 1, 'truediv': 1, 'pow': 1
    }

    for op_key, default in default_map.items():
        if op_key in func_name:
            return random.choice(args) if len(args) >= 2 else (args[0] if args else sp.Integer(default))

    # 其他情况：随机选择一个参数
    return random.choice(args) if args else sp.Integer(1)


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

        current_expr = simplify_expr(reduced_expr)
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
    try:
        result = evaluate_expr(expr, x_values)
        return True, result
    except Exception as e:
        if error_callback:
            error_callback(str(e))
        return False, None

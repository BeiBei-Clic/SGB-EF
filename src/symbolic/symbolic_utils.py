"""
符号表达式处理和计算工具函数
"""

import random
import time
import numpy as np
import sympy as sp
from typing import List, Tuple, Union
from scipy.optimize import minimize
from src.symbolic.corruption import corrupt_expression, replace_variables
from src.utils.logger import Logger

# 创建全局 Logger 实例
_logger = Logger(enabled=True)


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
        _logger.write(f"{timestamp} {message}")

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


def randomized_alignment_with_gap(tokens1: List[str], tokens2: List[str]) -> Tuple[List[str], List[str]]:
    """随机化辅助对齐方法 (来自《Edit Flows》论文)

    通过在DP回溯时随机选择最优路径来解决"gap集中"问题：
    - Levenshtein对齐按照固定优先级回溯，导致gap集中在序列开头
    - 随机化对齐在所有最优路径中随机选择，保持Token相对顺序不变

    核心原理：
    1. 构建标准编辑距离DP矩阵
    2. 回溯时，当存在多个最优方向时随机选择一个
    3. 保证 frm_blanks(z1) == tokens1 和 frm_blanks(z2) == tokens2

    Args:
        tokens1: 源token序列（当前表达式）
        tokens2: 目标token序列（目标表达式）

    Returns:
        Tuple[List[str], List[str]]: 对齐后的两个等长序列，包含<gap>标记

    Example:
        tokens1 = ['add', 'x0', 'x1']
        tokens2 = ['add', 'sin', 'x0', 'cos', 'x1']

        可能的对齐结果（每次运行gap位置不同）:
        z1 = ['add', 'x0', '<gap>', 'x1', '<gap>']
        z2 = ['add', 'sin', 'x0', 'cos', 'x1']

        注意：移除<gap>后，z2始终保持['add', 'sin', 'x0', 'cos', 'x1']
    """
    m, n = len(tokens1), len(tokens2)

    # 步骤1：构建标准编辑距离DP矩阵
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # 匹配，代价不变
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # 步骤2：随机回溯（关键修改！）
    # 在每一步，收集所有最优方向，然后随机选择一个
    z1, z2 = [], []
    i, j = m, n

    while i > 0 or j > 0:
        choices = []
        current_cost = dp[i][j]

        # 检查左上方 (匹配或替换)
        if i > 0 and j > 0:
            cost_diag = dp[i-1][j-1]
            # 如果是匹配，代价不变；如果是替换，代价+1
            expected_cost = cost_diag if tokens1[i-1] == tokens2[j-1] else cost_diag + 1
            if expected_cost == current_cost:
                choices.append(('DIAG', tokens1[i-1], tokens2[j-1]))

        # 检查上方 (删除 tokens1[i-1])
        if i > 0 and dp[i-1][j] + 1 == current_cost:
            choices.append(('UP', tokens1[i-1], '<gap>'))

        # 检查左方 (向 tokens1 插入 tokens2[j-1])
        if j > 0 and dp[i][j-1] + 1 == current_cost:
            choices.append(('LEFT', '<gap>', tokens2[j-1]))

        # 从所有最优选项中随机选一个（论文的随机性所在）
        move, t1, t2 = random.choice(choices)

        z1.append(t1)
        z2.append(t2)

        if move == 'DIAG':
            i -= 1
            j -= 1
        elif move == 'UP':
            i -= 1
        elif move == 'LEFT':
            j -= 1

    # 步骤3：反转序列，因为我们是反向构建的
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


def tree_to_expr(tree_str: str) -> sp.Expr:
    """将前序遍历字符串转换为SymPy表达式"""
    if not tree_str or tree_str.strip() == '':
        return sp.Integer(1)

    # 如果是单纯的符号或常数
    if ',' not in tree_str:
        if tree_str == 'constant':
            return sp.Symbol('c')  # 用符号c表示常数
        elif tree_str.startswith('x') and tree_str[1:].isdigit():
            return sp.Symbol(tree_str)
        else:
            # 尝试解析为数字
            try:
                return sp.Rational(float(tree_str))
            except:
                return sp.Symbol(tree_str)

    # 使用迭代方式解析前序遍历
    tokens = tree_str.split(',')

    def parse_expression(tokens_iter):
        """递归解析表达式"""
        try:
            token = next(tokens_iter)
        except StopIteration:
            return sp.Integer(1)

        if token == 'constant':
            return sp.Symbol('c')
        elif token.startswith('x') and token[1:].isdigit():
            return sp.Symbol(token)
        elif token in UNARY_OPS:
            # 一元操作: op,operand
            operand = parse_expression(tokens_iter)
            return apply_unary_op(token, operand)
        elif token in BINARY_OPS:
            # 二元操作: op,left,right
            left = parse_expression(tokens_iter)
            right = parse_expression(tokens_iter)
            return apply_binary_op(token, left, right)
        else:
            # 尝试解析为数字
            try:
                return sp.Rational(float(token))
            except:
                return sp.Symbol(token)

    # 创建迭代器并解析
    return parse_expression(iter(tokens))




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
        # 检查结果是否包含nan或inf
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            if error_callback:
                error_callback("Result contains NaN or Inf values")
            return False, None
        # 检查结果是否包含超大数值（绝对值超过阈值）
        MAX_ABS_VALUE = 1e4  # 最大绝对值阈值
        max_abs = np.max(np.abs(result))
        if max_abs > MAX_ABS_VALUE:
            if error_callback:
                error_callback(f"Result contains values exceeding threshold: max_abs={max_abs:.2f} > {MAX_ABS_VALUE}")
            return False, None
        return True, result
    except Exception as e:
        if error_callback:
            error_callback(str(e))
        return False, None


def extract_constants(expr: sp.Expr) -> List[sp.Symbol]:
    """提取表达式中的常数符号"""
    return [s for s in expr.free_symbols if s.name == 'c']


def optimize_constants(expr: sp.Expr, x_values: np.ndarray, y_values: np.ndarray) -> Tuple[sp.Expr, float, bool]:
    """
    使用BFGS优化表达式中的常数

    Args:
        expr: 包含常数符号的表达式
        x_values: 输入数据
        y_values: 目标值

    Returns:
        (优化后的表达式, 最优损失, 是否成功)
    """
    constants = extract_constants(expr)

    if not constants:
        # 没有常数，直接计算当前表达式的MSE
        try:
            n_dims = x_values.shape[1] if x_values.ndim > 1 else 1
            symbols = [sp.Symbol(f'x{i}') for i in range(n_dims)]

            # 创建计算函数
            f = sp.lambdify(symbols, expr, 'numpy')

            # 计算预测值
            if x_values.ndim > 1:
                y_pred = f(*[x_values[:, i] for i in range(n_dims)])
            else:
                y_pred = f(x_values)

            # 计算MSE
            mse = float(np.mean((y_pred - y_values) ** 2))
            return expr, mse, True
        except Exception as e:
            return expr, float('inf'), False

    # 获取表达式中的变量
    n_dims = x_values.shape[1] if x_values.ndim > 1 else 1
    symbols = [sp.Symbol(f'x{i}') for i in range(n_dims)]

    # 构建数值计算函数
    numerical_expr = expr
    for c in constants:
        numerical_expr = numerical_expr.subs(c, sp.Symbol(f'const_{constants.index(c)}'))

    # 创建lambdify函数
    all_symbols = symbols + [sp.Symbol(f'const_{i}') for i in range(len(constants))]
    f = sp.lambdify(all_symbols, numerical_expr, 'numpy')

    def objective(const_values):
        """目标函数：计算均方误差"""
        try:
            if x_values.ndim > 1:
                x_args = [x_values[:, i] for i in range(n_dims)]
            else:
                x_args = [x_values]

            # 将常数作为额外参数传入
            args = x_args + list(const_values)
            y_pred = f(*args)

            # 计算MSE
            mse = np.mean((y_pred - y_values) ** 2)
            return mse
        except:
            # 如果计算失败，返回很大的值
            return 1e10

    # 初始常数值（设为1）
    initial_guess = np.ones(len(constants))

    try:
        # 使用BFGS优化
        result = minimize(
            objective,
            initial_guess,
            method='BFGS',
            options={'maxiter': 100, 'disp': False}
        )

        if result.success:
            # 将优化后的常数替换回表达式
            optimized_expr = expr
            for i, c in enumerate(constants):
                optimized_expr = optimized_expr.subs(c, result.x[i])

            return optimized_expr, result.fun, True
        else:
            return expr, float('inf'), False

    except Exception as e:
        return expr, float('inf'), False


def evaluate_expression_with_constants(tree_str: str, x_values: np.ndarray, y_values: np.ndarray,
                                      error_callback=None) -> Tuple[bool, sp.Expr, float]:
    """
    评估包含常数的表达式

    1. 将前序遍历转换为表达式
    2. 用1替换常数进行语法检查
    3. 如果语法正确，进行BFGS优化

    Returns:
        (是否成功, 优化后的表达式, 最终损失)
    """
    try:
        # 1. 转换前序遍历为表达式
        expr = tree_to_expr(tree_str)

        # 2. 用1替换常数进行初步评估
        test_expr = expr
        for c in extract_constants(expr):
            test_expr = test_expr.subs(c, 1)

        # 3. 评估语法是否正确
        success, _ = evaluate_expression_safe(test_expr, x_values, error_callback)
        if not success:
            return False, expr, float('inf')

        # 4. 如果语法正确，进行常数优化
        optimized_expr, loss, opt_success = optimize_constants(expr, x_values, y_values)

        return True, optimized_expr, loss

    except Exception as e:
        if error_callback:
            error_callback(f"表达式评估失败: {str(e)}")
        return False, None, float('inf')

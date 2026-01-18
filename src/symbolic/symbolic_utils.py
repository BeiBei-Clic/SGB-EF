"""
符号表达式处理和计算工具函数
"""

import random
import numpy as np
import sympy as sp
from typing import List, Tuple, Union
from scipy.optimize import minimize
from src.symbolic.corruption import corrupt_expression
from src.utils.logger import Logger

# 创建全局 Logger 实例
_logger = Logger(enabled=True)

# 运算符定义
UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'abs', 'arcsin', 'tanh']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']


# ==================== 表达式生成与转换 ====================

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
    elif expr == sp.pi:
        return "pi"

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
    elif 'tanh' in func_name:
        return f'tanh,{expr_to_tree(args[0])}'
    elif 'sqrt' in func_name:
        return f'sqrt,{expr_to_tree(args[0])}'
    elif 'abs' in func_name:
        return f'abs,{expr_to_tree(args[0])}'  # Abs(x) -> abs,x
    else:
        return str(expr)


def generate_random_expr(input_dimension: int, max_depth: int = 4) -> sp.Expr:
    """生成随机表达式"""
    # 创建符号
    symbols = [sp.Symbol(f'x{i}') for i in range(input_dimension)]

    # 递归生成表达式的计数器
    recursion_count = 0
    max_recursions = 1000  # 防止无限递归

    def generate_expr(depth: int) -> sp.Expr:
        nonlocal recursion_count
        recursion_count += 1

        if recursion_count > max_recursions:
            raise Exception(f"递归次数过多: {recursion_count} > {max_recursions}")

        if depth >= max_depth:
            # 叶子节点：符号、常数或 pi（~5% 概率）
            if random.random() < 0.05:
                return sp.pi
            return random.choice(symbols + [sp.Rational(random.randint(-5, 5))])

        node_type = "operation" if random.random() < 0.5 else "leaf"

        if node_type == "operation":
            op_type = random.choice(['unary', 'binary'])

            if op_type == 'unary':
                op = random.choice(UNARY_OPS)
                operand = generate_expr(depth + 1)
                # 应用一元运算符
                if op == 'sin': return sp.sin(operand)
                elif op == 'cos': return sp.cos(operand)
                elif op == 'tan': return sp.tan(operand)
                elif op == 'exp': return sp.exp(operand)
                elif op == 'ln': return sp.log(abs(operand) + sp.Rational(1, 1000))
                elif op == 'tanh': return sp.tanh(operand)
                elif op == 'sqrt': return sp.sqrt(abs(operand))
                elif op == 'abs': return abs(operand)
                elif op == 'arcsin':
                    # 限制 arcsin 参数范围在 [-1, 1] 内避免复数
                    safe_operand = sp.sin(operand)  # sin 的结果在 [-1, 1]
                    return sp.asin(safe_operand)
                else: return operand
            else:
                op = random.choice(BINARY_OPS)
                left = generate_expr(depth + 1)
                right = generate_expr(depth + 1)
                # 应用二元运算符
                if op == 'add': return left + right
                elif op == 'sub': return left - right
                elif op == 'mul': return left * right
                elif op == 'div': return left / (right + sp.Rational(1, 100))
                elif op == 'pow': return left ** right
                else: return left + right
        else:
            # 叶子节点
            if random.random() < 0.05:
                return sp.pi
            return random.choice(symbols + [sp.Rational(random.randint(-5, 5))])

    # 生成表达式
    expr = generate_expr(0)

    # 简化表达式
    try:
        expr = simplify_expr(expr)
    except Exception:
        # 简化失败就使用原始表达式
        pass

    return expr


# ==================== 表达式简化 ====================

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
    if func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'asin', 'tanh']:
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


def reduce_expression_random_subtree(expr: sp.Expr) -> sp.Expr:
    """随机删减表达式中的任意子树，不局限于根节点

    改进点：
    - 30%概率删减根节点（保持原方法的行为）
    - 70%概率深入子树删减（新增行为，让编辑分布更均匀）

    Args:
        expr: 要删减的表达式

    Returns:
        删减后的表达式
    """
    # 如果表达式已经是常数或符号，无法进一步删减
    if expr.is_Number or expr.is_Symbol:
        return expr

    # 如果表达式没有参数，返回常数1
    if not hasattr(expr, 'args') or not expr.args:
        return sp.Integer(1)

    func_name = str(expr.func).lower()

    # 决定删减策略：删减根节点 vs 深入子树删减
    # 30%概率删减根节点，70%概率深入子树删减
    strategy = np.random.choice(['root', 'subtree'], p=[0.3, 0.7])

    if strategy == 'root':
        # 使用原有的根节点删减逻辑（保持向后兼容）
        if func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'asin', 'tanh']:
            # 一元操作：直接返回操作数
            return expr.args[0] if expr.args else sp.Integer(1)
        else:
            # 二元操作：随机选择一个子树
            return random.choice(expr.args) if expr.args else sp.Integer(1)

    else:  # subtree策略
        # 深入子树删减：随机选择一个参数进行内部删减
        if not expr.args:
            return sp.Integer(1)

        # 随机选择一个子树位置
        subtree_idx = random.randint(0, len(expr.args) - 1)
        selected_subtree = expr.args[subtree_idx]

        # 对选中的子树进行删减（递归调用原方法）
        if selected_subtree.is_Number or selected_subtree.is_Symbol:
            # 子树是叶子节点，简化为0或1
            reduced_subtree = sp.Integer(0) if random.random() < 0.5 else sp.Integer(1)
        elif hasattr(selected_subtree, 'args') and selected_subtree.args:
            # 递归删减子树（使用原方法，保持简单）
            reduced_subtree = reduce_expression(selected_subtree)
        else:
            reduced_subtree = sp.Integer(1)

        # 构建新的表达式，用删减后的子树替换原子树
        new_args = list(expr.args)
        new_args[subtree_idx] = reduced_subtree
        return expr.func(*new_args)


def generate_reduction_sequence(target_expr: sp.Expr, use_random_subtree: bool = True, sample_id: str = None) -> List[sp.Expr]:
    """生成从目标表达式逐步删减的序列，直到得到最简表达式

    Args:
        target_expr: 目标表达式
        use_random_subtree: 是否使用随机子树删减方法（默认True）
            - True: 使用reduce_expression_random_subtree（推荐）
            - False: 使用原版reduce_expression（只删根节点）
        sample_id: 样本ID（用于日志记录）

    Returns:
        删减序列，从目标表达式到最简表达式
    """
    import time

    reduction_sequence = []
    current_expr = target_expr

    # 持续删减直到表达式无法进一步简化
    max_iterations = 20  # 防止无限循环
    iterations = 0

    _logger.sample_step(sample_id, "REDUCTION_START",
                       f"目标表达式: {str(target_expr)} | max_iterations: {max_iterations}")

    while iterations < max_iterations:
        iteration_start = time.time()
        iterations += 1

        _logger.sample_step(sample_id, f"REDUCTION_ITERATION {iterations}/{max_iterations}",
                           f"当前表达式: {str(current_expr)}")

        reduction_sequence.append(current_expr)

        # 检查是否已经是最简形式（常数或符号）
        if current_expr.is_Number or current_expr.is_Symbol:
            iteration_time = (time.time() - iteration_start) * 1000
            _logger.sample_step(sample_id, "REDUCTION_COMPLETE",
                               f"原因: 达到最简形式({type(current_expr).__name__}) | "
                               f"总迭代数: {iterations} | 耗时: {iteration_time:.2f}ms")
            break

        # 应用一次删减操作
        reduce_start = time.time()
        if use_random_subtree:
            # 使用新的随机子树删减方法（让编辑分布更均匀）
            reduced_expr = reduce_expression_random_subtree(current_expr)
        else:
            # 使用原版删减方法（只删根节点）
            reduced_expr = reduce_expression(current_expr)
        reduce_time = (time.time() - reduce_start) * 1000

        _logger.sample_step(sample_id, f"REDUCTION_OP iteration={iterations}",
                           f"删减后: {str(reduced_expr)} | 耗时: {reduce_time:.2f}ms")

        # 如果删减后的表达式与原表达式相同，说明无法进一步删减
        if reduced_expr == current_expr:
            # 强制简化为常数
            _logger.sample_step(sample_id, f"REDUCTION_FORCE iteration={iterations}",
                               "删减后表达式相同，强制简化为常数1")
            reduced_expr = sp.Integer(1)

        # 简化表达式
        simplify_start = time.time()
        current_expr = simplify_expr(reduced_expr)
        simplify_time = (time.time() - simplify_start) * 1000

        iteration_time = (time.time() - iteration_start) * 1000

        _logger.sample_step(sample_id, f"SIMPLIFY iteration={iterations}",
                           f"简化后: {str(current_expr)} | "
                           f"简化耗时: {simplify_time:.2f}ms | "
                           f"总耗时: {iteration_time:.2f}ms")

    # 检查是否达到最大迭代次数
    if iterations >= max_iterations:
        _logger.sample_step(sample_id, "REDUCTION_COMPLETE",
                           f"原因: 达到最大迭代次数({max_iterations}) | "
                           f"最终表达式: {str(current_expr)}")

    # 确保序列中包含最简形式
    if not reduction_sequence or (reduction_sequence[-1].is_Number or reduction_sequence[-1].is_Symbol):
        if reduction_sequence and reduction_sequence[-1] != current_expr:
            reduction_sequence.append(current_expr)

    return reduction_sequence


# ==================== 序列对齐 ====================

def _build_dp_matrix(tokens1: List[str], tokens2: List[str]) -> List[List[int]]:
    """构建编辑距离DP矩阵

    Args:
        tokens1: 第一个token序列
        tokens2: 第二个token序列

    Returns:
        完成的DP矩阵
    """
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

    return dp


def levenshtein_alignment_with_gap(tokens1: List[str], tokens2: List[str]) -> Tuple[List[str], List[str]]:
    """使用Levenshtein距离计算两个token序列的对齐，返回包含gap token的等长序列"""
    dp = _build_dp_matrix(tokens1, tokens2)
    m, n = len(tokens1), len(tokens2)

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

    通过在DP回溯时随机选择最优路径来解决"gap集中"问题。

    核心原理：
    1. 构建标准编辑距离DP矩阵
    2. 回溯时，当存在多个最优方向时随机选择一个
    3. 保证 frm_blanks(z1) == tokens1 和 frm_blanks(z2) == tokens2

    Args:
        tokens1: 源token序列（当前表达式）
        tokens2: 目标token序列（目标表达式）

    Returns:
        Tuple[List[str], List[str]]: 对齐后的两个等长序列，包含<gap>标记
    """
    m, n = len(tokens1), len(tokens2)

    # 步骤1：构建标准编辑距离DP矩阵
    dp = _build_dp_matrix(tokens1, tokens2)

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


# ==================== 表达式计算与评估 ====================

def tree_to_expr(tree_str: str) -> sp.Expr:
    """将前序遍历字符串转换为SymPy表达式"""
    if not tree_str or tree_str.strip() == '':
        return sp.Integer(1)

    # 如果是单纯的符号或常数
    if ',' not in tree_str:
        if tree_str == 'constant':
            return sp.Symbol('c')  # 用符号c表示常数
        elif tree_str == 'pi':
            return sp.pi
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
        elif token == 'pi':
            return sp.pi
        elif token.startswith('x') and token[1:].isdigit():
            return sp.Symbol(token)
        elif token in UNARY_OPS:
            # 一元操作: op,operand
            operand = parse_expression(tokens_iter)
            # 应用一元运算符
            if token == 'sin': return sp.sin(operand)
            elif token == 'cos': return sp.cos(operand)
            elif token == 'tan': return sp.tan(operand)
            elif token == 'arcsin': return sp.asin(operand)
            elif token == 'exp': return sp.exp(operand)
            elif token == 'ln': return sp.log(abs(operand) + sp.Rational(1, 1000))
            elif token == 'tanh': return sp.tanh(operand)
            elif token == 'sqrt': return sp.sqrt(abs(operand))
            elif token == 'abs': return abs(operand)
            else: return operand
        elif token in BINARY_OPS:
            # 二元操作: op,left,right
            left = parse_expression(tokens_iter)
            right = parse_expression(tokens_iter)
            # 应用二元运算符
            if token == 'add': return left + right
            elif token == 'sub': return left - right
            elif token == 'mul': return left * right
            elif token == 'div': return left / (right + sp.Rational(1, 100))
            elif token == 'pow': return left ** right
            else: return left + right
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
        # 在给定x值上计算表达式
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
            result = np.full(x_values.shape[0] if x_values.ndim > 1 else len(x_values), value)
        else:
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

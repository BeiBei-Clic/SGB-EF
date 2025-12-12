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
import logging
import threading
import signal
from typing import List, Dict, Tuple
from tqdm import tqdm

# 自定义超时异常
class TimeoutError(Exception):
    """自定义超时异常"""
    pass

warnings.filterwarnings('ignore', category=RuntimeWarning)

UNARY_OPS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']

# 全局日志记录
import datetime
LOG_FILE = "logs/sample_generation.log"
STUCK_LOG_FILE = "logs/sample_stuck.log"

class TimeoutError(Exception):
    """自定义超时异常"""
    pass

def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("操作超时")

def with_timeout(func, timeout_seconds, *args, **kwargs):
    """为函数调用添加超时保护"""
    if hasattr(signal, 'SIGALRM'):  # Unix系统支持
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # 取消超时
            return result
        except TimeoutError:
            signal.alarm(0)  # 取消超时
            raise TimeoutError(f"函数 {func.__name__} 在 {timeout_seconds} 秒后超时")
        finally:
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
    else:
        # Windows系统不支持signal.SIGALRM，直接调用函数
        return func(*args, **kwargs)

def log_sample_step(sample_id, step, info=""):
    """记录样本生成步骤"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"{timestamp} [{sample_id}] {step}"
    if info:
        log_msg += f" - {info}"

    # 确保logs目录存在
    os.makedirs("logs", exist_ok=True)

    # 写入详细日志
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def log_sample_success(sample_id):
    """记录样本成功完成，并清理详细日志"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    success_msg = f"{timestamp} [{sample_id}] SUCCESS"

    # 写入成功标记
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(success_msg + "\n")

def log_sample_stuck(sample_id, duration, steps):
    """记录卡住的样本到专门的stuck日志"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    stuck_info = {
        "timestamp": timestamp,
        "sample_id": sample_id,
        "duration_seconds": round(duration, 2),
        "steps": steps
    }

    # 写入卡住日志
    with open(STUCK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"卡住样本记录:\n")
        f.write(f"  时间: {timestamp}\n")
        f.write(f"  样本ID: {sample_id}\n")
        f.write(f"  持续时间: {duration:.2f}秒\n")
        f.write(f"  步骤: {steps}\n")
        f.write("=" * 50 + "\n")

def cleanup_successful_logs():
    """清理已完成样本的详细日志，保留卡住的"""
    if not os.path.exists(LOG_FILE):
        return

    start_time = time.time()
    max_cleanup_time = 30  # 最多30秒清理时间

    try:
        stuck_samples = set()
        if os.path.exists(STUCK_LOG_FILE):
            # 从卡住日志中提取卡住的样本ID，限制读取时间
            with open(STUCK_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if time.time() - start_time > max_cleanup_time:
                        print("日志清理超时，跳过此次清理")
                        return
                    if "样本ID:" in line:
                        sample_id = line.split("样本ID: ")[1].strip()
                        stuck_samples.add(sample_id)

        # 检查日志文件大小，如果太大则直接清空
        if os.path.getsize(LOG_FILE) > 50 * 1024 * 1024:  # 50MB
            print("日志文件过大，直接清空成功样本日志")
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"# 日志已清理 - {datetime.datetime.now()}\n")
            return

        # 更高效的过滤：直接写新文件而不是重读
        temp_file = LOG_FILE + ".tmp"
        with open(LOG_FILE, "r", encoding="utf-8") as infile, \
             open(temp_file, "w", encoding="utf-8") as outfile:

            outfile.write(f"# 清理后的日志 - {datetime.datetime.now()}\n")

            for line in infile:
                if time.time() - start_time > max_cleanup_time:
                    print("日志清理超时，终止清理")
                    break

                # 简化过滤逻辑：只过滤包含SUCCESS的行
                if "SUCCESS" not in line:
                    outfile.write(line)

        # 替换原文件
        os.replace(temp_file, LOG_FILE)

    except Exception as e:
        print(f"清理日志时出错: {e}")
        # 如果清理失败，直接清空日志文件避免卡住
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"# 日志清理失败后重置 - {datetime.datetime.now()}\n")
        except:
            pass

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

    # 使用带时间限制的化简，然后检查是否包含复数
    expr = generate_expr(0)
    expr = timed_simplify(expr, max_time=1)

    # 检查并移除复数单位I
    if expr.has(sp.I):
        # 如果包含复数单位，替换为1
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

def preprocess_expression(expr: sp.Expr) -> sp.Expr:
    """预处理表达式，替换有问题的值"""
    # 替换复数单位I为1，避免复数计算问题
    expr = expr.replace(sp.I, sp.Integer(1))

    # 替换 zoo (复无穷大，SymPy中ComplexInfinity的实际名称)
    expr = expr.replace(sp.zoo, sp.Integer(1))

    # 替换其他无穷大形式
    expr = expr.replace(sp.oo, sp.Integer(1000000))
    expr = expr.replace(-sp.oo, sp.Integer(-1000000))

    # 处理可能的 NaN 值
    expr = expr.replace(sp.nan, sp.Integer(0))

    # 如果表达式仍包含复数运算符，尝试替换
    if expr.has(sp.I):
        expr = sp.re(expr)  # 提取实数部分

    return expr

def evaluate_expr(expr: sp.Expr, x_values: np.ndarray) -> np.ndarray:
    """在给定x值上计算表达式"""
    try:
        # 预处理表达式，替换 ComplexInfinity 和其他有问题的值
        expr = preprocess_expression(expr)

        n_dims = x_values.shape[1] if x_values.ndim > 1 else 1
        variables = [sp.Symbol(f'x{i}') for i in range(n_dims)]
        expr_vars = [var for var in variables if var in expr.free_symbols]

        if not expr_vars:
            # 对于常数表达式，提取实数部分
            value = float(sp.re(expr)) if expr.is_complex else float(expr)
            return np.full(x_values.shape[0] if x_values.ndim > 1 else len(x_values), value)

        f = sp.lambdify(expr_vars, expr, 'numpy')

        if x_values.ndim > 1:
            result = f(*[x_values[:, i] for i in range(len(expr_vars))])
        else:
            result = f(x_values)

        # 处理复数结果：提取实数部分
        if np.iscomplexobj(result):
            result = np.real(result)

        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(result, -1e6, 1e6)

    except (ValueError, TypeError, OverflowError) as e:
        # 处理复数转换错误和其他数值计算错误
        if "Cannot convert complex to float" in str(e) or "complex" in str(e).lower():
            # 如果是复数相关错误，返回全零数组
            shape = x_values.shape[0] if x_values.ndim > 1 else len(x_values)
            return np.zeros(shape)
        else:
            # 其他数值错误，重新抛出
            raise e

def generate_sample(input_dimension: int, n_points: int = 100, max_depth: int = 4) -> Dict:
    """生成单个样本"""
    sample_id = f"sample_{input_dimension}dim_{int(time.time() * 1000) % 1000000}"
    start_time = time.time()
    steps = []

    log_sample_step(sample_id, f"开始生成 {input_dimension}维样本")

    # 统一生成数据点，确保每个数据点是[x0, x1, x2, ...]的形式
    log_sample_step(sample_id, "生成数据点")
    x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, input_dimension))
    x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式
    steps.append("数据点生成完成")

    # 转换为numpy数组用于表达式计算
    x_array = np.array(x_values)

    # 生成目标表达式
    log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
    target_expr = generate_random_expr(input_dimension, max_depth)
    steps.append(f"目标表达式: {str(target_expr)}")

    # 计算目标值
    log_sample_step(sample_id, "计算目标表达式值")
    y_values = evaluate_expr(target_expr, x_array)
    steps.append("目标值计算完成")

    # 生成当前表达式
    log_sample_step(sample_id, "生成当前表达式(破坏)")
    curr_expr = corrupt_expression(target_expr, 0.5)
    steps.append(f"当前表达式: {str(curr_expr)}")

    log_sample_success(sample_id)

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

def save_samples_to_txt(samples: List[Dict], filename: str):
    """将样本保存到txt文件，每行一个样本"""
    print(f"保存数据到 {filename}...")

    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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

    # 设置真正的随机种子，确保每次运行生成不同的数据
    current_time = int(time.time()) % (2**32 - 1)  # 确保种子在有效范围内
    random.seed(current_time)
    np.random.seed(current_time)

    # 检查是否存在缓存文件
    filename = f"data/flow_samples_{num_samples}_{max_dim}dim_{n_points}pts_{max_depth}depth.txt"

    if use_cache and os.path.exists(filename):
        print(f"发现缓存文件 {filename}，直接加载数据...")
        return load_samples_from_txt(filename)

    # 分批生成数据样本，支持断点续传
    all_samples = []
    total_dimension_count = {}
    BATCH_SIZE = 50000  # 每批生成5万个样本

    print(f"分批生成 {num_samples} 个连续流训练样本，每批 {BATCH_SIZE} 个...")

    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    # 检查已完成的批次
    completed_batches = []
    for batch_idx in range(num_batches):
        batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
        if os.path.exists(batch_filename):
            completed_batches.append(batch_idx)

    if completed_batches:
        print(f"发现已完成 {len(completed_batches)} 个批次，将从第 {len(completed_batches) + 1} 批开始继续生成...")
        start_batch = len(completed_batches)
    else:
        start_batch = 0

    for batch_idx in range(start_batch, num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        current_batch_size = end_idx - start_idx

        print(f"\n生成第 {batch_idx + 1}/{num_batches} 批数据 ({current_batch_size} 个样本)...")

        batch_samples = []
        dimension_count = {}
        sample_count = 0
        pbar = tqdm(total=current_batch_size, desc=f"第{batch_idx + 1}批")

        while sample_count < current_batch_size:
            sample_start_time = time.time()
            sample_id = f"batch{batch_idx+1}_sample{sample_count}_{int(time.time() * 1000) % 1000000}"
            steps = []

            # 每个样本的最大处理时间（秒）
            SAMPLE_TIMEOUT = 5.0

            try:
                # 检查样本生成是否超时
                if time.time() - sample_start_time > SAMPLE_TIMEOUT:
                    log_sample_step(sample_id, "样本生成超时", f"超过{SAMPLE_TIMEOUT}秒")
                    sample_count += 1
                    pbar.update(1)
                    continue

                dim = random.randint(1, max_dim)
                dimension_count[dim] = dimension_count.get(dim, 0) + 1

                log_sample_step(sample_id, f"开始生成样本", f"批次{batch_idx+1}, 维度{dim}, 样本数{sample_count}/{current_batch_size}")

                # 生成数据点，统一处理确保每个数据点是[x0, x1, x2, ...]的形式
                log_sample_step(sample_id, "生成数据点", f"{n_points}个点, {dim}维")
                x_values_raw = np.random.uniform(-5.0, 5.0, (n_points, dim))
                x_values = [list(point) for point in x_values_raw]  # 转换为[[x0, x1, x2], [x3, x4, x5], ...]的形式
                x_array = np.array(x_values)  # 用于表达式计算
                steps.append("数据点生成完成")

                # 生成目标表达式
                log_sample_step(sample_id, "生成目标表达式", f"最大深度{max_depth}")
                try:
                    target_expr = generate_random_expr(dim, max_depth)
                    expr_str = str(target_expr)
                    steps.append(f"目标表达式: {expr_str}")
                except TimeoutError:
                    log_sample_step(sample_id, "跳过生成超时的表达式", "生成超时2秒")
                    sample_count += 1
                    pbar.update(1)
                    continue
                except Exception as expr_error:
                    log_sample_step(sample_id, "跳过生成失败的表达式", f"生成错误: {str(expr_error)}")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 保留必要的防御性检查
                if len(expr_str) > 24:
                    log_sample_step(sample_id, "跳过复杂表达式", f"长度{len(expr_str)} > 24")
                    sample_count += 1
                    pbar.update(1)
                    continue

                if target_expr.has(sp.I) or 'I' in expr_str:
                    log_sample_step(sample_id, "跳过复数表达式", f"包含复数单位I")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 计算目标表达式值
                log_sample_step(sample_id, "计算目标表达式值")
                try:
                    y_target = evaluate_expr(target_expr, x_array)
                    steps.append("目标值计算完成")
                except TimeoutError:
                    log_sample_step(sample_id, "跳过计算超时的表达式", "计算超时2秒")
                    sample_count += 1
                    pbar.update(1)
                    continue
                except Exception as eval_error:
                    log_sample_step(sample_id, "跳过计算失败的表达式", f"计算错误: {str(eval_error)}")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 生成删减序列
                log_sample_step(sample_id, "生成删减序列")
                try:
                    reduction_sequence = generate_reduction_sequence(target_expr)
                    steps.append(f"删减序列长度: {len(reduction_sequence)}")
                except TimeoutError:
                    log_sample_step(sample_id, "跳过生成超时的删减序列", "生成超时1秒")
                    sample_count += 1
                    pbar.update(1)
                    continue
                except Exception as reduction_error:
                    log_sample_step(sample_id, "跳过生成失败的删减序列", f"生成错误: {str(reduction_error)}")
                    sample_count += 1
                    pbar.update(1)
                    continue

                # 为删减序列中的每个表达式创建样本
                for i, reduced_expr in enumerate(reduction_sequence):
                    if sample_count >= current_batch_size:
                        break

                    log_sample_step(sample_id, f"处理删减表达式 {i+1}/{len(reduction_sequence)}", f"表达式: {str(reduced_expr)}")

                    # 对删减后的表达式应用额外的随机破坏
                    curr_expr = corrupt_expression(reduced_expr, corruption_prob=0.3)

                    # 检查删减后的表达式是否包含复数单位
                    if curr_expr.has(sp.I) or 'I' in str(curr_expr):
                        log_sample_step(sample_id, f"跳过复数删减表达式 {i+1}", f"表达式包含复数单位")
                        continue

                    # 计算当前值
                    try:
                        y_curr = evaluate_expr(curr_expr, x_array)
                    except TimeoutError:
                        log_sample_step(sample_id, f"跳过计算超时的删减表达式 {i+1}", "计算超时1秒")
                        continue
                    except Exception as eval_error:
                        log_sample_step(sample_id, f"跳过计算失败的删减表达式 {i+1}", f"计算错误: {str(eval_error)}")
                        continue

                    # 转换为token序列
                    target_tokens = expr_to_tree(target_expr).split(',')
                    curr_tokens = expr_to_tree(curr_expr).split(',')

                    # 对齐到Z空间，包含gap token
                    z0_tokens, z1_tokens = levenshtein_alignment_with_gap(curr_tokens, target_tokens)

                    batch_samples.append({
                        "input_dimension": dim,
                        "x_values": x_values,  # 保持与原代码一致的字段名
                        "y_target": y_target.tolist(),
                        "y_curr": y_curr.tolist(),
                        "residuals": (y_target - y_curr).tolist(),
                        "tree_gt": expr_to_tree(target_expr),
                        "exp_gt": str(target_expr),
                        "tree_cur1": expr_to_tree(curr_expr),
                        "exp_cur1": str(curr_expr),
                        "curr_tokens": curr_tokens,
                        "target_tokens": target_tokens,
                        "z0_tokens": z0_tokens,
                        "z1_tokens": z1_tokens
                    })

                    sample_count += 1
                    pbar.update(1)

                log_sample_success(sample_id)

                if sample_count >= current_batch_size:
                    break

            except Exception as e:
                # 记录卡住的样本
                duration = time.time() - sample_start_time
                steps.append(f"错误: {str(e)}")
                log_sample_stuck(sample_id, duration, steps)
                print(f"警告: 生成样本时出错，跳过该样本: {e}")
                continue

        pbar.close()

        # 累积维度统计
        for dim, count in dimension_count.items():
            total_dimension_count[dim] = total_dimension_count.get(dim, 0) + count

        # 立即保存当前批次
        batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
        save_samples_to_txt(batch_samples, batch_filename)

        # 将批次样本添加到总样本中
        all_samples.extend(batch_samples)

        print(f"第 {batch_idx + 1} 批完成并已保存到 {batch_filename}")
        print(f"当前批次维度分布:")
        for dim, count in sorted(dimension_count.items()):
            print(f"  {dim}维: {count} 个样本")

        # 每20个批次清理一次日志，保留卡住的记录（最后一批不清理）
        if (batch_idx + 1) % 20 == 0 and batch_idx + 1 < num_batches:
            print("清理成功样本的详细日志...")
            cleanup_successful_logs()

    # 加载已完成批次的样本
    print(f"\n加载已完成批次的样本...")
    for batch_idx in range(num_batches):
        batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
        if os.path.exists(batch_filename):
            batch_samples = load_samples_from_txt(batch_filename)
            all_samples.extend(batch_samples)
            # 统计维度分布
            for sample in batch_samples:
                dim = sample['input_dimension']
                total_dimension_count[dim] = total_dimension_count.get(dim, 0) + 1

    # 合并所有批次文件到一个文件
    print(f"合并 {num_batches} 个批次文件到主文件...")
    with open(filename, 'w', encoding='utf-8') as main_file:
        for batch_idx in range(num_batches):
            batch_filename = filename.replace('.txt', f'_batch_{batch_idx + 1}.txt')
            if os.path.exists(batch_filename):
                with open(batch_filename, 'r', encoding='utf-8') as batch_file:
                    main_file.write(batch_file.read())
                # 删除批次文件
                os.remove(batch_filename)
                print(f"已合并并删除批次文件: {batch_filename}")

    print(f"\n总体样本维度分布:")
    for dim, count in sorted(total_dimension_count.items()):
        print(f"{dim}维: {count} 个样本")

    print(f"所有数据已保存到: {filename}")

    return all_samples

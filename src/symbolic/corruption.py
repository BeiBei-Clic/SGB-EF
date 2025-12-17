"""
表达式破坏和变异工具函数
"""

import random
import time
import sympy as sp
from src.utils.timeout_utils import TimeoutError


def timed_simplify(expr: sp.Expr, max_time: float = 1) -> sp.Expr:
    """带时间限制的化简函数"""
    deadline = time.time() + max_time
    simplified = expr

    if time.time() < deadline:
        simplified = sp.together(simplified)
    if time.time() < deadline:
        simplified = sp.radsimp(simplified)

    return simplified


def replace_variables(expr: sp.Expr) -> sp.Expr:
    """替换表达式中的变量，模拟变量混淆错误"""
    # 获取表达式中的所有变量
    variables = list(expr.free_symbols)

    # 如果没有变量或者只有一个变量，直接返回原表达式
    if len(variables) <= 1:
        return expr

    # 创建变量替换映射
    # 策略1：随机交换两个变量
    if random.random() < 0.5:
        var1, var2 = random.sample(variables, 2)
        replacement_map = {var1: var2, var2: var1}

    # 策略2：将一个变量随机替换为另一个变量
    else:
        var_to_replace = random.choice(variables)
        var_candidates = [v for v in variables if v != var_to_replace]
        replacement_var = random.choice(var_candidates)
        replacement_map = {var_to_replace: replacement_var}

    # 应用替换
    return expr.xreplace(replacement_map)


def corrupt_expression(expr: sp.Expr) -> sp.Expr:
    """对表达式进行破坏，模拟计算错误"""
    corruption_start = time.time()
    """对表达式应用随机破坏"""
    corruption_type = random.choice(['simplify', 'replace_constant', 'mutate_operator', 'replace_variable'])

    if corruption_type == 'simplify':
        scaled_expr = expr * random.uniform(0.5, 2.0)
        return timed_simplify(scaled_expr, max_time=1)
    elif corruption_type == 'replace_constant' and expr.is_Number:
        return sp.Rational(random.randint(-5, 5))
    elif corruption_type == 'replace_variable':
        return replace_variables(expr)
    elif corruption_type == 'mutate_operator' and hasattr(expr, 'args') and len(expr.args) >= 1:
        func_name = str(expr.func).lower()

        # 加法操作符的变异
        if 'add' in func_name:
            mutation_type = random.choice(['sub', 'mul', 'div', 'pow'])
            if mutation_type == 'sub':
                return expr.args[0] - expr.args[1]
            elif mutation_type == 'mul':
                return expr.args[0] * expr.args[1]
            elif mutation_type == 'div':
                return expr.args[0] / (expr.args[1] + sp.Rational(1, 100))
            elif mutation_type == 'pow':
                return expr.args[0] ** (expr.args[1] + sp.Rational(1, 2))

        # 减法操作符的变异
        elif 'sub' in func_name:
            mutation_type = random.choice(['add', 'mul', 'div', 'pow'])
            if mutation_type == 'add':
                return expr.args[0] + expr.args[1]
            elif mutation_type == 'mul':
                return expr.args[0] * expr.args[1]
            elif mutation_type == 'div':
                return expr.args[0] / (expr.args[1] + sp.Rational(1, 100))
            elif mutation_type == 'pow':
                return expr.args[0] ** (expr.args[1] + sp.Rational(1, 2))

        # 乘法操作符的变异
        elif 'mul' in func_name:
            mutation_type = random.choice(['add', 'sub', 'div', 'pow'])
            if mutation_type == 'add':
                return expr.args[0] + expr.args[1]
            elif mutation_type == 'sub':
                return expr.args[0] - expr.args[1]
            elif mutation_type == 'div':
                return expr.args[0] / (expr.args[1] + sp.Rational(1, 100))
            elif mutation_type == 'pow':
                return expr.args[0] ** (expr.args[1] + sp.Rational(1, 2))

        # 除法操作符的变异
        elif 'div' in func_name or 'truediv' in func_name:
            mutation_type = random.choice(['add', 'sub', 'mul', 'pow'])
            if mutation_type == 'add':
                return expr.args[0] + expr.args[1]
            elif mutation_type == 'sub':
                return expr.args[0] - expr.args[1]
            elif mutation_type == 'mul':
                return expr.args[0] * expr.args[1]
            elif mutation_type == 'pow':
                return expr.args[0] ** (expr.args[1] + sp.Rational(1, 2))

        # 幂运算操作符的变异
        elif 'pow' in func_name:
            mutation_type = random.choice(['add', 'sub', 'mul', 'div', 'sqrt', 'exp'])
            if mutation_type == 'add':
                return expr.args[0] + expr.args[1]
            elif mutation_type == 'sub':
                return expr.args[0] - expr.args[1]
            elif mutation_type == 'mul':
                return expr.args[0] * expr.args[1]
            elif mutation_type == 'div':
                return expr.args[0] / (expr.args[1] + sp.Rational(1, 100))
            elif mutation_type == 'sqrt':
                return sp.sqrt(abs(expr.args[0]) + sp.Rational(1, 100))
            elif mutation_type == 'exp':
                return sp.exp(expr.args[0] + sp.Rational(1, 100))

        # 三角函数和一元运算符的变异
        elif func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']:
            operand = expr.args[0] if expr.args else sp.Integer(1)

            # 三角函数之间的相互替换
            if func_name == 'sin':
                mutation_type = random.choice(['cos', 'tan', 'abs', 'neg', 'square'])
                if mutation_type == 'cos':
                    return sp.cos(operand)
                elif mutation_type == 'tan':
                    return sp.tan(operand + sp.Rational(1, 100))  # 避免在零点tan奇异
                elif mutation_type == 'abs':
                    return abs(operand)
                elif mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'square':
                    return operand ** 2

            elif func_name == 'cos':
                mutation_type = random.choice(['sin', 'tan', 'abs', 'neg', 'square'])
                if mutation_type == 'sin':
                    return sp.sin(operand)
                elif mutation_type == 'tan':
                    return sp.tan(operand + sp.Rational(1, 100))
                elif mutation_type == 'abs':
                    return abs(operand)
                elif mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'square':
                    return operand ** 2

            elif func_name == 'tan':
                mutation_type = random.choice(['sin', 'cos', 'abs', 'neg', 'square'])
                if mutation_type == 'sin':
                    return sp.sin(operand)
                elif mutation_type == 'cos':
                    return sp.cos(operand)
                elif mutation_type == 'abs':
                    return abs(operand)
                elif mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'square':
                    return operand ** 2

            # 指数和对数函数的替换
            elif func_name == 'exp':
                mutation_type = random.choice(['log', 'sqrt', 'abs', 'neg', 'identity', 'square'])
                if mutation_type == 'log':
                    return sp.log(abs(operand) + sp.Rational(1, 100))
                elif mutation_type == 'sqrt':
                    return sp.sqrt(abs(operand) + sp.Rational(1, 100))
                elif mutation_type == 'abs':
                    return abs(operand)
                elif mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'identity':
                    return operand
                elif mutation_type == 'square':
                    return operand ** 2

            elif func_name == 'log':
                mutation_type = random.choice(['exp', 'sqrt', 'abs', 'neg', 'identity', 'square'])
                if mutation_type == 'exp':
                    return sp.exp(abs(operand))
                elif mutation_type == 'sqrt':
                    return sp.sqrt(abs(operand) + sp.Rational(1, 100))
                elif mutation_type == 'abs':
                    return abs(operand)
                elif mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'identity':
                    return operand
                elif mutation_type == 'square':
                    return operand ** 2

            # 开根号函数的替换
            elif func_name == 'sqrt':
                mutation_type = random.choice(['abs', 'neg', 'exp', 'log', 'identity', 'square'])
                if mutation_type == 'abs':
                    return abs(operand)
                elif mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'exp':
                    return sp.exp(abs(operand))
                elif mutation_type == 'log':
                    return sp.log(abs(operand) + sp.Rational(1, 100))
                elif mutation_type == 'identity':
                    return operand
                elif mutation_type == 'square':
                    return operand ** 2

            # 绝对值函数的替换
            elif func_name == 'abs':
                mutation_type = random.choice(['neg', 'exp', 'log', 'sqrt', 'identity', 'square', 'sin'])
                if mutation_type == 'neg':
                    return -operand
                elif mutation_type == 'exp':
                    return sp.exp(operand)
                elif mutation_type == 'log':
                    return sp.log(abs(operand) + sp.Rational(1, 100))
                elif mutation_type == 'sqrt':
                    return sp.sqrt(abs(operand))
                elif mutation_type == 'identity':
                    return operand
                elif mutation_type == 'square':
                    return operand ** 2
                elif mutation_type == 'sin':
                    return sp.sin(operand)

    corruption_time = (time.time() - corruption_start) * 1000
    if corruption_time > 500:  # 超过500ms记录警告
        print(f"WARNING: Expression corruption took {corruption_time:.1f}ms for type '{corruption_type}'")

    return expr
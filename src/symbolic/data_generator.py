"""
符号回归数据生成器，用于EditFlow预训练
生成训练三元组 (E_curr, E_target, r, z)
"""

import numpy as np
import torch
from typing import Tuple, List, Dict
from dataclasses import dataclass
import random
import sympy as sp


@dataclass
class SymbolicExpression:
    """符号表达式数据结构"""
    expression: sp.Expr
    tokens: List[str]
    variables: List[str]

    def evaluate(self, x_values: np.ndarray) -> np.ndarray:
        """在给定x值上计算表达式"""
        f = sp.lambdify(self.variables, self.expression, 'numpy')
        result = f(*x_values.T) if x_values.ndim > 1 else f(x_values)
        # 处理无穷大和NaN值
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(result, -1e6, 1e6)


class SymbolicVocabulary:
    """符号表达式词汇表"""

    def __init__(self):
        # 一元运算符
        self.unary_ops = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'neg']
        # 二元运算符
        self.binary_ops = ['add', 'sub', 'mul', 'div', 'pow']

        # 变量token（支持多维度）
        self.variables = [f'x{i+1}' for i in range(10)]  # 最多支持10维

        # 常数token
        self.constants = ['0', '1', '2', '3', '4', '5', 'C']  # C为可训练常数

        # 特殊token
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.gap_token = '<GAP>'

        # 构建词汇表
        all_tokens = (self.unary_ops + self.binary_ops + self.variables +
                     self.constants + [self.pad_token, self.bos_token,
                                      self.eos_token, self.gap_token])

        self.token_to_id = {token: i for i, token in enumerate(all_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

        # 特殊token ID
        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.gap_id = self.token_to_id[self.gap_token]

    def expression_to_tokens(self, expr: sp.Expr) -> List[str]:
        """将sympy表达式转换为token序列（树格式）"""
        def expr_to_tokens(e):
            # 变量
            if e.is_Symbol:
                return [str(e)]
            # 数字
            elif e.is_Number:
                val = float(e)
                if abs(val - round(val)) < 1e-6:
                    return [str(int(round(val)))]
                else:
                    return ['C']

            # 获取函数名
            func_name = str(e.func).lower()

            # 处理不同函数类型
            if 'add' in func_name:
                return ['add'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            elif 'sub' in func_name:
                return ['sub'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            elif 'mul' in func_name:
                return ['mul'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            elif 'div' in func_name or 'truediv' in func_name:
                return ['div'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            elif 'pow' in func_name:
                return ['pow'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            elif 'sin' in func_name:
                return ['sin'] + expr_to_tokens(e.args[0])
            elif 'cos' in func_name:
                return ['cos'] + expr_to_tokens(e.args[0])
            elif 'tan' in func_name:
                return ['tan'] + expr_to_tokens(e.args[0])
            elif 'exp' in func_name:
                return ['exp'] + expr_to_tokens(e.args[0])
            elif 'log' in func_name:
                return ['log'] + expr_to_tokens(e.args[0])
            elif 'sqrt' in func_name:
                return ['sqrt'] + expr_to_tokens(e.args[0])
            elif 'neg' in func_name:
                return ['neg'] + expr_to_tokens(e.args[0])
            else:
                return ['C']  # 默认用常数

        tokens = expr_to_tokens(expr)
        return [self.bos_token] + tokens + [self.eos_token]


class RandomExpressionGenerator:
    """随机表达式生成器，基于节点添加方法"""

    def __init__(self, input_dimension: int = 1, random_seed: int = None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.input_dimension = input_dimension
        self.unary_ops = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'neg']
        self.binary_ops = ['add', 'sub', 'mul', 'div', 'pow']

        # 创建变量符号
        self.symbols = [sp.Symbol(f'x{i+1}') for i in range(input_dimension)]

    def generate_random_expression(self, max_depth: int = 4) -> sp.Expr:
        """使用递归节点添加生成随机表达式"""
        def generate_expr(depth: int) -> sp.Expr:
            # 达到最大深度，返回叶子节点
            if depth >= max_depth:
                return random.choice(self.symbols + [sp.Rational(random.randint(-5, 5))])

            # 70%概率创建操作节点
            if random.random() < 0.7:
                op_type = random.choice(['unary', 'binary'])

                if op_type == 'unary':
                    op = random.choice(self.unary_ops)
                    operand = generate_expr(depth + 1)
                    return self._apply_unary_operator(op, operand)
                else:
                    op = random.choice(self.binary_ops)
                    left = generate_expr(depth + 1)
                    right = generate_expr(depth + 1)
                    return self._apply_binary_operator(op, left, right)
            else:
                # 30%概率返回叶子节点
                return random.choice(self.symbols + [sp.Rational(random.randint(-5, 5))])

        expr = generate_expr(0)
        return sp.simplify(expr)

    def _apply_unary_operator(self, op: str, operand: sp.Expr) -> sp.Expr:
        """应用一元运算符"""
        op_map = {
            'sin': lambda x: sp.sin(x),
            'cos': lambda x: sp.cos(x),
            'tan': lambda x: sp.tan(x),
            'exp': lambda x: sp.exp(x),
            'log': lambda x: sp.log(abs(x) + 1e-8),
            'sqrt': lambda x: sp.sqrt(abs(x)),
            'neg': lambda x: -x
        }
        return op_map.get(op, lambda x: x)(operand)

    def _apply_binary_operator(self, op: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
        """应用二元运算符"""
        op_map = {
            'add': lambda l, r: l + r,
            'sub': lambda l, r: l - r,
            'mul': lambda l, r: l * r,
            'div': lambda l, r: l / (r + sp.Rational(1, 100)),
            'pow': lambda l, r: l ** (sp.Rational(5) if r.is_Number and float(r) > 5 else
                                     sp.Rational(-5) if r.is_Number and float(r) < -5 else r)
        }
        return op_map.get(op, lambda l, r: l + r)(left, right)


class ExpressionCorruption:
    """表达式结构破坏器"""

    def __init__(self, vocab: SymbolicVocabulary):
        self.vocab = vocab

    def corrupt_expression(self, expr: sp.Expr, corruption_prob: float = 0.5) -> sp.Expr:
        """对表达式应用随机破坏"""
        def corrupt_recursive(e: sp.Expr, depth: int = 0) -> sp.Expr:
            if depth > 3:  # 限制递归深度
                return e

            if random.random() < corruption_prob:
                corruption_type = random.choice(['drop_term', 'mutate_op', 'replace_term'])

                if corruption_type == 'drop_term' and e.is_Add and len(e.args) > 1:
                    # 从加法中删除一项
                    keep_term = random.choice(e.args)
                    return corrupt_recursive(keep_term, depth + 1)

                elif corruption_type == 'mutate_op':
                    # 变异运算符
                    if e.is_Add and len(e.args) >= 2:
                        return corrupt_recursive(e.args[0] - e.args[1], depth + 1)
                    elif e.is_Mul and len(e.args) >= 2:
                        return corrupt_recursive(e.args[0] / e.args[1], depth + 1)

                elif corruption_type == 'replace_term' and hasattr(e, 'args') and e.args:
                    # 用常数替换子表达式
                    new_constant = sp.Rational(random.randint(-3, 3))
                    if e.is_Add:
                        args = list(e.args)
                        args[random.randint(0, len(args)-1)] = new_constant
                        return sp.Add(*args)
                    elif e.is_Mul:
                        args = list(e.args)
                        args[random.randint(0, len(args)-1)] = new_constant
                        return sp.Mul(*args)

            # 递归破坏子表达式
            if hasattr(e, 'args') and e.args:
                new_args = [corrupt_recursive(arg, depth + 1) for arg in e.args]
                func_name = str(e.func).lower()

                if 'add' in func_name:
                    return sp.Add(*new_args)
                elif 'mul' in func_name:
                    return sp.Mul(*new_args)
                elif func_name in self.vocab.unary_ops:
                    generator = RandomExpressionGenerator()
                    return generator._apply_unary_operator(func_name, new_args[0])

            return e

        corrupted = corrupt_recursive(expr)
        return corrupted


class AlignmentCalculator:
    """表达式对齐计算器"""

    @staticmethod
    def levenshtein_distance_with_ops(s1: List[str], s2: List[str]) -> List[str]:
        """计算Levenshtein距离并提取编辑操作"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 初始化DP表
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        # 回溯获取编辑操作
        edit_ops = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                edit_ops.append('keep')
                i, j = i-1, j-1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                edit_ops.append(f'sub:{s1[i-1]}->{s2[j-1]}')
                i, j = i-1, j-1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                edit_ops.append(f'del:{s1[i-1]}')
                i = i-1
            else:
                edit_ops.append(f'ins:{s2[j-1]}')
                j = j-1

        return list(reversed(edit_ops))


class SymbolicRegressionDataGenerator:
    """符号回归EditFlow预训练主数据生成器"""

    def __init__(self, input_dimension: int = 1, max_expression_depth: int = 4,
                 x_range: Tuple[float, float] = (-5.0, 5.0), n_points: int = 100,
                 corruption_prob: float = 0.5, random_seed: int = None):

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.input_dimension = input_dimension
        self.max_expression_depth = max_expression_depth
        self.corruption_prob = corruption_prob

        self.vocab = SymbolicVocabulary()
        self.expr_generator = RandomExpressionGenerator(input_dimension, random_seed)
        self.corruption = ExpressionCorruption(self.vocab)
        self.alignment = AlignmentCalculator()

        # 生成x值
        if input_dimension == 1:
            self.x_values = np.linspace(x_range[0], x_range[1], n_points)
        else:
            self.x_values = np.random.uniform(x_range[0], x_range[1], (n_points, input_dimension))

    def generate_target_expression(self) -> SymbolicExpression:
        """生成目标表达式"""
        expr = self.expr_generator.generate_random_expression(self.max_expression_depth)

        # 提取表达式中使用的变量
        variables = [str(var) for var in expr.free_symbols]
        if not variables:
            variables = [f'x{i+1}' for i in range(self.input_dimension)]

        tokens = self.vocab.expression_to_tokens(expr)

        return SymbolicExpression(
            expression=expr,
            tokens=tokens,
            variables=variables
        )

    def corrupt_expression(self, target_expr: SymbolicExpression) -> SymbolicExpression:
        """破坏目标表达式创建当前表达式"""
        corrupted_expr = self.corruption.corrupt_expression(
            target_expr.expression, self.corruption_prob
        )

        tokens = self.vocab.expression_to_tokens(corrupted_expr)

        return SymbolicExpression(
            expression=corrupted_expr,
            tokens=tokens,
            variables=target_expr.variables
        )

    def optimize_constants(self, curr_expr: SymbolicExpression, target_values: np.ndarray):
        """优化常数以最小化残差"""
        # 计算当前表达式值
        curr_values = curr_expr.evaluate(self.x_values)

        # 简单的缩放优化
        valid_mask = np.abs(curr_values) > 1e-8
        if valid_mask.sum() > 10:
            target_valid = target_values[valid_mask]
            curr_valid = curr_values[valid_mask]

            # 最小二乘求解缩放因子
            scale = np.dot(target_valid, curr_valid) / np.dot(curr_valid, curr_valid)
            scale = np.clip(scale, 0.1, 10.0)

            # 应用缩放
            optimized_expr = scale * curr_expr.expression
            tokens = self.vocab.expression_to_tokens(optimized_expr)

            optimized_expr_obj = SymbolicExpression(
                expression=optimized_expr,
                tokens=tokens,
                variables=curr_expr.variables
            )

            # 计算残差
            pred_values = optimized_expr_obj.evaluate(self.x_values)
            residual = target_values - pred_values

            return optimized_expr_obj, residual

        return curr_expr, target_values - curr_values

    def calculate_alignment(self, curr_tokens: List[str], target_tokens: List[str]) -> List[str]:
        """计算表达式间的对齐操作"""
        alignment_ops = self.alignment.levenshtein_distance_with_ops(
            curr_tokens, target_tokens
        )
        return alignment_ops

    def generate_training_triplet(self):
        """生成训练三元组 (E_curr, E_target, r, z)"""
        # 步骤A: 生成目标真值
        target_expr = self.generate_target_expression()
        target_values = target_expr.evaluate(self.x_values)

        # 步骤B: 创建破坏的当前表达式
        curr_expr = self.corrupt_expression(target_expr)

        # 步骤C: 优化常数并计算残差
        optimized_curr, residual = self.optimize_constants(curr_expr, target_values)

        # 步骤D: 计算对齐路径
        alignment_ops = self.calculate_alignment(
            optimized_curr.tokens, target_expr.tokens
        )

        return optimized_curr, target_expr, residual, alignment_ops

    def generate_batch(self, batch_size: int):
        """生成一批训练三元组"""
        batch = []
        for _ in range(batch_size):
            triplet = self.generate_training_triplet()
            batch.append(triplet)
        return batch


if __name__ == "__main__":
    # 测试数据生成器
    generator = SymbolicRegressionDataGenerator(
        input_dimension=1,
        max_expression_depth=4,
        random_seed=42
    )

    # 生成样本三元组
    curr, target, residual, alignment = generator.generate_training_triplet()

    print(f"目标表达式: {target.expression}")
    print(f"当前表达式: {curr.expression}")
    print(f"目标tokens: {target.tokens}")
    print(f"当前tokens: {curr.tokens}")
    print(f"残差形状: {residual.shape}")
    print(f"样本残差值: {residual[:10]}")
    print(f"对齐操作: {alignment[:10]}")

    # 测试批量生成
    batch = generator.generate_batch(5)
    print(f"\n成功生成 {len(batch)} 个三元组")
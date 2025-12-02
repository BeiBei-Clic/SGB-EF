import random
import json
import numpy as np
import sympy as sp
from pathlib import Path


class SymbolicExpressionGenerator:
    def __init__(self, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.unary_ops = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'neg']
        self.binary_ops = ['add', 'sub', 'mul', 'div', 'pow']

    def generate_random_expression(self, input_dimension, max_depth=4):
        symbols = [sp.Symbol(f'x{i+1}') for i in range(input_dimension)]

        def generate_expr(depth):
            if depth >= max_depth:
                return random.choice(symbols + [sp.Rational(random.randint(-5, 5))])

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
                return random.choice(symbols + [sp.Rational(random.randint(-5, 5))])

        expr = generate_expr(0)
        # 简化前检查表达式的合理性
        simplified = sp.simplify(expr)

        return simplified

    def _apply_unary_operator(self, op, operand):
        if op == 'sin': return sp.sin(operand)
        if op == 'cos': return sp.cos(operand)
        if op == 'tan': return sp.tan(operand)
        if op == 'exp': return sp.exp(operand)
        if op == 'log': return sp.log(abs(operand) + 1e-8)
        if op == 'sqrt': return sp.sqrt(abs(operand))
        if op == 'neg': return -operand
        return operand

    def _apply_binary_operator(self, op, left, right):
        if op == 'add': return left + right
        if op == 'sub': return left - right
        if op == 'mul': return left * right
        if op == 'div': return left / (right + sp.Rational(1, 100))
        if op == 'pow': return left ** right
        return left + right

    def expression_to_tree_format(self, expr):
        def expr_to_tokens(e):
            if e.is_Symbol or e.is_Number:
                return [str(e)]

            func_name = str(e.func).lower()

            if 'add' in func_name:
                return ['add'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            if 'mul' in func_name:
                return ['mul'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            if 'pow' in func_name:
                return ['pow'] + expr_to_tokens(e.args[0]) + expr_to_tokens(e.args[1])
            if 'sin' in func_name:
                return ['sin'] + expr_to_tokens(e.args[0])
            if 'cos' in func_name:
                return ['cos'] + expr_to_tokens(e.args[0])
            if 'tan' in func_name:
                return ['tan'] + expr_to_tokens(e.args[0])
            if 'exp' in func_name:
                return ['exp'] + expr_to_tokens(e.args[0])
            if 'log' in func_name:
                return ['log'] + expr_to_tokens(e.args[0])
            if 'sqrt' in func_name:
                return ['sqrt'] + expr_to_tokens(e.args[0])
            if 'neg' in func_name:
                return ['neg'] + expr_to_tokens(e.args[0])

            return [str(e)]

        tokens = expr_to_tokens(expr)
        return ','.join(tokens)


class DatasetManager:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def save(self, datasets, filename="all_datasets.txt"):
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            for dataset in datasets:
                json.dump(dataset, f)
                f.write('\n')
        return str(filepath)

    def generate_datasets(self, num_datasets=5, input_dimensions=[1,2,3,4,5],
                      samples_per_dataset=100, max_expression_depth=4, random_seed=None):
        expr_generator = SymbolicExpressionGenerator(random_seed=random_seed)
        all_datasets = []

        for dim in input_dimensions:
            for i in range(num_datasets):
                print(f"Generating dataset {i+1}/{num_datasets} for dimension {dim}...")

                expr = expr_generator.generate_random_expression(dim, max_expression_depth)
                tree_str = expr_generator.expression_to_tree_format(expr)
                readable_expr = str(expr)

                X = np.random.uniform(-10, 10, (samples_per_dataset, dim))
                symbols = [sp.Symbol(f'x{j+1}') for j in range(dim)]
                func = sp.lambdify(symbols, expr, 'numpy')

                y = func(*X.T)
                y = np.clip(y, -1e6, 1e6)

                all_datasets.append({
                    "input_dimension": dim,
                    "x": X.tolist(),
                    "y": y.tolist(),
                    "tree": tree_str,
                    "expression": readable_expr
                })

        filepath = self.save(all_datasets, "train_datasets.txt")
        print(f"Saved datasets to: {filepath}")
        return filepath


def main():
    print("Starting symbolic regression dataset generation...")
    dataset_manager = DatasetManager()

    saved_file = dataset_manager.generate_datasets(
        num_datasets=5,
        input_dimensions=[1,2,3,4,5],
        samples_per_dataset=100,
        max_expression_depth=4,
        random_seed=42
    )

    print(f"\nAll datasets saved to single file: {saved_file}")
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
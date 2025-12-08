"""
特殊token管理器 - 统一管理运算符、函数和变量token
"""

from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer


class SpecialTokensManager:
    """统一管理特殊token的类，避免在多个地方重复定义"""

    # 运算符定义
    OPERATORS = ['add', 'sub', 'mul', 'div', 'pow']

    # 函数定义
    FUNCTIONS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']

    # 默认最大变量维度
    DEFAULT_MAX_DIM = 10

    def __init__(self, tokenizer: PreTrainedTokenizer, max_dim: int = DEFAULT_MAX_DIM):
        """
        初始化特殊token管理器

        Args:
            tokenizer: 预训练的分词器
            max_dim: 最大变量维度
        """
        self.tokenizer = tokenizer
        self.max_dim = max_dim

        # 构建所有特殊token的映射
        self.special_tokens = {}
        self._build_token_mappings()

    def _build_token_mappings(self):
        """构建token映射"""
        # 添加运算符
        for op in self.OPERATORS:
            self.special_tokens[op] = op

        # 添加函数
        for func in self.FUNCTIONS:
            self.special_tokens[func] = func

        # 添加变量
        for i in range(self.max_dim):
            var_name = f'x{i}'
            self.special_tokens[var_name] = var_name

    def get_special_tokens(self) -> Dict[str, str]:
        """获取所有特殊token映射"""
        return self.special_tokens.copy()

    def get_function_token_map(self) -> Dict[str, int]:
        """
        获取函数到token ID的映射

        Returns:
            函数名到token ID的映射字典
        """
        func_map = {}

        for func_name in self.FUNCTIONS:
            # 使用分词器编码函数名，获取token ID
            tokens = self.tokenizer.encode(func_name, add_special_tokens=False)
            if len(tokens) == 1:
                # 理想情况：函数名被分词为单个token
                func_map[func_name] = tokens[0]
            elif len(tokens) > 1:
                # 警告：函数名被拆分为多个token
                # 仍然使用第一个token，但记录警告
                func_map[func_name] = tokens[0]

        return func_map

    def get_operators(self) -> List[str]:
        """获取运算符列表"""
        return self.OPERATORS.copy()

    def get_functions(self) -> List[str]:
        """获取函数列表"""
        return self.FUNCTIONS.copy()

    def get_variables(self) -> List[str]:
        """获取变量列表"""
        return [f'x{i}' for i in range(self.max_dim)]

    def tokenize_expression(self, expression: str) -> List[int]:
        """
        将表达式字符串转换为token序列

        Args:
            expression: 表达式字符串，以逗号分隔的tokens

        Returns:
            token ID列表
        """
        if not expression:
            return []

        tokens = expression.split(',')
        token_ids = []

        for token in tokens:
            if token in self.special_tokens:
                # 使用预训练模型tokenizer处理特殊token
                encoded = self.tokenizer.encode(self.special_tokens[token], add_special_tokens=False)
                token_ids.extend(encoded)
            elif token.replace('.', '').replace('-', '').isdigit():
                # 处理数字
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                token_ids.extend(encoded)

        return token_ids

    def print_function_mapping(self):
        """打印函数映射信息，用于调试"""
        func_map = self.get_function_token_map()
        if func_map:
            print("函数token映射:")
            for func_name, token_id in func_map.items():
                token_text = self.tokenizer.decode([token_id]) if self.tokenizer else f"ID: {token_id}"
                print(f"  {func_name} -> {token_id} ('{token_text}')")
        else:
            print("警告: 没有可用的函数映射")

        # 打印运算符映射
        print("运算符token映射:")
        for op_name in self.OPERATORS:
            tokens = self.tokenizer.encode(op_name, add_special_tokens=False)
            if len(tokens) == 1:
                token_id = tokens[0]
                token_text = self.tokenizer.decode([token_id]) if self.tokenizer else f"ID: {token_id}"
                print(f"  {op_name} -> {token_id} ('{token_text}')")
            else:
                print(f"  {op_name} -> multiple tokens: {tokens}")

    def is_function(self, token: str) -> bool:
        """检查token是否为函数"""
        return token in self.FUNCTIONS

    def is_operator(self, token: str) -> bool:
        """检查token是否为运算符"""
        return token in self.OPERATORS

    def is_variable(self, token: str) -> bool:
        """检查token是否为变量"""
        return token in self.get_variables()

    def is_special_token(self, token: str) -> bool:
        """检查token是否为特殊token"""
        return token in self.special_tokens
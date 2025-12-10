"""
特殊token管理器 - 统一管理运算符、函数和变量token
"""

from typing import Dict, List, Optional, Tuple, Set
from transformers import PreTrainedTokenizer


class SpecialTokensManager:
    """统一管理特殊token的类，避免在多个地方重复定义"""

    # 运算符定义
    OPERATORS = ['add', 'sub', 'mul', 'div', 'pow']

    # 函数定义
    FUNCTIONS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']

    # 占位符定义
    GAP_TOKEN = '<gap>'

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
        self.original_vocab_size = tokenizer.vocab_size
        self.added_tokens = set()  # 记录添加的新token

        # 构建所有特殊token的映射
        self.special_tokens = {}
        self.token_to_id = {}  # 添加token到ID的映射

        for op in self.OPERATORS:
            self.special_tokens[op] = op

        # 添加函数
        for func in self.FUNCTIONS:
            self.special_tokens[func] = func

        # 添加变量
        for i in range(self.max_dim):
            var_name = f'x{i}'
            self.special_tokens[var_name] = var_name

        # 添加占位符token
        self.special_tokens[self.GAP_TOKEN] = self.GAP_TOKEN

    def get_special_tokens(self) -> Dict[str, str]:
        """获取所有特殊token映射"""
        return self.special_tokens.copy()

    def get_gap_token_id(self) -> int:
        """
        获取gap token的ID

        Returns:
            gap token的ID
        """
        # 使用tokenizer编码gap token获取ID
        tokens = self.tokenizer.encode(self.GAP_TOKEN, add_special_tokens=False)
        return tokens[0]

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

    def is_special_token(self, token: str) -> bool:
        """检查token是否为特殊token"""
        return token in self.special_tokens

    def check_and_add_tokens(self, tokens_to_check: List[str]) -> Dict[str, int]:
        """
        检查token是否在分词器中存在，如果不存在则添加

        Args:
            tokens_to_check: 需要检查的token列表

        Returns:
            token到ID的映射字典
        """
        token_to_id = {}
        added_tokens_info = []

        for token in tokens_to_check:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1 and self.tokenizer.decode([token_ids[0]]) == token:
                token_to_id[token] = token_ids[0]
            else:
                self.tokenizer.add_tokens([token])
                self.added_tokens.add(token)
                new_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                token_to_id[token] = new_id
                added_tokens_info.append(f"{token} -> {new_id}")

        if added_tokens_info:
            print("添加的新token映射:")
            for info in added_tokens_info:
                print(f"  {info}")
            current_vocab_size = self.original_vocab_size + len(added_tokens_info)
            print(f"词表大小从 {self.original_vocab_size} 更新为 {current_vocab_size}")

        return token_to_id

    def ensure_special_tokens(self) -> Dict[str, int]:
        """
        确保所有特殊符号和变量都在分词器中存在

        Returns:
            完整的token到ID映射字典
        """
        all_tokens = list(self.special_tokens.keys())
        return self.check_and_add_tokens(all_tokens)

    def get_current_vocab_size(self) -> int:
        """获取当前词表大小"""
        return len(self.tokenizer.get_vocab())
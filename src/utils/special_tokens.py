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
        self._cached_vocab = None  # 缓存词表
        self._tokens_processed = False  # 标记是否已处理过特殊token

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

    def _get_cached_vocab(self) -> Dict[str, int]:
        """获取缓存的词表，如果词表被修改了则重新获取"""
        if not self._tokens_processed or self._cached_vocab is None:
            self._cached_vocab = self.tokenizer.get_vocab()
        return self._cached_vocab

    def get_gap_token_id(self) -> int:
        """
        获取gap token的ID

        Returns:
            gap token的ID
        """
        # 确保特殊token已经被处理
        if not self._tokens_processed:
            self.ensure_special_tokens()

        vocab = self._get_cached_vocab()
        return vocab[self.GAP_TOKEN]

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

        # 确保所有特殊token都已在词表中
        if not self._tokens_processed:
            self.ensure_special_tokens()

        vocab = self._get_cached_vocab()

        for token in tokens:
            if token in self.special_tokens:
                # 特殊token应该直接对应一个token ID
                if token in vocab:
                    token_ids.append(vocab[token])
                else:
                    raise ValueError(f"特殊token '{token}' 未在词表中找到")
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
        existing_tokens_info = []

        # 获取当前词表
        vocab = self.tokenizer.get_vocab()
        current_vocab_size = len(vocab)

        for token in tokens_to_check:
            # 检查token是否在词表中直接存在
            if token in vocab:
                # 直接找到对应的token ID
                token_id = vocab[token]
                token_to_id[token] = token_id
                existing_tokens_info.append(f"{token} -> {token_id}")
            else:
                # token不存在，需要添加
                self.tokenizer.add_tokens([token])
                self.added_tokens.add(token)
                # 重新获取词表
                vocab = self.tokenizer.get_vocab()
                new_id = vocab[token]
                token_to_id[token] = new_id
                added_tokens_info.append(f"{token} -> {new_id}")

        # 标记特殊token已被处理，并更新缓存
        self._tokens_processed = True
        self._cached_vocab = vocab

        # 显示所有特殊词汇的token映射信息（只在第一次有实际添加时才显示详细信息）
        if added_tokens_info:
            print("=== 特殊词汇Token映射信息 ===")
            if existing_tokens_info:
                print("已存在的token映射:")
                for info in existing_tokens_info:
                    print(f"  {info}")

            print("新增的token映射:")
            for info in added_tokens_info:
                print(f"  {info}")

            final_vocab_size = len(vocab)
            print(f"词表大小从 {current_vocab_size} 更新为 {final_vocab_size}")
            print("=" * 30)

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
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

    # 常数定义
    CONSTANT_TOKEN = 'constant'

    # BERT特殊token定义（如果分词器没有，我们会添加这些）
    BOS_TOKEN = '<s>'  # 开始符号
    EOS_TOKEN = '</s>'  # 结束符号
    PAD_TOKEN = '<pad>'  # 填充符号
    UNK_TOKEN = '<unk>'  # 未知符号
    MASK_TOKEN = '<mask>'  # 掩码符号

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

        # 添加常数token
        self.special_tokens[self.CONSTANT_TOKEN] = self.CONSTANT_TOKEN

        # 添加BERT特殊token（如果分词器没有对应的token）
        self.special_tokens[self.BOS_TOKEN] = self.BOS_TOKEN
        self.special_tokens[self.EOS_TOKEN] = self.EOS_TOKEN
        self.special_tokens[self.PAD_TOKEN] = self.PAD_TOKEN
        self.special_tokens[self.UNK_TOKEN] = self.UNK_TOKEN
        self.special_tokens[self.MASK_TOKEN] = self.MASK_TOKEN

    def get_special_tokens(self) -> Dict[str, str]:
        """获取所有特殊token映射"""
        return self.special_tokens.copy()

    def _get_cached_vocab(self) -> Dict[str, int]:
        """获取缓存的词表，如果词表被修改了则重新获取"""
        if not self._tokens_processed or self._cached_vocab is None:
            self._cached_vocab = self.tokenizer.get_vocab()
        return self._cached_vocab

    def get_token_id(self, token_name: str) -> int:
        """
        获取任意特殊token的ID

        Args:
            token_name: token名称，支持 'gap', 'pad', 'bos', 'eos', 'unk', 'mask'

        Returns:
            token的ID
        """
        # 确保特殊token已经被处理
        if not self._tokens_processed:
            self.ensure_special_tokens()

        vocab = self._get_cached_vocab()

        # token名称映射
        token_map = {
            'gap': self.GAP_TOKEN,
            'constant': self.CONSTANT_TOKEN,
            'pad': self.PAD_TOKEN,
            'bos': self.BOS_TOKEN,
            'eos': self.EOS_TOKEN,
            'unk': self.UNK_TOKEN,
            'mask': self.MASK_TOKEN
        }

        if token_name not in token_map:
            raise ValueError(f"未知的特殊token名称: {token_name}")

        token = token_map[token_name]
        return vocab[token]

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
                # 处理数字：统一使用constant token
                if self.CONSTANT_TOKEN in vocab:
                    token_ids.append(vocab[self.CONSTANT_TOKEN])
                else:
                    raise ValueError(f"常数token '{self.CONSTANT_TOKEN}' 未在词表中找到")

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

    def setup_tokenizer_special_tokens(self):
        """
        设置分词器的特殊token属性，确保分词器使用我们的特殊token
        如果分词器已经有对应的token，就使用分词器的；否则使用我们添加的
        """
        # 获取token映射
        token_map = {
            'pad': self.PAD_TOKEN,
            'bos': self.BOS_TOKEN,
            'eos': self.EOS_TOKEN,
            'unk': self.UNK_TOKEN,
            'mask': self.MASK_TOKEN
        }

        # 确保特殊token已添加
        self.ensure_special_tokens()
        vocab = self._get_cached_vocab()

        # 设置分词器的特殊token属性
        # 优先使用分词器原有的特殊token，如果没有才使用我们添加的
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token_id = vocab[self.tokenizer.pad_token]
        else:
            self.tokenizer.pad_token = self.PAD_TOKEN
            self.tokenizer.pad_token_id = vocab[self.PAD_TOKEN]

        if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token is not None:
            self.tokenizer.bos_token_id = vocab[self.tokenizer.bos_token]
        else:
            self.tokenizer.bos_token = self.BOS_TOKEN
            self.tokenizer.bos_token_id = vocab[self.BOS_TOKEN]

        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            self.tokenizer.eos_token_id = vocab[self.tokenizer.eos_token]
        else:
            self.tokenizer.eos_token = self.EOS_TOKEN
            self.tokenizer.eos_token_id = vocab[self.EOS_TOKEN]

        if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
            self.tokenizer.unk_token_id = vocab[self.tokenizer.unk_token]
        else:
            self.tokenizer.unk_token = self.UNK_TOKEN
            self.tokenizer.unk_token_id = vocab[self.UNK_TOKEN]

        if hasattr(self.tokenizer, 'mask_token') and self.tokenizer.mask_token is not None:
            self.tokenizer.mask_token_id = vocab[self.tokenizer.mask_token]
        else:
            self.tokenizer.mask_token = self.MASK_TOKEN
            self.tokenizer.mask_token_id = vocab[self.MASK_TOKEN]

        # 对于BERT，还需要设置cls_token和sep_token
        if hasattr(self.tokenizer, 'cls_token'):
            if self.tokenizer.cls_token is not None:
                self.tokenizer.cls_token_id = vocab[self.tokenizer.cls_token]
            else:
                self.tokenizer.cls_token = self.BOS_TOKEN
                self.tokenizer.cls_token_id = vocab[self.BOS_TOKEN]

        if hasattr(self.tokenizer, 'sep_token'):
            if self.tokenizer.sep_token is not None:
                self.tokenizer.sep_token_id = vocab[self.tokenizer.sep_token]
            else:
                self.tokenizer.sep_token = self.EOS_TOKEN
                self.tokenizer.sep_token_id = vocab[self.EOS_TOKEN]

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
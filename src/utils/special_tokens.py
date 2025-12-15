"""
特殊token管理器 - 统一管理运算符、函数和变量token
"""

from typing import Dict, List, Optional, Tuple, Set
from transformers import PreTrainedTokenizer


class SpecialTokensManager:
    """统一管理特殊token的类"""

    # 常量定义
    OPERATORS = ['add', 'sub', 'mul', 'div', 'pow']
    FUNCTIONS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
    GAP_TOKEN = '<gap>'
    CONSTANT_TOKEN = 'constant'
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    MASK_TOKEN = '<mask>'
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
        self.added_tokens = set()
        self._cached_vocab = None
        self._tokens_processed = False

        # 静态变量来跟踪全局状态，避免重复处理
        if not hasattr(SpecialTokensManager, '_processed_tokenizers'):
            SpecialTokensManager._processed_tokenizers = set()

        # 构建所有特殊token的映射（简化版本）
        self.special_tokens = {
            **{op: op for op in self.OPERATORS},
            **{func: func for func in self.FUNCTIONS},
            **{f'x{i}': f'x{i}' for i in range(self.max_dim)},
            self.GAP_TOKEN: self.GAP_TOKEN,
            self.CONSTANT_TOKEN: self.CONSTANT_TOKEN,
            self.BOS_TOKEN: self.BOS_TOKEN,
            self.EOS_TOKEN: self.EOS_TOKEN,
            self.PAD_TOKEN: self.PAD_TOKEN,
            self.UNK_TOKEN: self.UNK_TOKEN,
            self.MASK_TOKEN: self.MASK_TOKEN,
        }

    def _get_cached_vocab(self) -> Dict[str, int]:
        """获取缓存的词表，如果词表被修改了则重新获取"""
        if not self._tokens_processed or self._cached_vocab is None:
            self._cached_vocab = self.tokenizer.get_vocab()
        return self._cached_vocab

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

    def setup_tokenizer_special_tokens(self, verbose: bool = False):
        """
        设置分词器的特殊token属性
        优先使用分词器原有的特殊token，如果没有才使用我们添加的
        """
        self.ensure_special_tokens(verbose)
        vocab = self._get_cached_vocab()

        # 使用循环批量设置特殊token（简化重复代码）
        token_configs = [
            ('pad_token', 'pad_token_id', self.PAD_TOKEN),
            ('bos_token', 'bos_token_id', self.BOS_TOKEN),
            ('eos_token', 'eos_token_id', self.EOS_TOKEN),
            ('unk_token', 'unk_token_id', self.UNK_TOKEN),
            ('mask_token', 'mask_token_id', self.MASK_TOKEN),
        ]

        for token_attr, id_attr, default_token in token_configs:
            if hasattr(self.tokenizer, token_attr) and getattr(self.tokenizer, token_attr) is not None:
                setattr(self.tokenizer, id_attr, vocab[getattr(self.tokenizer, token_attr)])
            else:
                setattr(self.tokenizer, token_attr, default_token)
                setattr(self.tokenizer, id_attr, vocab[default_token])

        # BERT特有token设置
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

    def ensure_special_tokens(self, verbose: bool = False):
        """确保所有特殊符号和变量都在分词器中存在"""
        # 检查这个tokenizer是否已经被处理过
        tokenizer_id = id(self.tokenizer)
        if tokenizer_id in SpecialTokensManager._processed_tokenizers:
            self._tokens_processed = True
            self._cached_vocab = self.tokenizer.get_vocab()
            return

        vocab = self.tokenizer.get_vocab()
        missing_tokens = [token for token in self.special_tokens.keys() if token not in vocab]

        if missing_tokens:
            self.tokenizer.add_tokens(missing_tokens)
            self.added_tokens.update(missing_tokens)
            if verbose:
                print(f"已添加 {len(missing_tokens)} 个新token到词表")

        # 标记这个tokenizer为已处理
        SpecialTokensManager._processed_tokenizers.add(tokenizer_id)
        self._tokens_processed = True
        self._cached_vocab = self.tokenizer.get_vocab()

    def get_function_token_map(self) -> Dict[str, int]:
        """获取函数token到ID的映射字典"""
        if not self._tokens_processed:
            self.ensure_special_tokens()

        vocab = self._get_cached_vocab()
        return {func: vocab[func] for func in self.FUNCTIONS if func in vocab}
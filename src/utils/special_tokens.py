"""
特殊token管理器 - 统一管理运算符、函数和变量token
"""

from typing import Dict, List, Optional, Tuple, Set, Union
from transformers import PreTrainedTokenizer
import json
import os


# ============================================================================
# 符号回归专属小词汇表定义
# ============================================================================
class SymbolicVocab:
    """符号回归专属的紧凑词汇表"""

    # 运算符 (5个)
    OPERATORS = ['add', 'sub', 'mul', 'div', 'pow']

    # 函数 (7个)
    FUNCTIONS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']

    # 特殊token (7个)
    SPECIAL_TOKENS = {
        '<gap>': '<gap>',
        'constant': 'constant',  # 注意：constant不带尖括号，与SpecialTokensManager保持一致
        '<s>': '<s>',        # BOS
        '</s>': '</s>',      # EOS
        '<pad>': '<pad>',
        '<unk>': '<unk>',
        '<mask>': '<mask>',
    }

    @classmethod
    def get_full_vocab(cls, max_dim: int = 10) -> Dict[str, int]:
        """
        构建完整的词汇表

        Args:
            max_dim: 最大变量维度

        Returns:
            词汇表字典 {token: vocab_id}
        """
        vocab = {}

        # 1. 添加特殊token (ID: 0-6)
        for idx, (token_name, token_str) in enumerate(cls.SPECIAL_TOKENS.items()):
            vocab[token_str] = idx

        # 2. 添加运算符 (ID: 7-11)
        offset = len(cls.SPECIAL_TOKENS)
        for idx, op in enumerate(cls.OPERATORS):
            vocab[op] = offset + idx

        # 3. 添加函数 (ID: 12-18)
        offset += len(cls.OPERATORS)
        for idx, func in enumerate(cls.FUNCTIONS):
            vocab[func] = offset + idx

        # 4. 添加变量 (ID: 19-...)
        offset += len(cls.FUNCTIONS)
        for i in range(max_dim):
            vocab[f'x{i}'] = offset + i

        return vocab

    @classmethod
    def get_vocab_size(cls, max_dim: int = 10) -> int:
        """获取词汇表大小"""
        return len(cls.SPECIAL_TOKENS) + len(cls.OPERATORS) + len(cls.FUNCTIONS) + max_dim

    @classmethod
    def get_all_tokens(cls, max_dim: int = 10) -> List[str]:
        """获取所有token的列表（按ID顺序）"""
        tokens = list(cls.SPECIAL_TOKENS.values())
        tokens.extend(cls.OPERATORS)
        tokens.extend(cls.FUNCTIONS)
        tokens.extend([f'x{i}' for i in range(max_dim)])
        return tokens


class SymbolicRegressionTokenizer(PreTrainedTokenizer):
    """
    符号回归专属的分词器

    使用小词汇表，仅包含符号回归所需的token：
    - 运算符: add, sub, mul, div, pow
    - 函数: sin, cos, tan, exp, log, sqrt, abs
    - 变量: x0, x1, x2, ..., x(max_dim-1)
    - 特殊token: <gap>, <constant>, <s>, </s>, <pad>, <unk>, <mask>
    """

    def __init__(
        self,
        max_dim=10,
        vocab_file=None,
        do_lower_case=False,
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<s>',
        eos_token='</s>',
        mask_token='<mask>',
        **kwargs
    ):
        """
        初始化符号回归分词器

        Args:
            max_dim: 最大变量维度
            vocab_file: 词汇表文件路径（可选）
            do_lower_case: 是否转换为小写（不使用）
            unk_token: 未知token
            pad_token: 填充token
            bos_token: 序列开始token
            eos_token: 序列结束token
            mask_token: 掩码token
        """
        self.max_dim = max_dim

        # 构建词汇表
        self.vocab = SymbolicVocab.get_full_vocab(max_dim)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        # 保存配置
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        # 设置token对应的ID
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.pad_token_id = self.vocab.get(pad_token, 0)
        self.bos_token_id = self.vocab.get(bos_token, 0)
        self.eos_token_id = self.vocab.get(eos_token, 0)
        self.mask_token_id = self.vocab.get(mask_token, 0)

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小"""
        return SymbolicVocab.get_vocab_size(self.max_dim)

    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表字典"""
        return self.vocab.copy()

    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """将token或token列表转换为ID或ID列表"""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """将ID或ID列表转换为token或token列表"""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(idx) for idx in ids]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        分词文本

        对于符号回归，输入应该是逗号分隔的token字符串
        例如: "add,x0,x1" -> ["add", "x0", "x1"]
        """
        if not text or not text.strip():
            return []
        # 按逗号分割
        return text.split(',')

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """内部分词方法"""
        return self.tokenize(text, **kwargs)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """
        将token ID列表解码为字符串

        Args:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊token
            clean_up_tokenization_spaces: 是否清理空格（不使用）

        Returns:
            逗号分隔的token字符串
        """
        tokens = self.convert_ids_to_tokens(token_ids)

        if skip_special_tokens:
            # 过滤掉特殊token
            special_tokens = set(SymbolicVocab.SPECIAL_TOKENS.values())
            tokens = [t for t in tokens if t not in special_tokens]

        return ','.join(tokens)

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
        """
        保存词汇表到文件

        Args:
            save_directory: 保存目录

        Returns:
            保存的文件路径元组
        """
        if not os.path.isdir(save_directory):
            raise ValueError(f"目录不存在: {save_directory}")

        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

        # 保存配置
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {
            "tokenizer_type": "SymbolicRegressionTokenizer",
            "max_dim": self.max_dim,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        从预训练模型或目录加载分词器

        Args:
            pretrained_model_name_or_path: 预训练模型名称或路径

        Returns:
            SymbolicRegressionTokenizer实例
        """
        # 尝试加载配置文件
        config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")

        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            max_dim = config.get('max_dim', 10)
        else:
            max_dim = kwargs.get('max_dim', 10)

        return cls(max_dim=max_dim, *args, **kwargs)


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

        # 如果使用的是SymbolicRegressionTokenizer，不需要添加任何token
        if isinstance(self.tokenizer, SymbolicRegressionTokenizer):
            if verbose:
                print(f"使用SymbolicRegressionTokenizer，已包含所有符号回归token，无需添加")

            # 标记这个tokenizer为已处理
            SpecialTokensManager._processed_tokenizers.add(tokenizer_id)
            self._tokens_processed = True
            self._cached_vocab = self.tokenizer.get_vocab()
            return

        # 对于其他tokenizer，检查并添加缺失的token
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
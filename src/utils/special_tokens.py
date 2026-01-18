"""
特殊token管理器 - 统一管理运算符、函数和变量token
"""

from typing import Dict, List, Union
from transformers import PreTrainedTokenizer


# ============================================================================
# 符号回归专属小词汇表定义
# ============================================================================
class SymbolicVocab:
    """符号回归专属的紧凑词汇表"""

    # 运算符 (5个)
    OPERATORS = ['add', 'sub', 'mul', 'div', 'pow']

    # 函数 (9个)
    FUNCTIONS = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'abs', 'arcsin', 'tanh']

    # 常数 (1个)
    CONSTANTS = ['pi']

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
        for idx, (_, token_str) in enumerate(cls.SPECIAL_TOKENS.items()):
            vocab[token_str] = idx

        # 2. 添加运算符 (ID: 7-11)
        offset = len(cls.SPECIAL_TOKENS)
        for idx, op in enumerate(cls.OPERATORS):
            vocab[op] = offset + idx

        # 3. 添加函数 (ID: 12-19)
        offset += len(cls.OPERATORS)
        for idx, func in enumerate(cls.FUNCTIONS):
            vocab[func] = offset + idx

        # 4. 添加常数 (ID: 20)
        offset += len(cls.FUNCTIONS)
        for idx, const in enumerate(cls.CONSTANTS):
            vocab[const] = offset + idx

        # 5. 添加变量 (ID: 21-...)
        offset += len(cls.CONSTANTS)
        for i in range(max_dim):
            vocab[f'x{i}'] = offset + i

        return vocab

    @classmethod
    def get_vocab_size(cls, max_dim: int = 10) -> int:
        """获取词汇表大小"""
        return len(cls.SPECIAL_TOKENS) + len(cls.OPERATORS) + len(cls.FUNCTIONS) + len(cls.CONSTANTS) + max_dim


class SymbolicRegressionTokenizer(PreTrainedTokenizer):
    """
    符号回归专属的分词器

    使用小词汇表，仅包含符号回归所需的token：
    - 运算符: add, sub, mul, div, pow
    - 函数: sin, cos, tan, exp, ln, sqrt, abs, arcsin, tanh
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

    def get_vocab(self):
        """返回词汇表字典（PreTrainedTokenizer要求的方法）"""
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
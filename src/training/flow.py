"""
EditFlow连续流匹配的核心组件
"""

import torch
import json
import os
from typing import List, Dict, Tuple, Optional
from ..symbolic.data_generator import generate_flow_samples
from ..utils.special_tokens import SpecialTokensManager


class KappaScheduler:
    """时间调度器，用于控制流的插值"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler_type = scheduler_type

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """返回时间t的调度系数"""
        return 3 * t**2 - 2 * t**3 if self.scheduler_type == 'cubic' else t

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """返回时间t的调度系数导数"""
        return 6 * t - 6 * t**2 if self.scheduler_type == 'cubic' else torch.ones_like(t)


def sample_conditional_path(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, scheduler: KappaScheduler, debug: bool = False) -> torch.Tensor:
    """在给定时间t采样条件路径"""
    batch_size, seq_len, vocab_size = p0.shape
    t = t.view(-1, 1, 1) if t.dim() == 1 else t
    kappa_t = scheduler(t)
    kappa_t = kappa_t.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)

    # 线性插值
    pt = (1 - kappa_t) * p0 + kappa_t * p1

    # 确保概率为非负并归一化
    pt = torch.clamp(pt, min=0.0)
    pt_sum = pt.sum(dim=-1, keepdim=True)
    # 防止除零
    pt_sum = torch.clamp(pt_sum, min=1e-8)
    pt = pt / pt_sum

    # 采样
    pt_flat = pt.view(-1, pt.size(-1))
    sampled = torch.multinomial(pt_flat, 1)
    return sampled.view(batch_size, seq_len)



def remove_gap_tokens(z_t: torch.Tensor, special_tokens_manager: SpecialTokensManager) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列"""
    pad_token_id = special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')
    gap_token_id = special_tokens_manager.tokenizer.convert_tokens_to_ids('<gap>')
    batch_size, z_seq_len = z_t.shape
    device = z_t.device

    z_gap_mask = (z_t == gap_token_id)
    z_pad_mask = (z_t == pad_token_id)

    # 使用掩码操作移除gap tokens
    x_t_list = []
    for i in range(batch_size):
        non_gap_mask = ~z_gap_mask[i]
        x_row = z_t[i][non_gap_mask]
        x_t_list.append(x_row)

    max_len = max(len(x) for x in x_t_list)
    x_t_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    x_pad_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i, x_row in enumerate(x_t_list):
        x_t_padded[i, :len(x_row)] = x_row
        x_pad_mask_padded[i, :len(x_row)] = (x_row == pad_token_id)

    return x_t_padded, x_pad_mask_padded, z_gap_mask, z_pad_mask


def fill_gap_tokens_with_repeats(x_ut: torch.Tensor, z_gap_mask: torch.Tensor, z_pad_mask: torch.Tensor) -> torch.Tensor:
    """用重复值填充gap token位置"""
    batch_size, z_seq_len = z_gap_mask.shape
    _, x_seq_len, vocab_size = x_ut.shape

    # 计算每个位置对应的非gap位置
    non_gap_mask = ~z_gap_mask
    indices = non_gap_mask.cumsum(dim=1) - 1
    indices = indices.clamp(min=0, max=x_seq_len-1)

    # 收集对应的特征
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]
    result[z_pad_mask] = 0

    return result


class ContinuousFlowLoss:
    """连续时间流匹配损失函数"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler = KappaScheduler(scheduler_type)

    def make_ut_mask_from_z(self, z_t: torch.Tensor, z_1: torch.Tensor, vocab_size: int,
                           gap_token: int, special_tokens_manager: SpecialTokensManager) -> torch.Tensor:
        batch_size, z_seq_len = z_t.shape
        n_ops = 2 * vocab_size + 1

        pad_token = special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')

        z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
        z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq
        z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq
        z_sub = z_neq & ~z_ins & ~z_del

        u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
        u_mask[z_ins, z_1[z_ins]] = True
        u_mask[z_sub, z_1[z_sub] + vocab_size] = True
        u_mask[:, :, -1][z_del] = True

        return u_mask

    def __call__(self, u_cat: torch.Tensor, u_mask: torch.Tensor,
                 t: torch.Tensor, vocab_size: int) -> torch.Tensor:
        u_total = u_cat.sum(dim=(1, 2))
        sched_coeff = (self.scheduler.derivative(t) / (1 - self.scheduler(t) + 1e-8)).squeeze(-1)
        sched_coeff = torch.clamp(sched_coeff, min=-10, max=10)

        log_u_cat = torch.log(torch.clamp(u_cat, min=1e-12, max=1e12))
        cross_entropy = (log_u_cat * u_mask.float()).sum(dim=(1, 2))

        loss = u_total - cross_entropy * sched_coeff
        return loss.mean()


class FlowDataset(torch.utils.data.Dataset):
    """连续流数据集 (z0, z1, x_values, residuals) - 基于文件按需读取，避免内存溢出"""

    def __init__(self, positions, filename, tokenizer, max_dim=10, max_expr_length=128):
        """
        基于文件位置索引的数据集

        Args:
            positions: 文件中样本的位置索引列表
            filename: 数据文件路径
            tokenizer: 分词器
            max_dim: 最大维度
            max_expr_length: 表达式最大长度
        """
        self.positions = positions
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_expr_length = max_expr_length
        self.max_dim = max_dim
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=max_dim)

        # 设置分词器的特殊token属性
        self.special_tokens_manager.setup_tokenizer_special_tokens()

        self.vocab_size = self.special_tokens_manager.get_current_vocab_size()
        self.pad_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')
        self.bos_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<s>')
        self.gap_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<gap>')

    def __len__(self):
        return len(self.positions)

    def _tokenize_expression_tokens(self, tokens: List[str]) -> List[int]:
        """将token列表转换为token ID列表"""
        token_ids = []
        for token in tokens:
            tokenized = self.special_tokens_manager.tokenize_expression(token)
            token_ids.extend(tokenized)
        return token_ids

    def __getitem__(self, idx):
        # 从文件指定位置读取样本
        with open(self.filename, 'r', encoding='utf-8') as f:
            f.seek(self.positions[idx])
            line = f.readline().strip()
            sample = json.loads(line)

        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])

        z0_tokens = self._tokenize_expression_tokens(sample['z0_tokens'])
        z1_tokens = self._tokenize_expression_tokens(sample['z1_tokens'])

        max_len = self.max_expr_length
        def pad_z_sequence(tokens):
            # 过滤掉None值，并确保所有token都是整数
            filtered_tokens = [t for t in tokens if t is not None and isinstance(t, int)]
            if len(filtered_tokens) != len(tokens):
                print(f"警告: 过滤了 {len(tokens) - len(filtered_tokens)} 个无效token (原始: {tokens})")
            tokens = [self.bos_token] + filtered_tokens[:max_len-1]
            tokens.extend([self.pad_token] * (max_len - len(tokens)))
            return torch.LongTensor(tokens)

        return {
            'x_values': x_values,
            'residuals': residuals,
            'z0_token_ids': pad_z_sequence(z0_tokens),
            'z1_token_ids': pad_z_sequence(z1_tokens),
            'gap_token': self.gap_token
        }


def custom_collate_fn(batch):
    return {
        'x_values': torch.stack([item['x_values'] for item in batch]),
        'residuals': torch.stack([item['residuals'] for item in batch]),
        'z0_token_ids': torch.stack([item['z0_token_ids'] for item in batch]),
        'z1_token_ids': torch.stack([item['z1_token_ids'] for item in batch]),
        'gap_token': batch[0]['gap_token']
    }



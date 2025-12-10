"""
EditFlow连续流匹配的核心组件
"""

import torch
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


def sample_conditional_path(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, scheduler: KappaScheduler) -> torch.Tensor:
    """在给定时间t采样条件路径"""
    batch_size, seq_len, vocab_size = p0.shape
    t = t.view(-1, 1, 1) if t.dim() == 1 else t
    kappa_t = scheduler(t)
    kappa_t = kappa_t.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)

    pt = (1 - kappa_t) * p0 + kappa_t * p1
    pt = pt / pt.sum(dim=-1, keepdim=True)

    pt_flat = pt.view(-1, pt.size(-1))
    sampled = torch.multinomial(pt_flat, 1)
    return sampled.view(batch_size, seq_len)


def tokens_to_prob(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """将token序列转换为概率分布"""
    batch_size, seq_len = tokens.shape
    probs = torch.zeros(batch_size, seq_len, vocab_size, device=tokens.device)

    # 确保token IDs在有效范围内，防止越界
    valid_tokens = torch.clamp(tokens, 0, vocab_size - 1)
    probs.scatter_(2, valid_tokens.unsqueeze(-1), 1.0)
    return probs


def remove_gap_tokens(z_t: torch.Tensor, pad_token_id: int, gap_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列"""
    gap_token_id = gap_token_id
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
                           gap_token: int, pad_token: int) -> torch.Tensor:
        batch_size, z_seq_len = z_t.shape
        n_ops = 2 * vocab_size + 1

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
    """连续流数据集 (z0, z1, x_values, residuals)"""

    def __init__(self, samples: List[Dict], tokenizer, max_dim=10):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token = tokenizer.pad_token_id
        self.bos_token = tokenizer.cls_token_id  # BERT使用cls_token
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=max_dim)
        self.gap_token = self.special_tokens_manager.get_gap_token_id()

    def __len__(self):
        return len(self.samples)

    def _tokenize_expression_tokens(self, tokens: List[str]) -> List[int]:
        """将token列表转换为token ID列表"""
        token_ids = []
        for token in tokens:
            tokenized = self.special_tokens_manager.tokenize_expression(token)
            token_ids.extend(tokenized)
        return token_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])
        # residuals应该保持为1D张量，在collate_fn中会被正确地堆叠为2D

        z0_tokens = self._tokenize_expression_tokens(sample['z0_tokens'])
        z1_tokens = self._tokenize_expression_tokens(sample['z1_tokens'])

        max_len = 128
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



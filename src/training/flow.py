"""
EditFlow连续流匹配的核心组件
"""

import torch
import json
import os
import time
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
    kappa_t = scheduler(t).view(batch_size, 1, 1).expand(batch_size, seq_len, 1)

    pt = (1 - kappa_t) * p0 + kappa_t * p1
    pt = torch.clamp(pt, min=0.0)
    pt_sum = torch.clamp(pt.sum(dim=-1, keepdim=True), min=1e-8)
    pt = pt / pt_sum

    pt_flat = pt.view(-1, pt.size(-1))
    return torch.multinomial(pt_flat, 1).view(batch_size, seq_len)



def remove_gap_tokens(z_t: torch.Tensor, special_tokens_manager: SpecialTokensManager) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列

    重要：此函数只移除 gap_token，保留 BOS token 和所有其他 tokens
    确保返回的序列格式为 [BOS] + [non_gap_tokens] + [PAD...]
    """
    pad_token_id = special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')
    gap_token_id = special_tokens_manager.tokenizer.convert_tokens_to_ids('<gap>')
    bos_token_id = special_tokens_manager.tokenizer.convert_tokens_to_ids('<s>')
    batch_size, z_seq_len = z_t.shape
    device = z_t.device

    z_gap_mask = (z_t == gap_token_id)
    z_pad_mask = (z_t == pad_token_id)

    # 使用掩码操作移除gap tokens（只移除gap，保留BOS和其他tokens）
    x_t_list = []
    for i in range(batch_size):
        non_gap_mask = ~z_gap_mask[i]
        x_row = z_t[i][non_gap_mask]
        x_t_list.append(x_row)

        # 验证：确保BOS token被保留（如果输入中存在）
        if len(x_row) > 0 and z_t[i, 0] == bos_token_id:
            assert x_row[0] == bos_token_id, f"BOS token必须被保留在位置0"

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
                 t: torch.Tensor, vocab_size: int, accelerator=None) -> torch.Tensor:

        # NaN/Inf检测
        nan_count = torch.isnan(u_cat).sum().item()
        inf_count = torch.isinf(u_cat).sum().item()


        # 分布式NaN检测：如果使用accelerator，确保所有进程同步跳过
        has_nan_or_inf = torch.tensor(1 if (nan_count > 0 or inf_count > 0) else 0,
                                    device=u_cat.device)

        if accelerator is not None and accelerator.distributed_type != "NO":
            # 同步所有进程的NaN检测结果
            # 使用accelerator.gather收集所有进程的NaN检测结果
            gathered_results = accelerator.gather(has_nan_or_inf)
            has_nan_or_inf = gathered_results.sum()
        else:
            has_nan_or_inf = has_nan_or_inf.item()

        # 如果任何进程包含NaN或Inf，抛出异常让训练循环跳过该批次
        if has_nan_or_inf > 0:
            raise ValueError(f"批次包含异常值: 本地NaN={nan_count}, Inf={inf_count}, 分布式检测={has_nan_or_inf}")


        u_total = u_cat.sum(dim=(1, 2))


        sched_coeff = (self.scheduler.derivative(t) / (1 - self.scheduler(t) + 1e-8)).squeeze(-1)
        sched_coeff = torch.clamp(sched_coeff, min=-10, max=40)


        log_u_cat = torch.log(torch.clamp(u_cat, min=1e-12, max=1e12))


        cross_entropy = (log_u_cat * u_mask.float()).sum(dim=(1, 2))


        loss = (u_total - cross_entropy * sched_coeff).mean()


        return loss


class FlowDataset(torch.utils.data.Dataset):
    """连续流数据集 (z0, z1, x_values, residuals) - 支持内存预加载和文件按需读取"""

    def __init__(self, positions, filename, tokenizer, max_dim=10, max_expr_length=128, verbose=False, preload_to_memory=False):
        """
        基于文件位置索引的数据集

        Args:
            positions: 文件中样本的位置索引列表
            filename: 数据文件路径
            tokenizer: 分词器
            max_dim: 最大维度
            max_expr_length: 表达式最大长度
            verbose: 是否输出详细日志
            preload_to_memory: 是否预加载数据到内存（性能优化）
        """
        self.positions = positions
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_expr_length = max_expr_length
        self.max_dim = max_dim
        self.preload_to_memory = preload_to_memory
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=max_dim)

        # 设置分词器的特殊token属性
        self.special_tokens_manager.setup_tokenizer_special_tokens(verbose)

        self.vocab_size = len(self.special_tokens_manager.tokenizer.get_vocab())
        self.pad_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')
        self.bos_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<s>')
        self.gap_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<gap>')

        # 预加载数据到内存（可选）
        self.data_cache = None
        if self.preload_to_memory:
            self._preload_data(verbose)

    def __len__(self):
        return len(self.positions)

    def _preload_data(self, verbose=False):
        """预加载数据到内存以提高IO性能"""
        if verbose:
            print(f"预加载 {len(self.positions)} 个样本到内存...")

        self.data_cache = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            for pos in self.positions:
                f.seek(pos)
                line = f.readline().strip()
                sample = json.loads(line)
                self.data_cache.append(sample)

        if verbose:
            print(f"数据预加载完成，占用约 {len(self.data_cache) * 0.5:.1f} MB 内存")

    def _tokenize_expression_tokens(self, tokens: List[str]) -> List[int]:
        """将token列表转换为token ID列表"""
        token_ids = []
        for token in tokens:
            tokenized = self.special_tokens_manager.tokenize_expression(token)
            token_ids.extend(tokenized)
        return token_ids

    def __getitem__(self, idx):
        # 如果数据已预加载到内存，直接从缓存读取
        if self.data_cache is not None:
            sample = self.data_cache[idx]
        else:
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
    """处理不同维度数据的collate函数，使用padding + mask方案"""
    if len(batch) == 0:
        return {
            'x_values': torch.empty(0, 0, 0),
            'residuals': torch.empty(0, 0),
            'dim_mask': torch.empty(0, 0),
            'z0_token_ids': torch.empty(0, 0),
            'z1_token_ids': torch.empty(0, 0),
            'gap_token': None
        }

    # 找到最大维度和n_points
    max_dim = 0
    max_n_points = 0
    original_dims = []

    for i, item in enumerate(batch):
        x_val = item['x_values']  # [n_points, current_dim]
        resid = item['residuals']  # [n_points]

        # 确保x_values至少是2维的
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(1)  # [n_points, 1]

        # 确保residuals是1维的
        if resid.dim() > 1:
            resid = resid.squeeze()

        current_dim = x_val.shape[1]
        current_n_points = x_val.shape[0]

        max_dim = max(max_dim, current_dim)
        max_n_points = max(max_n_points, current_n_points)
        original_dims.append(current_dim)

    # Padding所有数据到最大形状
    x_values_padded = []
    residuals_padded = []
    dim_masks = []
    point_masks = []

    for i, item in enumerate(batch):
        x_val = item['x_values'].clone()  # [n_points, current_dim]
        resid = item['residuals'].clone()  # [n_points]

        # 确保x_values至少是2维的
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(1)

        # 确保residuals是1维的
        if resid.dim() > 1:
            resid = resid.squeeze()

        current_n_points = x_val.shape[0]
        current_dim = x_val.shape[1]

        # Padding n_points维度（如果需要）
        if current_n_points < max_n_points:
            padding_points = torch.zeros(max_n_points - current_n_points, x_val.shape[1], dtype=x_val.dtype)
            x_val = torch.cat([x_val, padding_points], dim=0)

            padding_resid = torch.zeros(max_n_points - current_n_points, dtype=resid.dtype)
            resid = torch.cat([resid, padding_resid], dim=0)

        # Padding dim维度
        if current_dim < max_dim:
            padding_dim = torch.zeros(x_val.shape[0], max_dim - current_dim, dtype=x_val.dtype)
            x_val = torch.cat([x_val, padding_dim], dim=1)

        # 创建维度mask：1表示有效维度，0表示padding
        dim_mask = torch.zeros(max_dim, dtype=torch.float32)
        dim_mask[:current_dim] = 1.0

        # 创建点mask：1表示真实点，0表示填充点
        point_mask = torch.zeros(max_n_points, dtype=torch.float32)
        point_mask[:current_n_points] = 1.0

        x_values_padded.append(x_val)          # [max_n_points, max_dim]
        residuals_padded.append(resid)         # [max_n_points]
        dim_masks.append(dim_mask)             # [max_dim]
        point_masks.append(point_mask)         # [max_n_points]

    result = {
        'x_values': torch.stack(x_values_padded),     # [batch_size, max_n_points, max_dim]
        'residuals': torch.stack(residuals_padded),   # [batch_size, max_n_points]
        'dim_mask': torch.stack(dim_masks),           # [batch_size, max_dim]
        'point_mask': torch.stack(point_masks),       # [batch_size, max_n_points]
        'z0_token_ids': torch.stack([item['z0_token_ids'] for item in batch]),
        'z1_token_ids': torch.stack([item['z1_token_ids'] for item in batch]),
        'gap_token': batch[0]['gap_token'],
        'original_dims': original_dims  # 记录原始维度，供调试使用
    }

    return result



"""
EditFlow连续流匹配的核心组件
"""

import torch
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from ..symbolic.data_generator import generate_flow_samples
# from ..utils.special_tokens import SpecialTokensManager  # 已移除：使用小词表后不再需要


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



def remove_gap_tokens(z_t: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列

    重要：此函数只移除 gap_token，保留 BOS token 和所有其他 tokens
    确保返回的序列格式为 [BOS] + [non_gap_tokens] + [PAD...]
    """
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    gap_token_id = tokenizer.convert_tokens_to_ids('<gap>')
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
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
                           gap_token: int, tokenizer) -> torch.Tensor:
        batch_size, z_seq_len = z_t.shape
        n_ops = 2 * vocab_size + 1

        pad_token = tokenizer.convert_tokens_to_ids('<pad>')

        z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
        z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq
        z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq
        z_sub = z_neq & ~z_ins & ~z_del

        u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
        u_mask[z_ins, z_1[z_ins]] = True
        u_mask[z_sub, z_1[z_sub] + vocab_size] = True
        u_mask[:, :, -1][z_del] = True

        return u_mask

    def __call__(self, u_cat_x: torch.Tensor, u_z: torch.Tensor, u_mask: torch.Tensor,
                 vocab_size: int, accelerator=None, logger=None) -> torch.Tensor:
        """
        连续流损失计算

        Args:
            u_cat_x: X空间的预测速率 [batch, x_seq_len, 2*vocab_size+1]（原始空间，不含gap）
            u_z: Z空间的预测速率 [batch, z_seq_len, 2*vocab_size+1]（扩展空间，含gap重复）
            u_mask: 操作掩码 [batch, z_seq_len, 2*vocab_size+1]
            vocab_size: 词汇表大小
            accelerator: Accelerate加速器
            logger: 日志记录器（可选）

        Returns:
            loss: 标量损失值
        """
        # NaN/Inf检测 - 检测两个输入
        nan_count_x = torch.isnan(u_cat_x).sum().item()
        inf_count_x = torch.isinf(u_cat_x).sum().item()
        nan_count_z = torch.isnan(u_z).sum().item()
        inf_count_z = torch.isinf(u_z).sum().item()

        # 额外检查：u_z 是否包含接近0的值（可能导致 log(0)）
        u_z_min = float(u_z.min().item())
        u_z_max = float(u_z.max().item())
        u_z_mean = float(u_z.mean().item())
        u_z_std = float(u_z.std().item())
        u_z_has_near_zero = bool((u_z < 1e-10).any().item())
        u_z_has_negative = bool((u_z < 0).any().item())
        u_z_num_zeros = int((u_z == 0).sum().item())

        # 分布式NaN检测：如果使用accelerator，确保所有进程同步检测结果
        has_nan_or_inf = torch.tensor(1 if ((nan_count_x > 0 or inf_count_x > 0) or (nan_count_z > 0 or inf_count_z > 0)) else 0,
                                    device=u_cat_x.device)

        if accelerator is not None and accelerator.distributed_type != "NO":
            # 同步所有进程的NaN检测结果
            # 使用accelerator.gather收集所有进程的NaN检测结果
            gathered_results = accelerator.gather(has_nan_or_inf)
            has_nan_or_inf = gathered_results.sum()
        else:
            has_nan_or_inf = has_nan_or_inf.item()

        # 记录NaN/Inf检测结果（仅用于监控，不再抛出异常跳过）
        if has_nan_or_inf > 0 and logger is not None and accelerator.is_local_main_process:
            logger.error("INPUT_NAN_INF",
                        f"检测到异常值: X空间NaN={nan_count_x}, Inf={inf_count_x}, "
                        f"Z空间NaN={nan_count_z}, Inf={inf_count_z}, 分布式检测={has_nan_or_inf}",
                        "compute_loss", level=1)

        # 关键修复：u_total 在 X 空间计算（原始序列空间，无gap重复）
        # u_cat_x 形状: [batch, x_seq_len, 2*vocab_size+1]
        # 这确保了每个位置的速率只被计算一次，不会因gap重复而被重复计数
        u_total = u_cat_x.sum(dim=(1, 2))

        # 归一化 u_z 使其成为有效的概率分布
        # u_z 形状: [batch, z_seq_len, 2*vocab_size+1]
        # 使用 logsumexp 技巧提高数值稳定性
        u_z_max_for_softmax = u_z.max(dim=-1, keepdim=True)[0]  # [batch, z_seq_len, 1]
        u_z_stable = u_z - u_z_max_for_softmax  # 减去最大值防止溢出
        u_z_sum_stable = torch.exp(u_z_stable).sum(dim=-1, keepdim=True) + 1e-8
        log_u_z = u_z_stable - torch.log(u_z_sum_stable)  # log_softmax

        # 统计信息：log_u_z（应该在负无穷到0之间）
        log_u_z_min = float(log_u_z.min().item())
        log_u_z_max = float(log_u_z.max().item())
        log_u_z_mean = float(log_u_z.mean().item())
        log_u_z_has_inf = bool(torch.isinf(log_u_z).any().item())
        log_u_z_has_nan = bool(torch.isnan(log_u_z).any().item())

        # 统计信息：u_mask（标记需要预测的位置）
        u_mask_num_true = int(u_mask.sum().item())
        u_mask_total = int(u_mask.numel())
        u_mask_sparsity = float(1.0 - (u_mask_num_true / u_mask_total))

        # cross_entropy 在 Z 空间计算（负对数似然）
        # u_mask 标记了正确的操作位置（one-hot编码）
        # 只在 u_mask=True 的位置累加（其他位置不影响损失）
        masked_log_u_z = log_u_z * u_mask.float()
        cross_entropy = masked_log_u_z.sum(dim=(1, 2))

        # 统计信息：cross_entropy（负对数似然，应该是负数）
        cross_entropy_min = float(cross_entropy.min().item())
        cross_entropy_max = float(cross_entropy.max().item())
        cross_entropy_mean = float(cross_entropy.mean().item())
        cross_entropy_std = float(cross_entropy.std().item() if cross_entropy.numel() > 1 else 0.0)

        # 最终损失：负对数似然（要最小化）
        # 不再使用 u_total 和 sched_coeff
        loss = -cross_entropy.mean()

        # 统计信息：loss
        loss_value = float(loss.item())
        loss_is_nan = bool(torch.isnan(loss).item())
        loss_is_inf = bool(torch.isinf(loss).item())

        # 记录所有统计信息到日志
        if logger is not None:
            logger.log(f"LOSS_STATS",
                      f"u_z: min={u_z_min:.6f}, max={u_z_max:.6f}, mean={u_z_mean:.6f}, std={u_z_std:.6f} | "
                      f"zeros={u_z_num_zeros}, near_zero={u_z_has_near_zero}, negative={u_z_has_negative} | "
                      f"log_u_z: min={log_u_z_min:.6f}, max={log_u_z_max:.6f}, mean={log_u_z_mean:.6f} | "
                      f"has_inf={log_u_z_has_inf}, has_nan={log_u_z_has_nan} | "
                      f"u_mask: {u_mask_num_true}/{u_mask_total} ({(1-u_mask_sparsity)*100:.2f}%) | "
                      f"cross_entropy: min={cross_entropy_min:.6f}, max={cross_entropy_max:.6f}, "
                      f"mean={cross_entropy_mean:.6f}, std={cross_entropy_std:.6f} | "
                      f"loss: {loss_value:.6f}, is_nan={loss_is_nan}, is_inf={loss_is_inf}",
                      level=2)

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

        # 直接使用tokenizer获取所需信息
        self.vocab_size = len(tokenizer.get_vocab())
        self.pad_token = tokenizer.convert_tokens_to_ids('<pad>')
        self.bos_token = tokenizer.convert_tokens_to_ids('<s>')
        self.gap_token = tokenizer.convert_tokens_to_ids('<gap>')

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
        # 直接使用tokenizer转换tokens
        return self.tokenizer.convert_tokens_to_ids(tokens)

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
        y_target = torch.FloatTensor(sample['y_target'])  # 修改：加载y_target而非residuals
        residuals = torch.FloatTensor(sample['residuals'])  # 保留residuals用于其他用途

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
            'y_target': y_target,  # 新增：返回y_target
            'residuals': residuals,  # 保留：用于向后兼容或其他用途
            'z0_token_ids': pad_z_sequence(z0_tokens),
            'z1_token_ids': pad_z_sequence(z1_tokens),
            'gap_token': self.gap_token
        }


def custom_collate_fn(batch):
    """处理不同维度数据的collate函数，使用padding + mask方案

    修改：添加y_target的处理（架构改进：使用目标值而非残差作为条件）
    """
    if len(batch) == 0:
        return {
            'x_values': torch.empty(0, 0, 0),
            'y_target': torch.empty(0, 0),  # 修改：添加y_target
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
        y_tgt = item['y_target']  # [n_points]  # 新增：获取y_target

        # 确保x_values至少是2维的
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(1)  # [n_points, 1]

        # 确保y_target是1维的
        if y_tgt.dim() > 1:
            y_tgt = y_tgt.squeeze()

        current_dim = x_val.shape[1]
        current_n_points = x_val.shape[0]

        max_dim = max(max_dim, current_dim)
        max_n_points = max(max_n_points, current_n_points)
        original_dims.append(current_dim)

    # Padding所有数据到最大形状
    x_values_padded = []
    y_target_padded = []  # 新增：y_target的padding
    residuals_padded = []
    dim_masks = []
    point_masks = []

    for i, item in enumerate(batch):
        x_val = item['x_values'].clone()  # [n_points, current_dim]
        y_tgt = item['y_target'].clone()  # [n_points]  # 新增：克隆y_target
        resid = item['residuals'].clone()  # [n_points]  # 保留：用于向后兼容

        # 确保x_values至少是2维的
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(1)

        # 确保y_target和residuals是1维的
        if y_tgt.dim() > 1:
            y_tgt = y_tgt.squeeze()
        if resid.dim() > 1:
            resid = resid.squeeze()

        current_n_points = x_val.shape[0]
        current_dim = x_val.shape[1]

        # Padding n_points维度（如果需要）
        if current_n_points < max_n_points:
            padding_points = torch.zeros(max_n_points - current_n_points, x_val.shape[1], dtype=x_val.dtype)
            x_val = torch.cat([x_val, padding_points], dim=0)

            padding_y_tgt = torch.zeros(max_n_points - current_n_points, dtype=y_tgt.dtype)
            y_tgt = torch.cat([y_tgt, padding_y_tgt], dim=0)

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
        y_target_padded.append(y_tgt)          # [max_n_points]  # 新增
        residuals_padded.append(resid)         # [max_n_points]
        dim_masks.append(dim_mask)             # [max_dim]
        point_masks.append(point_mask)         # [max_n_points]

    result = {
        'x_values': torch.stack(x_values_padded),       # [batch_size, max_n_points, max_dim]
        'y_target': torch.stack(y_target_padded),       # [batch_size, max_n_points]  # 新增
        'residuals': torch.stack(residuals_padded),     # [batch_size, max_n_points]  # 保留
        'dim_mask': torch.stack(dim_masks),             # [batch_size, max_dim]
        'point_mask': torch.stack(point_masks),         # [batch_size, max_n_points]
        'z0_token_ids': torch.stack([item['z0_token_ids'] for item in batch]),
        'z1_token_ids': torch.stack([item['z1_token_ids'] for item in batch]),
        'gap_token': batch[0]['gap_token'],
        'original_dims': original_dims  # 记录原始维度，供调试使用
    }

    return result



"""
Edit Flows 概率插值机制
实现从 (z_0, z_1) 和时间步 t 生成中间样本 z_t
"""

import torch
import torch.nn.functional as A


class KappaScheduler:
    """时间调度器基类"""
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    """三次调度器：κ(t) = -2t³ + 3t²

    边界条件：κ(0)=0, κ(1)=1，边界处导数为0
    """
    def __init__(self, a: float = 0.0, b: float = 0.0):
        self.a = a
        self.b = b

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return -2 * (t**3) + 3 * (t**2) + self.a * (t**3 - 2*t**2 + t) + self.b * (t**3 - t**2)


def tokens_to_probs(z_token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """将 token IDs 转换为 one-hot 概率分布

    Args:
        z_token_ids: token IDs [batch_size, seq_len]
        vocab_size: 词汇表大小

    Returns:
        probs: one-hot 概率分布 [batch_size, seq_len, vocab_size]
    """
    probs = A.one_hot(z_token_ids, vocab_size).float()
    return probs


def sample_conditional_pt(
    p0: torch.Tensor,
    p1: torch.Tensor,
    t: torch.Tensor,
    kappa_scheduler: KappaScheduler
) -> torch.Tensor:
    """在概率空间插值并采样

    Args:
        p0: z_0 的概率分布 [batch_size, seq_len, vocab_size]
        p1: z_1 的概率分布 [batch_size, seq_len, vocab_size]
        t: 时间步 [batch_size] 或 [batch_size, 1]
        kappa_scheduler: 时间调度器

    Returns:
        z_t: 采样的 token IDs [batch_size, seq_len]
    """
    # 确保 t 的形状正确
    if t.dim() == 1:
        t = t.reshape(-1, 1, 1)
    elif t.dim() == 2:
        t = t.reshape(-1, 1, 1)

    # 计算插值系数
    kappa_t = kappa_scheduler(t)

    # 在概率空间插值：p_t = (1 - κ(t)) * p_0 + κ(t) * p_1
    p_t = (1 - kappa_t) * p0 + kappa_t * p1

    # 从 p_t 采样得到 z_t
    batch_size, seq_len, vocab_size = p_t.shape
    p_t_flat = p_t.reshape(batch_size * seq_len, vocab_size)
    z_t_flat = torch.multinomial(p_t_flat, 1).squeeze(-1)
    z_t = z_t_flat.reshape(batch_size, seq_len)
    return z_t


def interpolate_z_to_zt(
    z0_token_ids: torch.Tensor,
    z1_token_ids: torch.Tensor,
    t: torch.Tensor,
    vocab_size: int,
    kappa_scheduler: KappaScheduler
) -> torch.Tensor:
    """从 (z_0, z_1) 和时间步 t 生成中间样本 z_t

    Args:
        z0_token_ids: 初始状态 token IDs [batch_size, seq_len]
        z1_token_ids: 目标状态 token IDs [batch_size, seq_len]
        t: 时间步 [batch_size]
        vocab_size: 词汇表大小
        kappa_scheduler: 时间调度器

    Returns:
        z_t: 中间状态 token IDs [batch_size, seq_len]
    """
    p0 = tokens_to_probs(z0_token_ids, vocab_size)
    p1 = tokens_to_probs(z1_token_ids, vocab_size)
    z_t = sample_conditional_pt(p0, p1, t, kappa_scheduler)
    return z_t

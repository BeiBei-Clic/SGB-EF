"""
时间步嵌入模块 - 用于扩散模型和流匹配模型
基于 Sinusoidal Embedding 和 AdaLN (Adaptive Layer Normalization)
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步嵌入

    使用不同频率的正弦/余弦函数编码时间步信息，与 RoPE 位置编码兼容。
    RoPE 编码序列位置，时间步嵌入编码训练进度，两者正交。

    Args:
        embedding_dim: 嵌入维度
        max_period: 最大周期，控制频率范围 (默认 10000.0，与 Transformer 一致)
    """

    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # 预计算频率（一半用于 sin，一半用于 cos）
        half_dim = embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: (batch_size,) 时间步，范围 [0, 1]

        Returns:
            embedding: (batch_size, embedding_dim) 时间步嵌入
        """
        batch_size = timestep.shape[0]
        device = timestep.device

        # 扩展维度用于广播: (batch_size, half_dim)
        t = timestep.to(device).unsqueeze(-1) * self.freqs.unsqueeze(0)

        # 交替使用 sin 和 cos
        embedding = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

        # 处理奇数维度情况
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class AdaLNModulator(nn.Module):
    """自适应层归一化调制器

    通过时间步调节层归一化参数 (scale 和 shift)。

    Args:
        time_embed_dim: 时间步嵌入维度
        hidden_dim: 需要调制的隐藏层维度
    """

    def __init__(self, time_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim

        # MLP: SiLU -> Linear(time_embed_dim, hidden_dim * 2)
        # 输出 scale 和 shift 参数
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim * 2)
        )

        # 初始化为接近恒等映射 (scale≈0, shift≈0)
        # 这样训练初期时间步信息影响较小，避免不稳定
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, time_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            time_embed: (batch_size, time_embed_dim) 时间步嵌入

        Returns:
            scale: (batch_size, hidden_dim) 缩放参数
            shift: (batch_size, hidden_dim) 平移参数
        """
        # (batch_size, hidden_dim * 2)
        modulation_params = self.mlp(time_embed)

        # 分割为 scale 和 shift
        scale, shift = modulation_params.chunk(2, dim=-1)

        return scale, shift

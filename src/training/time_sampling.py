"""
时间步采样器 - 用于训练时从时间步分布中采样
支持多种采样策略：uniform、discrete、importance
"""

import torch
import numpy as np


class TimestepSampler:
    """时间步采样器

    从 [0, 1] 范围内采样时间步，用于训练时模拟不同阶段的去噪过程。

    Args:
        sampling_strategy: 采样策略
            - "uniform": 均匀采样 [0, 1]
            - "discrete": 离散均匀采样（从预定义的网格点）
            - "importance": 重要性采样（优先采样中等时间步）
        num_discrete_timesteps: 离散采样的时间步数量
    """

    def __init__(
        self,
        sampling_strategy: str = "uniform",
        num_discrete_timesteps: int = 10
    ):
        self.sampling_strategy = sampling_strategy
        self.num_discrete_timesteps = num_discrete_timesteps

        # 预计算离散时间步网格
        if sampling_strategy == "discrete":
            self.discrete_timesteps = torch.linspace(0, 1, num_discrete_timesteps)

    def sample(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        采样时间步

        Args:
            batch_size: 批次大小
            device: 设备

        Returns:
            timestep: (batch_size,) 采样得到的时间步，范围 [0, 1]
        """
        if self.sampling_strategy == "uniform":
            # 均匀采样 [0, 1]
            timestep = torch.rand(batch_size, device=device)

        elif self.sampling_strategy == "discrete":
            # 从预定义的离散网格点均匀采样
            indices = torch.randint(0, self.num_discrete_timesteps, (batch_size,), device=device)
            # 确保 discrete_timesteps 在正确的设备上
            discrete_timesteps_device = self.discrete_timesteps.to(device)
            timestep = discrete_timesteps_device[indices]

        elif self.sampling_strategy == "importance":
            # 重要性采样：优先采样中等时间步
            # 使用 Beta(2, 2) 分布，在中间区域有更高的概率密度
            timestep = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(device)

        else:
            raise ValueError(f"未知的采样策略: {self.sampling_strategy}")

        return timestep

    def get_timesteps_for_inference(
        self,
        num_steps: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        获取推理时使用的时间步序列（从 1 递减到 0）

        Args:
            num_steps: 推理步数
            device: 设备

        Returns:
            timesteps: (num_steps,) 时间步序列，从 1 递减到 0
        """
        return torch.linspace(1.0, 0.0, num_steps, device=device)

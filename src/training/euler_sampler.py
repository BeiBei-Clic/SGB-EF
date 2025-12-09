"""
多步Euler采样器 - 用于EditFlow连续流模型的推理
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

from .editflow_manager import KappaScheduler, remove_gap_tokens, fill_gap_tokens_with_repeats
from ..utils.special_tokens import SpecialTokensManager


class EulerSampler:
    """基于Euler方法的连续流采样器"""

    def __init__(self, model, condition_encoder, tokenizer, device='cuda',
                 scheduler_type='cubic', apply_condition=True):
        self.model = model
        self.condition_encoder = condition_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.scheduler = KappaScheduler(scheduler_type)
        self.apply_condition = apply_condition

        # 获取模型配置
        if hasattr(model, 'module'):
            self.config = model.module.config
        else:
            self.config = model.config

        # 特殊token
        self.pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)
        self.gap_token = self.special_tokens_manager.get_gap_token_id()

    def get_adaptive_step_size(self, h: float, t: torch.Tensor) -> torch.Tensor:
        """自适应步长，避免超过t=1"""
        coeff = (1 - self.scheduler(t)) / self.scheduler.derivative(t)
        _h = torch.full_like(t, h)
        h_adapt = torch.minimum(_h, coeff)
        return h_adapt

    def apply_edit_operations(self, x_t: torch.Tensor, ins_mask: torch.Tensor,
                            del_mask: torch.Tensor, sub_mask: torch.Tensor,
                            ins_tokens: torch.Tensor, sub_tokens: torch.Tensor) -> torch.Tensor:
        """应用编辑操作（插入、删除、替换）"""
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # 处理同时插入和删除的情况（替换）
        replace_mask = ins_mask & del_mask
        x_t_modified = x_t.clone()
        x_t_modified[replace_mask] = sub_tokens[replace_mask]

        # 更新插入和删除mask
        eff_ins_mask = ins_mask & ~replace_mask
        eff_del_mask = del_mask & ~replace_mask

        # 计算新长度
        xt_pad_mask = (x_t == self.pad_token)
        xt_seq_lens = (~xt_pad_mask).sum(dim=1)
        new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
        max_new_len = int(new_lengths.max().item())

        if max_new_len <= 0:
            return torch.full((batch_size, 1), self.pad_token, dtype=torch.long, device=device)

        # 预分配结果
        x_new = torch.full((batch_size, max_new_len), self.pad_token, dtype=torch.long, device=device)

        # 处理位置映射
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        cum_del = torch.cumsum(eff_del_mask.float(), dim=1)
        cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)
        cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)

        # 放置非删除的tokens
        new_pos = pos_idx + cum_ins_before - cum_del
        keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
        if keep_mask.any():
            x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

        # 放置插入的tokens
        if eff_ins_mask.any():
            ins_pos = new_pos + 1
            ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)
            if ins_valid.any():
                x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]

        return x_new

    def sample_step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor,
                    condition: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一步Euler采样"""
        batch_size, seq_len = x_t.shape

        # 模型前向传播
        with torch.no_grad():
            attention_mask = (x_t != self.pad_token).float()
            pred_rates, pred_ins_probs, pred_sub_probs = self.model(
                input_ids=x_t,
                time_steps=t,
                attention_mask=attention_mask,
                condition=condition if self.apply_condition else None
            )

        # 提取操作率
        lambda_ins = pred_rates[:, :, 0]      # (batch_size, seq_len)
        lambda_sub = pred_rates[:, :, 1]      # (batch_size, seq_len)
        lambda_del = pred_rates[:, :, 2]      # (batch_size, seq_len)

        # 自适应步长
        h_adapt = self.get_adaptive_step_size(h, t)

        # 采样操作mask
        ins_mask = torch.rand(lambda_ins.shape, device=lambda_ins.device) < (1 - torch.exp(-h_adapt * lambda_ins))
        del_sub_mask = torch.rand(lambda_sub.shape, device=lambda_sub.device) < (1 - torch.exp(-h_adapt * (lambda_sub + lambda_del)))

        # 分离删除和替换操作
        prob_del = torch.where(del_sub_mask, lambda_del / (lambda_sub + lambda_del + 1e-8), torch.zeros_like(lambda_del))
        del_mask = torch.bernoulli(prob_del).bool()
        sub_mask = del_sub_mask & ~del_mask

        # 采样插入和替换的tokens
        ins_tokens = torch.full(ins_probs.shape[:2], self.pad_token, dtype=torch.long)
        sub_tokens = torch.full(sub_probs.shape[:2], self.pad_token, dtype=torch.long)

        non_pad_mask = (x_t != self.pad_token)
        if non_pad_mask.any():
            ins_sampled = torch.multinomial(pred_ins_probs[non_pad_mask], 1).squeeze(-1)
            sub_sampled = torch.multinomial(pred_sub_probs[non_pad_mask], 1).squeeze(-1)
            ins_tokens[non_pad_mask] = ins_sampled
            sub_tokens[non_pad_mask] = sub_sampled

        # 应用编辑操作
        x_t = self.apply_edit_operations(x_t, ins_mask, del_mask, sub_mask, ins_tokens, sub_tokens)

        return x_t, t + h_adapt

    def sample(self, x_0: torch.Tensor, condition: Optional[torch.Tensor] = None,
               n_steps: int = 1000, h: Optional[float] = None, t_min: float = 0.0) -> torch.Tensor:
        """
        执行完整的Euler采样过程

        Args:
            x_0: 初始序列 (batch_size, seq_len)
            condition: 条件编码 (batch_size, condition_dim)
            n_steps: 采样步数
            h: 步长，如果为None则使用1/n_steps
            t_min: 起始时间

        Returns:
            采样的序列 (batch_size, final_seq_len)
        """
        self.model.eval()
        self.condition_encoder.eval()

        if h is None:
            h = 1.0 / n_steps

        x_t = x_0.clone()
        t = torch.full((x_t.size(0), 1), t_min, device=self.device)

        with tqdm(desc="Euler Sampling", total=n_steps) as pbar:
            while t.max() < 1.0:
                x_t, t = self.sample_step(x_t, t, h, condition)
                pbar.update(1)

                # 提前停止条件
                if t.min() >= 1.0:
                    break

        return x_t

    def generate_initial_sequence(self, batch_size: int, seq_len: int = 64,
                                 condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成初始序列"""
        # 简单实现：填充BOS token和随机tokens
        x_0 = torch.full((batch_size, seq_len), self.pad_token, dtype=torch.long, device=self.device)
        x_0[:, 0] = self.bos_token

        # 可以根据条件生成更智能的初始序列
        # 这里使用随机tokens作为示例
        if self.config.vocab_size > 3:
            x_0[:, 1:] = torch.randint(
                low=3, high=min(self.config.vocab_size, 20),
                size=(batch_size, seq_len - 1),
                device=self.device
            )

        return x_0

    def sample_from_condition(self, condition: torch.Tensor, batch_size: int = 1,
                            seq_len: int = 64, n_steps: int = 1000) -> torch.Tensor:
        """从条件编码采样"""
        # 生成初始序列
        x_0 = self.generate_initial_sequence(batch_size, seq_len, condition)

        # 执行采样
        return self.sample(x_0, condition, n_steps)
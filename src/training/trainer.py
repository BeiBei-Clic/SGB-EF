"""
EditFlow符号回归训练器
实现Discrete Flow Matching损失函数和训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

from ..symbolic.data_generator import SymbolicRegressionDataGenerator, SymbolicVocabulary
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer


class KappaScheduler:
    """调度系数kappa(t)，用于插值"""

    def __init__(self, scheduler_type: str = "cubic", a: float = 1.0, b: float = 1.0):
        self.scheduler_type = scheduler_type
        self.a = a
        self.b = b

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """计算kappa(t)"""
        if self.scheduler_type == "linear":
            return t
        elif self.scheduler_type == "cubic":
            return 3 * t**2 - 2 * t**3
        elif self.scheduler_type == "cosine":
            return 0.5 * (1 - torch.cos(torch.pi * t))
        else:
            return t

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """计算kappa'(t)"""
        if self.scheduler_type == "linear":
            return torch.ones_like(t)
        elif self.scheduler_type == "cubic":
            return 6 * t - 6 * t**2
        elif self.scheduler_type == "cosine":
            return 0.5 * torch.pi * torch.sin(torch.pi * t)
        else:
            return torch.ones_like(t)


class EditFlowLoss(nn.Module):
    """EditFlow损失函数 - 基于Bregman散度"""

    def __init__(self, scheduler: KappaScheduler):
        super().__init__()
        self.scheduler = scheduler

    def forward(self, rates: torch.Tensor, insert_probs: torch.Tensor,
                substitute_probs: torch.Tensor, target_rates: torch.Tensor,
                time_steps: torch.Tensor) -> torch.Tensor:
        """
        计算EditFlow损失
        Args:
            rates: (batch_size, seq_len, 3) 预测的编辑速率
            insert_probs: (batch_size, seq_len, vocab_size) 插入概率
            substitute_probs: (batch_size, seq_len, vocab_size) 替换概率
            target_rates: (batch_size, seq_len, 2*vocab_size+1) 目标速率
            time_steps: (batch_size, 1) 时间步
        Returns:
            loss: 损失值
        """
        batch_size, seq_len, _ = rates.shape

        # 计算调度系数
        kappa_t = self.scheduler(time_steps)
        kappa_prime_t = self.scheduler.derivative(time_steps)

        # 计算总速率
        total_rates = rates.sum(dim=-1)  # (batch_size, seq_len)

        # 构建完整的编辑速率向量
        # 插入速率 * 插入概率 + 替换速率 * 替换概率 + 删除速率
        insert_rates = rates[:, :, 0:1] * insert_probs  # (batch_size, seq_len, vocab_size)
        substitute_rates = rates[:, :, 1:2] * substitute_probs  # (batch_size, seq_len, vocab_size)
        delete_rates = rates[:, :, 2:3]  # (batch_size, seq_len, 1)

        # 拼接所有编辑操作
        edit_rates = torch.cat([insert_rates, substitute_rates, delete_rates], dim=-1)
        # (batch_size, seq_len, 2*vocab_size+1)

        # 计算Bregman散度损失
        # 使用交叉熵近似
        edit_rates = torch.clamp(edit_rates, min=1e-8)
        log_edit_rates = torch.log(edit_rates)

        # 目标掩码（哪些位置需要编辑）
        target_mask = (target_rates > 0).float()

        # 计算加权交叉熵
        cross_entropy_loss = -(target_rates * log_edit_rates).sum(dim=-1)  # (batch_size, seq_len)

        # 应用调度系数
        weighted_loss = cross_entropy_loss * (kappa_prime_t / (1 - kappa_t + 1e-8)).squeeze(-1)

        # 总损失 = 总速率 - 加权交叉熵
        loss = (total_rates * target_mask.sum(dim=-1) - weighted_loss).sum(dim=-1).mean()

        return loss


class EditFlowTrainer:
    """EditFlow训练器"""

    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_seq_len: int = 512,
                 learning_rate: float = 1e-4,
                 device: str = "auto",
                 condition_model_name: str = "distilbert-base-uncased"):

        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 词汇表
        self.vocab = SymbolicVocabulary()
        self.vocab_size = vocab_size

        # 条件编码器
        self.condition_encoder = ConditionEncoder(model_name=condition_model_name).to(self.device)

        # EditFlow模型
        self.model = EditFlowTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            condition_dim=self.condition_encoder.output_dim
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.condition_encoder.parameters()),
            lr=learning_rate
        )

        # 调度器
        self.scheduler = KappaScheduler("cubic")

        # 损失函数
        self.criterion = EditFlowLoss(self.scheduler)

        # 训练指标
        self.metrics = {
            "loss": [],
            "insert_rate": [],
            "substitute_rate": [],
            "delete_rate": [],
            "total_rate": []
        }

    def prepare_batch(self, batch: List) -> Tuple:
        """准备训练批次数据"""
        # 解包批次数据
        curr_expressions, target_expressions, residuals, alignments = zip(*batch)

        # 准备输入数据
        batch_size = len(batch)

        # 转换token IDs
        curr_token_ids = []
        target_token_ids = []

        max_len = 0
        for curr_expr, target_expr in zip(curr_expressions, target_expressions):
            curr_ids = [self.vocab.token_to_id.get(token, self.vocab.pad_id) for token in curr_expr.tokens]
            target_ids = [self.vocab.token_to_id.get(token, self.vocab.pad_id) for token in target_expr.tokens]

            curr_token_ids.append(curr_ids)
            target_token_ids.append(target_ids)
            max_len = max(max_len, max(len(curr_ids), len(target_ids)))

        # 填充到相同长度
        padded_curr = []
        padded_target = []
        attention_masks = []

        for curr_ids, target_ids in zip(curr_token_ids, target_token_ids):
            # 填充当前表达式
            curr_padded = curr_ids + [self.vocab.pad_id] * (max_len - len(curr_ids))
            padded_curr.append(curr_padded)

            # 填充目标表达式
            target_padded = target_ids + [self.vocab.pad_id] * (max_len - len(target_ids))
            padded_target.append(target_padded)

            # 注意力掩码
            mask = [1 if token != self.vocab.pad_id else 0 for token in curr_padded]
            attention_masks.append(mask)

        # 转换为tensor
        curr_ids_tensor = torch.tensor(padded_curr, dtype=torch.long).to(self.device)
        target_ids_tensor = torch.tensor(padded_target, dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(self.device)

        # 准备残差数据
        residual_tensors = []
        x_tensors = []

        for i, residual in enumerate(residuals):
            # 假设residual是numpy数组
            residual_tensor = torch.tensor(residual, dtype=torch.float32).to(self.device)
            x_values = torch.linspace(-5, 5, len(residual), dtype=torch.float32).to(self.device)

            residual_tensors.append(residual_tensor)
            x_tensors.append(x_values)

        # 填充残差到相同长度
        max_residual_len = max(len(r) for r in residual_tensors)
        padded_residuals = []
        padded_x_values = []

        for residual_tensor, x_tensor in zip(residual_tensors, x_tensors):
            # 填充
            if len(residual_tensor) < max_residual_len:
                pad_size = max_residual_len - len(residual_tensor)
                residual_padded = F.pad(residual_tensor, (0, pad_size))
                x_padded = F.pad(x_tensor, (0, pad_size))
            else:
                residual_padded = residual_tensor
                x_padded = x_tensor

            padded_residuals.append(residual_padded)
            padded_x_values.append(x_padded)

        # 堆叠
        residuals_batch = torch.stack(padded_residuals)  # (batch_size, max_len)
        x_values_batch = torch.stack(padded_x_values)  # (batch_size, max_len)

        # 时间步
        time_steps = torch.rand(batch_size, 1).to(self.device)

        return (curr_ids_tensor, target_ids_tensor, attention_mask_tensor,
                time_steps, residuals_batch, x_values_batch, alignments)

    def compute_target_rates(self, alignments: List[List[str]], seq_len: int) -> torch.Tensor:
        """计算目标编辑速率"""
        batch_size = len(alignments)
        target_rates = torch.zeros(batch_size, seq_len, 2 * self.vocab_size + 1).to(self.device)

        for batch_idx, alignment in enumerate(alignments):
            for pos, op in enumerate(alignment):
                if pos >= seq_len:
                    break

                if op.startswith('ins:'):
                    # 插入操作
                    token = op[4:]  # 提取token
                    if token in self.vocab.token_to_id:
                        token_id = self.vocab.token_to_id[token]
                        target_rates[batch_idx, pos, token_id] = 1.0
                elif op.startswith('sub:'):
                    # 替换操作
                    parts = op.split('->')
                    if len(parts) == 2:
                        target_token = parts[1]
                        if target_token in self.vocab.token_to_id:
                            token_id = self.vocab.token_to_id[target_token]
                            target_rates[batch_idx, pos, self.vocab_size + token_id] = 1.0
                elif op.startswith('del:'):
                    # 删除操作
                    target_rates[batch_idx, pos, -1] = 1.0

        return target_rates

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.condition_encoder.train()

        epoch_metrics = {key: [] for key in self.metrics}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # 准备数据
            (curr_ids, target_ids, attention_mask, time_steps,
             residuals, x_values, alignments) = self.prepare_batch(batch)

            # 编码条件
            condition = self.condition_encoder(x_values, residuals)

            # 前向传播
            rates, insert_probs, substitute_probs = self.model(
                curr_ids, time_steps, condition, attention_mask
            )

            # 计算目标速率
            target_rates = self.compute_target_rates(alignments, curr_ids.shape[1])

            # 计算损失
            loss = self.criterion(rates, insert_probs, substitute_probs,
                                target_rates, time_steps)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录指标
            with torch.no_grad():
                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["insert_rate"].append(rates[:, :, 0].mean().item())
                epoch_metrics["substitute_rate"].append(rates[:, :, 1].mean().item())
                epoch_metrics["delete_rate"].append(rates[:, :, 2].mean().item())
                epoch_metrics["total_rate"].append(rates.sum(dim=-1).mean().item())

        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}

        # 更新全局指标
        for key in avg_metrics:
            self.metrics[key].extend(epoch_metrics[key])

        return avg_metrics

    def train(self, dataloader: DataLoader, num_epochs: int, save_dir: str = "checkpoints"):
        """训练模型"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        for epoch in range(num_epochs):
            # 训练一个epoch
            avg_metrics = self.train_epoch(dataloader, epoch)

            # 打印指标
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Loss: {avg_metrics['loss']:.4f}")
            print(f"  Insert Rate: {avg_metrics['insert_rate']:.4f}")
            print(f"  Substitute Rate: {avg_metrics['substitute_rate']:.4f}")
            print(f"  Delete Rate: {avg_metrics['delete_rate']:.4f}")
            print(f"  Total Rate: {avg_metrics['total_rate']:.4f}")

            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch)

        print("训练完成！")

    def save_checkpoint(self, save_dir: str, epoch: int):
        """保存模型检查点"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'condition_encoder_state_dict': self.condition_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'vocab_size': self.vocab_size,
            'vocab': self.vocab.__dict__
        }

        torch.save(checkpoint, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt")
        print(f"检查点已保存: {save_dir}/checkpoint_epoch_{epoch+1}.pt")
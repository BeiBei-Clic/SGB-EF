"""
EditFlow Transformer - 用于符号回归的编辑流模型
基于Hugging Face Transformers组件实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Tuple, Optional
import math


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size, 1) 时间步
        Returns:
            time_emb: (batch_size, hidden_dim) 时间嵌入
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.hidden_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class CrossAttentionConditionInjection(nn.Module):
    """交叉注意力条件注入模块"""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 条件到键值的投影
        self.cond_to_k = nn.Linear(hidden_dim, hidden_dim)
        self.cond_to_v = nn.Linear(hidden_dim, hidden_dim)

        # 查询投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) 隐藏状态
            condition: (batch_size, hidden_dim) 条件向量
        Returns:
            output: (batch_size, seq_len, hidden_dim) 条件注入后的隐藏状态
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算查询
        q = self.q_proj(hidden_states)  # (batch_size, seq_len, hidden_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch_size, num_heads, seq_len, head_dim)

        # 计算键和值来自条件
        k = self.cond_to_k(condition)  # (batch_size, hidden_dim)
        v = self.cond_to_v(condition)  # (batch_size, hidden_dim)

        # 扩展条件以匹配序列长度
        k = k.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        v = v.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class EditFlowTransformerBlock(nn.Module):
    """EditFlow Transformer块"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 自注意力
        config = AutoConfig(
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout
        )

        # 使用transformer库的组件
        from transformers.models.bert.modeling_bert import BertSelfAttention, BertIntermediate, BertOutput

        self.self_attention = BertSelfAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # 条件注入
        self.condition_injection = CrossAttentionConditionInjection(hidden_dim, num_heads)

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            condition: (batch_size, hidden_dim)
            attention_mask: (batch_size, seq_len) 注意力掩码
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # 条件注入
        condition_output = self.condition_injection(hidden_states, condition)
        hidden_states = self.norm1(hidden_states + self.dropout(condition_output))

        # 自注意力
        attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        attention_output = attention_outputs[0]
        hidden_states = self.norm2(hidden_states + self.dropout(attention_output))

        # 前馈网络
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)
        hidden_states = self.norm3(hidden_states + self.dropout(layer_output))

        return hidden_states


class EditFlowTransformer(nn.Module):
    """EditFlow Transformer主模型"""

    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 pad_token_id: int = 0,
                 condition_dim: int = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # 条件维度（默认与hidden_dim相同）
        self.condition_dim = condition_dim or hidden_dim

        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # 时间嵌入
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 条件投影（将条件向量投影到模型维度）
        self.condition_projection = nn.Linear(self.condition_dim, hidden_dim)

        # Transformer层
        self.layers = nn.ModuleList([
            EditFlowTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出头：预测编辑操作
        # 输出3个速率：插入、替换、删除
        self.rates_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3个编辑操作的速率
        )

        # 插入操作的token概率分布
        self.insert_logits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)  # 词汇表大小
        )

        # 替换操作的token概率分布
        self.substitute_logits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)  # 词汇表大小
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, time_steps: torch.Tensor,
                condition: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple:
        """
        前向传播
        Args:
            input_ids: (batch_size, seq_len) 输入token IDs
            time_steps: (batch_size, 1) 时间步
            condition: (batch_size, condition_dim) 条件向量
            attention_mask: (batch_size, seq_len) 注意力掩码
        Returns:
            rates: (batch_size, seq_len, 3) 编辑速率
            insert_probs: (batch_size, seq_len, vocab_size) 插入概率
            substitute_probs: (batch_size, seq_len, vocab_size) 替换概率
        """
        batch_size, seq_len = input_ids.shape

        # 词嵌入
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, hidden_dim)

        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (batch_size, seq_len, hidden_dim)

        # 时间嵌入
        time_emb = self.time_embedding(time_steps)  # (batch_size, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)

        # 条件投影
        condition_proj = self.condition_projection(condition)  # (batch_size, hidden_dim)

        # 组合所有嵌入
        hidden_states = token_emb + pos_emb + time_emb

        # 通过Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, condition_proj, attention_mask)

        # 计算输出
        rates = F.softplus(self.rates_head(hidden_states))  # 确保正速率

        insert_logits = self.insert_logits_head(hidden_states)
        substitute_logits = self.substitute_logits_head(hidden_states)

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            rates = rates * mask
            insert_logits = insert_logits.masked_fill(~mask.bool(), -float('inf'))
            substitute_logits = substitute_logits.masked_fill(~mask.bool(), -float('inf'))

        # 计算概率
        insert_probs = F.softmax(insert_logits, dim=-1)
        substitute_probs = F.softmax(substitute_logits, dim=-1)

        return rates, insert_probs, substitute_probs


# 测试代码
if __name__ == "__main__":
    # 模型参数
    vocab_size = 1000
    hidden_dim = 256
    num_layers = 4
    num_heads = 8
    max_seq_len = 128
    batch_size = 2
    seq_len = 64

    # 创建模型
    model = EditFlowTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    # 测试数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    time_steps = torch.rand(batch_size, 1)
    condition = torch.randn(batch_size, 256)
    attention_mask = torch.ones(batch_size, seq_len)

    # 前向传播
    with torch.no_grad():
        rates, insert_probs, substitute_probs = model(
            input_ids, time_steps, condition, attention_mask
        )

    print(f"EditFlow Transformer测试:")
    print(f"输入形状: {input_ids.shape}")
    print(f"条件形状: {condition.shape}")
    print(f"速率输出形状: {rates.shape}")
    print(f"插入概率形状: {insert_probs.shape}")
    print(f"替换概率形状: {substitute_probs.shape}")
    print(f"速率范围: [{rates.min().item():.4f}, {rates.max().item():.4f}]")
    print(f"插入概率和: {insert_probs.sum(dim=-1).mean().item():.4f}")
    print(f"替换概率和: {substitute_probs.sum(dim=-1).mean().item():.4f}")

    print("EditFlow Transformer测试完成！")
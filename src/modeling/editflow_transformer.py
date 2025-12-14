"""
EditFlow Transformer - 基于Transformer的符号回归编辑流模型
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import math
from pathlib import Path


class EditFlowConfig:
    """EditFlow Transformer配置类 - 不继承PretrainedConfig以避免AutoModel系统冲突"""

    def __init__(self, max_seq_len=24, dropout=0.1, condition_dim=None,
                 base_model_name="google-bert/bert-base-uncased", use_condition_injection=True, vocab_size=None, **kwargs):
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.condition_dim = condition_dim  # 需要从外部设置
        self.base_model_name = base_model_name
        self.use_condition_injection = use_condition_injection
        self.vocab_size = vocab_size  # 词表大小，如果为None则从预训练模型获取

        # 这些参数将从预训练模型动态获取
        self.hidden_dim = None
        self.num_layers = None
        self.num_heads = None


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t):
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
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.cond_to_k = nn.Linear(hidden_dim, hidden_dim)
        self.cond_to_v = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states, condition):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k = self.cond_to_k(condition).unsqueeze(1).expand(-1, seq_len, -1)
        v = self.cond_to_v(condition).unsqueeze(1).expand(-1, seq_len, -1)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)


class EditFlowTransformer(nn.Module):
    """EditFlow Transformer - 基于Transformer的符号回归编辑流模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 加载预训练的transformer模型
        if hasattr(config, 'base_model_name') and config.base_model_name:
            # 设置缓存目录
            cache_dir = Path("models/huggingface_cache").resolve()
            os.makedirs(cache_dir, exist_ok=True)

            print(f"正在加载基础模型: {config.base_model_name}")
            print(f"模型缓存目录: {cache_dir}")

            # 加载预训练模型，自动适应架构
            self.base_model = AutoModel.from_pretrained(
                config.base_model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print(f"✓ 基础模型加载完成: {type(self.base_model)}")
            print(f"Base model class: {self.base_model.__class__}")

            # 自动获取模型配置
            self.model_config = self.base_model.config

            config.hidden_dim = getattr(self.model_config, 'hidden_size')
            config.num_heads = getattr(self.model_config, 'num_attention_heads')
            config.num_layers = getattr(self.model_config, 'num_hidden_layers')
        else:
            raise ValueError("必须指定base_model_name")

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # 条件处理
        self.condition_projection = nn.Linear(config.condition_dim, config.hidden_dim)
        if config.use_condition_injection:
            self.condition_injection = CrossAttentionConditionInjection(config.hidden_dim, config.num_heads)

        # 输出头
        self.rates_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 3)
        )

        self.insert_logits_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )

        self.substitute_logits_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )

        self.final_layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, input_ids, attention_mask=None, time_steps=None, condition=None):
        batch_size, seq_len = input_ids.shape

        # 自动生成时间嵌入（如果没有提供）
        if time_steps is None:
            # 默认使用随机时间步或固定时间步
            time_steps = torch.rand(batch_size, 1, device=input_ids.device)

        time_emb = self.time_embedding(time_steps)

        # 自动生成条件嵌入（如果没有提供）
        if condition is None:
            # 默认使用零条件
            condition = torch.zeros(batch_size, self.config.condition_dim, device=input_ids.device)

        # 获取基础模型的输出
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = base_outputs.last_hidden_state

        # 添加时间和位置嵌入
        # 扩展时间嵌入以匹配序列长度维度
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        hidden_states = hidden_states + time_emb

        # 条件注入
        condition_proj = self.condition_projection(condition)
        if self.config.use_condition_injection:
            hidden_states = hidden_states + self.condition_injection(hidden_states, condition_proj)

        # 最终处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出头
        rates = F.softplus(self.rates_head(hidden_states))
        insert_logits = self.insert_logits_head(hidden_states)
        substitute_logits = self.substitute_logits_head(hidden_states)

        # 应用掩码
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            rates = rates * mask
            # 使用一个很大的负数替代-inf，提高softmax的数值稳定性
            invalid_mask = ~attention_mask.bool().unsqueeze(-1)
            insert_logits = insert_logits.masked_fill(invalid_mask, -1e9)
            substitute_logits = substitute_logits.masked_fill(invalid_mask, -1e9)

        insert_probs = F.softmax(insert_logits, dim=-1)
        substitute_probs = F.softmax(substitute_logits, dim=-1)

        return rates, insert_probs, substitute_probs
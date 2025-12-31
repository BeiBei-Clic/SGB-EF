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
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype) * -(math.log(10000.0) / (half_dim - 1))
        emb = torch.cat([torch.sin(t * exponent), torch.cos(t * exponent)], dim=-1)

        if self.hidden_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class CrossAttentionConditionInjection(nn.Module):
    """
    真正的交叉注意力机制 - 让每个 Query Token 从条件序列中动态选择信息

    数学原理:
    - Query 来自 BERT 的隐藏状态序列
    - Key/Value 来自条件编码器输出的多个特征向量
    - 每个 Query 可以从条件序列中关注不同的特征
    """
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
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) BERT 输出序列
            condition: (batch_size, num_cond_tokens, hidden_dim) 条件序列（如 32 个特征向量）

        Returns:
            output: (batch_size, seq_len, hidden_dim) 注入条件后的序列
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_cond_tokens = condition.shape[1]  # 条件序列长度，如 32

        # Query 来自 BERT 隐藏状态
        q = self.q_proj(hidden_states)  # (batch_size, seq_len, hidden_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Key/Value 来自条件序列（不再 expand！）
        k = self.cond_to_k(condition)  # (batch_size, num_cond_tokens, hidden_dim)
        k = k.view(batch_size, num_cond_tokens, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, num_cond_tokens, head_dim)

        v = self.cond_to_v(condition)  # (batch_size, num_cond_tokens, hidden_dim)
        v = v.view(batch_size, num_cond_tokens, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, num_cond_tokens, head_dim)

        # 计算注意力分数: (batch_size, num_heads, seq_len, num_cond_tokens)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Softmax 归一化（对条件序列维度）
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, num_cond_tokens)

        # 加权求和: (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)


class EditFlowTransformer(nn.Module):
    """EditFlow Transformer - 基于Transformer的符号回归编辑流模型"""

    def __init__(self, config, verbose=False):
        super().__init__()
        self.config = config

        # 加载预训练模型
        if hasattr(config, 'base_model_name') and config.base_model_name:
            cache_dir = Path("models/huggingface_cache").resolve()
            os.makedirs(cache_dir, exist_ok=True)

            if verbose:
                print(f"正在加载基础模型: {config.base_model_name} (缓存: {cache_dir})")

            self.base_model = AutoModel.from_pretrained(config.base_model_name, cache_dir=cache_dir, trust_remote_code=True)
            self.model_config = self.base_model.config

            config.hidden_dim = getattr(self.model_config, 'hidden_size')
            config.num_heads = getattr(self.model_config, 'num_attention_heads')
            config.num_layers = getattr(self.model_config, 'num_hidden_layers')

            original_vocab_size = getattr(self.model_config, 'vocab_size', self.base_model.config.vocab_size)
            if hasattr(config, 'vocab_size') and config.vocab_size and config.vocab_size > original_vocab_size:
                if verbose:
                    print(f"调整模型embedding层: {original_vocab_size} -> {config.vocab_size}")
                self.base_model.resize_token_embeddings(config.vocab_size)
            # 禁用不需要的 pooler 层梯度（避免分布式训练错误）
            if hasattr(self.base_model, 'pooler'):
                for param in self.base_model.pooler.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("必须指定base_model_name")

        # 时间嵌入层
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # 条件处理层 - 适配新的序列格式
        # condition_dim 现在应该是 (num_seeds * dim_hidden)，而不是单个向量的维度
        # 但由于条件编码器现在直接返回 (batch_size, num_seeds, dim_hidden) 格式
        # 我们需要检查维度是否匹配
        if config.condition_dim != config.hidden_dim:
            if verbose:
                print(f"警告: condition_dim ({config.condition_dim}) 应该等于 hidden_dim ({config.hidden_dim})")
                print(f"现在条件编码器输出 (batch_size, num_seeds, dim_hidden) 格式")

        # 不再需要 condition_projection，因为条件已经是正确的维度
        # self.condition_projection = nn.Linear(config.condition_dim, config.hidden_dim)

        if config.use_condition_injection:
            self.condition_injection = CrossAttentionConditionInjection(config.hidden_dim, config.num_heads)

        # 输出头 - 复用head创建逻辑
        def create_output_head(output_dim):
            return nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, output_dim)
            )

        self.rates_head = create_output_head(3)
        self.insert_logits_head = create_output_head(config.vocab_size)
        self.substitute_logits_head = create_output_head(config.vocab_size)

        # 层归一化
        self.time_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.condition_layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, input_ids, attention_mask=None, time_steps=None, condition=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            time_steps: (batch_size, 1) 或 None
            condition: (batch_size, num_cond_tokens, hidden_dim) 或 None
                      注意：现在 condition 是序列格式，而不是单个向量
        """
        batch_size, seq_len = input_ids.shape

        # 默认时间步和条件
        if time_steps is None:
            time_steps = torch.rand(batch_size, 1, device=input_ids.device)

        # 默认条件：空的序列（1 个 token，全零）
        if condition is None:
            condition = torch.zeros(batch_size, 1, self.config.hidden_dim, device=input_ids.device)

        # 基础模型前向传播
        hidden_states = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 添加时间嵌入
        time_emb = self.time_embedding(time_steps).unsqueeze(1).expand(-1, seq_len, -1)
        hidden_states = self.time_layer_norm(hidden_states + time_emb)

        # 条件注入 - 直接使用条件序列，不再投影
        if self.config.use_condition_injection:
            # condition 应该是 (batch_size, num_cond_tokens, hidden_dim) 格式
            # 如果是 2D 张量（旧格式），需要扩展
            if condition.dim() == 2:
                # 旧格式: (batch_size, condition_dim)
                # 转换为: (batch_size, 1, hidden_dim)
                condition = condition.unsqueeze(1)
                # 如果维度不匹配，需要投影
                if condition.shape[-1] != self.config.hidden_dim:
                    # 动态创建投影层
                    if not hasattr(self, 'condition_projection'):
                        self.condition_projection = nn.Linear(condition.shape[-1], self.config.hidden_dim).to(condition.device)
                    condition = self.condition_projection(condition)

            # 现在应用交叉注意力
            hidden_states = self.condition_layer_norm(
                hidden_states + self.condition_injection(hidden_states, condition)
            )

        # 生成输出
        raw_rates = self.rates_head(hidden_states)
        rates = F.softmax(raw_rates, dim=-1)  # 三种操作归一化概率

        # DEBUG: 验证归一化（每个batch都打印一次）
        if hasattr(self, '_debug_printed') == False:
            self._debug_printed = True
            print(f"[DEBUG] First forward pass:")
            print(f"  raw_rates[0, 0] = {raw_rates[0, 0].detach().cpu()}")
            print(f"  rates[0, 0] = {rates[0, 0].detach().cpu()}")
            print(f"  rates[0, 0].sum() = {rates[0, 0].sum().detach().cpu().item():.6f}")

        insert_logits = self.insert_logits_head(hidden_states)
        substitute_logits = self.substitute_logits_head(hidden_states)

        # 应用掩码
        if attention_mask is not None:
            invalid_mask = ~attention_mask.bool().unsqueeze(-1)
            # 使用 FP16 兼容的负无穷值（-1e4 远大于 FP16 最小值 -65504）
            insert_logits = insert_logits.masked_fill(invalid_mask, -1e4)
            substitute_logits = substitute_logits.masked_fill(invalid_mask, -1e4)
            rates = rates * attention_mask.unsqueeze(-1)

        return rates, F.softmax(insert_logits, dim=-1), F.softmax(substitute_logits, dim=-1)
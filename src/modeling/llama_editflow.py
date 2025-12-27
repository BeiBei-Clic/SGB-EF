"""
基于LLaMA的EditFlow模型实现 - 用于符号回归任务
使用Hugging Face transformers库中的LlamaModel作为骨干网络
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import LlamaConfig, LlamaModel
from typing import Optional, Tuple, Dict


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码 - 将标量时间步映射到向量空间"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t):
        """
        Args:
            t: (batch_size,) 或 (batch_size, 1) 时间步
        Returns:
            emb: (batch_size, hidden_dim) 时间嵌入向量
        """
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
    交叉注意力条件注入 - 让每个Query Token从条件序列中动态选择信息

    数学原理:
    - Query 来自 LLaMA 的隐藏状态序列
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
            hidden_states: (batch_size, seq_len, hidden_dim) LLaMA 输出序列
            condition: (batch_size, num_cond_tokens, hidden_dim) 条件序列

        Returns:
            output: (batch_size, seq_len, hidden_dim) 注入条件后的序列
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_cond_tokens = condition.shape[1]

        # Query 来自 LLaMA 隐藏状态
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Key/Value 来自条件序列
        k = self.cond_to_k(condition)
        k = k.view(batch_size, num_cond_tokens, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, num_cond_tokens, head_dim)

        v = self.cond_to_v(condition)
        v = v.view(batch_size, num_cond_tokens, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, num_cond_tokens, head_dim)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)


class LlamaEditFlowBackbone(nn.Module):
    """
    基于LLaMA的EditFlow骨干网络

    核心特性:
    - 使用LlamaModel作为基础transformer
    - RoPE旋转位置编码
    - SwiGLU激活函数
    - 时间步注入
    - 交叉注意力条件注入
    - 五头输出（插入率、删除率、替换率、插入词汇分布、替换词汇分布）
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        condition_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 24,
        use_condition_injection: bool = True,
        verbose: bool = False
    ):
        super().__init__()

        # 1. 定义LLaMA配置（用于backbone模型）
        self.llama_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,  # SwiGLU的中间维度
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            max_position_embeddings=max_seq_len + 32,  # 预留空间给条件token
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            use_cache=False,  # 非自回归不需要缓存
            hidden_act="silu",  # SwiGLU使用SiLU
            is_causal=False,  # 启用双向注意力（EditFlow需要非自回归的双向注意力）
        )

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.max_seq_len = max_seq_len
        self.use_condition_injection = use_condition_injection

        if verbose:
            print(f"初始化LlamaEditFlowBackbone:")
            print(f"  词表大小: {vocab_size}")
            print(f"  隐藏维度: {hidden_dim}")
            print(f"  层数: {n_layers}")
            print(f"  注意力头数: {n_heads}")
            print(f"  条件维度: {condition_dim}")

        # 2. 基础LLaMA骨干网络
        self.backbone = LlamaModel(self.llama_config)

        # 3. 时间嵌入 (标量t -> hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 4. 条件投影 (SetTransformer输出 -> hidden_dim)
        if condition_dim != hidden_dim:
            self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        else:
            self.cond_proj = nn.Identity()

        # 5. 交叉注意力条件注入
        if use_condition_injection:
            self.condition_injection = CrossAttentionConditionInjection(hidden_dim, n_heads)

        # 6. 层归一化
        self.time_layer_norm = nn.LayerNorm(hidden_dim)
        self.condition_layer_norm = nn.LayerNorm(hidden_dim)

        # 7. 编辑流五大输出头 (论文公式13-15)

        # 预测速率 (Rates) - 使用softplus确保正值
        self.ins_rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.del_rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.sub_rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 预测分布 (Distributions)
        self.ins_vocab_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        self.sub_vocab_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: (batch_size, seq_len) 输入token IDs
            time_steps: (batch_size, 1) 或 (batch_size,) 时间步t
            condition: (batch_size, num_cond_tokens, condition_dim) 来自SetTransformer
            attention_mask: (batch_size, seq_len) 注意力掩码

        Returns:
            dict: 包含以下键:
                - 'rates': (ins_rate, del_rate, sub_rate) 每个都是 (batch_size, seq_len, 1)
                - 'insert_logits': (batch_size, seq_len, vocab_size)
                - 'substitute_logits': (batch_size, seq_len, vocab_size)
                - 'insert_probs': (batch_size, seq_len, vocab_size) softmax后的概率
                - 'substitute_probs': (batch_size, seq_len, vocab_size) softmax后的概率
        """
        batch_size, seq_len = input_ids.shape

        # 默认时间步
        if time_steps is None:
            time_steps = torch.rand(batch_size, 1, device=input_ids.device)
        elif time_steps.dim() == 1:
            time_steps = time_steps.unsqueeze(-1)

        # 默认条件：空的序列（1个token，全零）
        if condition is None:
            condition = torch.zeros(batch_size, 1, self.condition_dim, device=input_ids.device)

        # A. 获取token embedding
        inputs_embeds = self.backbone.embed_tokens(input_ids)  # (batch_size, seq_len, hidden_dim)

        # B. 注入时间信息
        t_emb = self.time_embedding(time_steps)  # (batch_size, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        inputs_embeds = self.time_layer_norm(inputs_embeds + t_emb)

        # C. 处理条件并投影
        if condition.dim() == 2:
            # 旧格式: (batch_size, condition_dim)
            condition = condition.unsqueeze(1)

        condition_proj = self.cond_proj(condition)  # (batch_size, num_cond_tokens, hidden_dim)

        # D. 通过LLaMA骨干（使用双向注意力）
        # 注意：LLaMA默认使用因果掩码，但EditFlow需要双向注意力
        # 因此我们需要创建全1的attention_mask来禁用因果掩码
        if attention_mask is not None:
            # 扩展attention_mask以包含条件token
            num_cond_tokens = condition_proj.shape[1]
            cond_mask = torch.ones(batch_size, num_cond_tokens,
                                  device=attention_mask.device,
                                  dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([cond_mask, attention_mask], dim=1)

            # 将0/1掩码转换为LLaMA需要的格式
            # LLaMA使用bool掩码，True表示可以attend
            full_attention_mask = full_attention_mask.bool()
        else:
            num_cond_tokens = condition_proj.shape[1]
            full_attention_mask = torch.ones(batch_size, seq_len + num_cond_tokens,
                                           device=input_ids.device, dtype=torch.bool)

        # 将条件token作为前缀添加到输入
        combined_embeds = torch.cat([condition_proj, inputs_embeds], dim=1)

        # E. LLaMA前向传播
        outputs = self.backbone(
            inputs_embeds=combined_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        )

        # F. 获取隐藏状态（跳过条件token部分）
        hidden_states = outputs.last_hidden_state[:, num_cond_tokens:, :]  # (batch_size, seq_len, hidden_dim)

        # G. 条件注入（可选）
        if self.use_condition_injection:
            hidden_states = self.condition_layer_norm(
                hidden_states + self.condition_injection(hidden_states, condition_proj)
            )

        # H. 计算五大输出

        # 速率预测（使用softplus确保正值）
        ins_rate = F.softplus(self.ins_rate_head(hidden_states))
        del_rate = F.softplus(self.del_rate_head(hidden_states))
        sub_rate = F.softplus(self.sub_rate_head(hidden_states))

        # 词汇分布预测
        ins_logits = self.ins_vocab_head(hidden_states)
        sub_logits = self.sub_vocab_head(hidden_states)

        # 应用注意力掩码
        if attention_mask is not None:
            invalid_mask = ~attention_mask.bool().unsqueeze(-1)
            # 使用FP16兼容的负无穷值
            ins_logits = ins_logits.masked_fill(invalid_mask, -1e4)
            sub_logits = sub_logits.masked_fill(invalid_mask, -1e4)
            ins_rate = ins_rate * attention_mask.unsqueeze(-1)
            del_rate = del_rate * attention_mask.unsqueeze(-1)
            sub_rate = sub_rate * attention_mask.unsqueeze(-1)

        # 计算概率分布
        insert_probs = F.softmax(ins_logits, dim=-1)
        substitute_probs = F.softmax(sub_logits, dim=-1)

        return {
            'rates': (ins_rate, del_rate, sub_rate),
            'insert_logits': ins_logits,
            'substitute_logits': sub_logits,
            'insert_probs': insert_probs,
            'substitute_probs': substitute_probs,
        }


class LlamaEditFlowConfig:
    """LlamaEditFlow配置类"""
    def __init__(
        self,
        vocab_size: int = 100,  # 符号回归专用词表
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        condition_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 24,
        use_condition_injection: bool = True,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.condition_dim = condition_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_condition_injection = use_condition_injection


# 为了向后兼容，创建一个包装类
class EditFlowTransformer(LlamaEditFlowBackbone):
    """
    向后兼容的包装类，保持与现有代码的接口一致性
    """
    def __init__(self, config, verbose=False):
        # 保存config以便外部访问
        self.config = config

        # 从config中提取参数（同时支持新旧两种命名方式）
        vocab_size = getattr(config, 'vocab_size', 100)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        # 支持两种命名：num_layers (旧) 和 n_layers (新LlamaEditFlowConfig)
        n_layers = getattr(config, 'num_layers', None) or getattr(config, 'n_layers', 6)
        # 支持两种命名：num_heads (旧) 和 n_heads (新LlamaEditFlowConfig)
        n_heads = getattr(config, 'num_heads', None) or getattr(config, 'n_heads', 8)
        condition_dim = getattr(config, 'condition_dim', 128)
        dropout = getattr(config, 'dropout', 0.1)
        max_seq_len = getattr(config, 'max_seq_len', 24)
        use_condition_injection = getattr(config, 'use_condition_injection', True)

        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            condition_dim=condition_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_condition_injection=use_condition_injection,
            verbose=verbose
        )

    def forward(self, input_ids, attention_mask=None, time_steps=None, condition=None):
        """
        保持与旧版本相同的接口

        Returns:
            rates: (batch_size, seq_len, 3) 三个速率合并
            insert_probs: (batch_size, seq_len, vocab_size)
            substitute_probs: (batch_size, seq_len, vocab_size)
        """
        output = super().forward(input_ids, time_steps, condition, attention_mask)

        # 合并三个速率为一个tensor
        ins_rate, del_rate, sub_rate = output['rates']
        rates = torch.cat([ins_rate, del_rate, sub_rate], dim=-1)

        return rates, output['insert_probs'], output['substitute_probs']

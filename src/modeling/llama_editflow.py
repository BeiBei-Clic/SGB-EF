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


class CrossAttentionConditionInjection(nn.Module):
    """
    交叉注意力条件注入 - 条件信息注入的唯一方式

    设计说明:
    - 不再使用Prefix Padding，避免与LLaMA自注意力的冗余
    - Query 来自 LLaMA 的隐藏状态序列
    - Key/Value 来自条件编码器输出的特征向量
    - 每个 Query 可以从条件序列中动态选择需要的信息

    数学原理:
    - Query: LLaMA处理后的每个位置表示
    - Key/Value: SetTransformer提取的条件特征
    - 输出: 加权后的条件信息，通过残差连接注入
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
    - 交叉注意力条件注入（Cross-Attention，非Prefix）
    - 五头输出（插入率、删除率、替换率、插入词汇分布、替换词汇分布）

    条件注入策略:
    条件信息仅通过CrossAttention机制注入，不使用Prefix Padding。
    这样可以避免双重条件注入的冗余，简化模型架构，减少参数量。
    LLaMA的强大自注意力机制已经足够处理输入序列内部的复杂关系。
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
            max_position_embeddings=max_seq_len,  # 条件通过交叉注意力注入，无需预留空间
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            use_cache=False,  # 非自回归不需要缓存
            hidden_act="silu",  # SwiGLU使用SiLU
            is_causal=False,  # 启用双向注意力（EditFlow需要非自回归的双向注意力）
        )

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.condition_dim = condition_dim
        self.dropout = dropout
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

        # 3. 条件投影 (SetTransformer输出 -> hidden_dim)
        # 注意：条件仅通过交叉注意力注入，不再作为前缀token
        if condition_dim != hidden_dim:
            self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        else:
            self.cond_proj = nn.Identity()

        # 4. 交叉注意力条件注入
        if use_condition_injection:
            self.condition_injection = CrossAttentionConditionInjection(hidden_dim, n_heads)

        # 5. 层归一化
        self.condition_layer_norm = nn.LayerNorm(hidden_dim)

        # 6. 编辑流五大输出头 (论文公式13-15)

        # 预测操作类型概率 (Rates) - 使用softmax归一化，确保三种操作互斥
        self.rates_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 输出3维：[ins_logit, del_logit, sub_logit]
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
        condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: (batch_size, seq_len) 输入token IDs
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

        # 默认条件：空的序列（1个token，全零）
        if condition is None:
            condition = torch.zeros(batch_size, 1, self.condition_dim, device=input_ids.device)

        # A. 获取token embedding
        inputs_embeds = self.backbone.embed_tokens(input_ids)  # (batch_size, seq_len, hidden_dim)

        # B. 处理条件并投影
        if condition.dim() == 2:
            # 旧格式: (batch_size, condition_dim)
            condition = condition.unsqueeze(1)

        condition_proj = self.cond_proj(condition)  # (batch_size, num_cond_tokens, hidden_dim)

        # C. 通过LLaMA骨干（使用双向注意力）
        # 注意：LLaMA默认使用因果掩码，但EditFlow需要双向注意力
        # 因此我们需要创建全1的attention_mask来禁用因果掩码
        if attention_mask is not None:
            # 将0/1掩码转换为LLaMA需要的格式
            # LLaMA使用bool掩码，True表示可以attend
            attention_mask = attention_mask.bool()
        else:
            attention_mask = torch.ones(batch_size, seq_len,
                                       device=input_ids.device, dtype=torch.bool)

        # E. LLaMA前向传播（直接使用inputs_embeds，不拼接条件token）
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # F. 获取隐藏状态
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # G. 条件注入（通过交叉注意力）
        if self.use_condition_injection:
            hidden_states = self.condition_layer_norm(
                hidden_states + self.condition_injection(hidden_states, condition_proj)
            )

        # H. 计算五大输出

        # 操作类型预测（使用softmax归一化，确保互斥）
        rates_logits = self.rates_head(hidden_states)  # (batch_size, seq_len, 3)
        rates_probs = F.softmax(rates_logits, dim=-1)   # (batch_size, seq_len, 3)

        # 分解为三个操作的概率（每个都是batch_size, seq_len, 1）
        ins_rate = rates_probs[:, :, 0:1]  # 插入概率
        del_rate = rates_probs[:, :, 1:2]  # 删除概率
        sub_rate = rates_probs[:, :, 2:3]  # 替换概率

        # 词汇分布预测
        ins_logits = self.ins_vocab_head(hidden_states)
        sub_logits = self.sub_vocab_head(hidden_states)

        # 应用注意力掩码
        if attention_mask is not None:
            invalid_mask = ~attention_mask.bool().unsqueeze(-1)
            # 使用dtype的真正负无穷值，确保softmax后概率接近0
            float_type = ins_logits.dtype
            min_val = torch.finfo(float_type).min / 2  # 留出余量避免数值问题
            ins_logits = ins_logits.masked_fill(invalid_mask, min_val)
            sub_logits = sub_logits.masked_fill(invalid_mask, min_val)
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

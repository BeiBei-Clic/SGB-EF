"""
基于LLaMA的EditFlow模型实现 - 用于符号回归任务
使用Hugging Face transformers库中的LlamaModel作为骨干网络
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel
from typing import Optional, Dict


# 导入时间步嵌入模块
from .time_embedding import SinusoidalTimeEmbedding, AdaLNModulator


class CrossAttentionConditionInjection(nn.Module):
    """
    交叉注意力条件注入

    Query来自LLaMA隐藏状态，Key/Value来自条件编码器输出。
    通过残差连接将条件信息注入到LLaMA输出中。
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
            hidden_states: (batch_size, seq_len, hidden_dim) LLaMA输出序列
            condition: (batch_size, num_cond_tokens, hidden_dim) 条件序列

        Returns:
            output: (batch_size, seq_len, hidden_dim) 注入条件后的序列
        """
        batch_size, seq_len = hidden_states.shape[:2]
        num_cond_tokens = condition.shape[1]

        # Query来自LLaMA隐藏状态，Key/Value来自条件序列
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Key/Value来自条件序列
        k = self.cond_to_k(condition).view(batch_size, num_cond_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cond_to_v(condition).view(batch_size, num_cond_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_output = torch.matmul(F.softmax(attn_scores, dim=-1), v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)


class LlamaEditFlowBackbone(nn.Module):
    """
    基于LLaMA的EditFlow骨干网络

    核心特性:
    - 使用LlamaModel作为基础transformer（RoPE位置编码、SwiGLU激活函数）
    - 交叉注意力条件注入（非Prefix方式）
    - 五头输出：操作类型概率、插入词汇分布、替换词汇分布
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
        time_embed_dim: int = 256,
        time_max_period: float = 10000.0,
        verbose: bool = False
    ):
        super().__init__()

        self.llama_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            max_position_embeddings=max_seq_len,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            use_cache=False,
            hidden_act="silu",
            is_causal=False,
        )
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.condition_dim = condition_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        if verbose:
            print(f"初始化LlamaEditFlowBackbone: vocab_size={vocab_size}, hidden_dim={hidden_dim}, n_layers={n_layers}")
            print(f"  时间步嵌入已启用: time_embed_dim={time_embed_dim}, time_max_period={time_max_period}")

        # 基础LLaMA骨干网络
        self.backbone = LlamaModel(self.llama_config)

        # 条件投影
        self.cond_proj = nn.Linear(condition_dim, hidden_dim) if condition_dim != hidden_dim else nn.Identity()

        # 交叉注意力条件注入（始终启用）
        self.condition_injection = CrossAttentionConditionInjection(hidden_dim, n_heads)
        self.condition_layer_norm = nn.LayerNorm(hidden_dim)

        # 时间步嵌入和 AdaLN 调制器（始终启用）
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=time_embed_dim,
            max_period=time_max_period
        )
        # 为每个 Transformer 层创建 AdaLN 调制器
        self.adaln_modulators = nn.ModuleList([
            AdaLNModulator(time_embed_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # 编辑流输出头
        self.rates_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)
        )
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
        timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: (batch_size, seq_len) 输入token IDs
            condition: (batch_size, num_cond_tokens, condition_dim) 来自SetTransformer
            attention_mask: (batch_size, seq_len) 注意力掩码
            timestep: (batch_size,) 时间步，范围 [0, 1]（必需参数）

        Returns:
            dict: 包含 rates_logits, insert_logits, substitute_logits, insert_probs, substitute_probs
        """
        batch_size, seq_len = input_ids.shape

        # 时间步嵌入（始终启用）
        if timestep is None:
            raise ValueError("timestep 是必需的参数")
        time_embed = self.time_embedding(timestep)

        # 处理条件
        if condition is None:
            condition = torch.zeros(batch_size, 1, self.condition_dim, device=input_ids.device)
        elif condition.dim() == 2:
            condition = condition.unsqueeze(1)

        condition_proj = self.cond_proj(condition)

        # 处理attention_mask（EditFlow需要双向注意力）
        attention_mask = attention_mask.bool() if attention_mask is not None else torch.ones(
            batch_size, seq_len, device=input_ids.device, dtype=torch.bool
        )

        # 嵌入 tokens
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # 手动遍历 LLaMA 层，应用 AdaLN 调制（始终启用）
        # 创建钩子函数用于 AdaLN 调制
        hooks = []
        for layer_idx in range(self.n_layers):
            def create_hook(idx):
                def hook(module, input, output):
                    # output 是 tuple，output[0] 是 hidden_states
                    h = output[0] if isinstance(output, tuple) else output
                    scale, shift = self.adaln_modulators[idx](time_embed)
                    h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                    return (h,) + output[1:] if isinstance(output, tuple) else h
                return hook

            hook = self.backbone.layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        # 调用 backbone（钩子会自动应用 AdaLN）
        hidden_states = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 条件注入（通过交叉注意力，始终启用）
        hidden_states = self.condition_layer_norm(
            hidden_states + self.condition_injection(hidden_states, condition_proj)
        )

        # 计算输出
        rates_logits = self.rates_head(hidden_states)
        ins_logits = self.ins_vocab_head(hidden_states)
        sub_logits = self.sub_vocab_head(hidden_states)

        # 应用注意力掩码
        invalid_mask = ~attention_mask.unsqueeze(-1)
        min_val = torch.finfo(ins_logits.dtype).min / 2
        ins_logits = ins_logits.masked_fill(invalid_mask, min_val)
        sub_logits = sub_logits.masked_fill(invalid_mask, min_val)
        rates_logits = rates_logits.masked_fill(invalid_mask.expand(-1, -1, 4), min_val)

        return {
            'rates_logits': rates_logits,
            'insert_logits': ins_logits,
            'substitute_logits': sub_logits,
            'insert_probs': F.softmax(ins_logits, dim=-1),
            'substitute_probs': F.softmax(sub_logits, dim=-1),
        }

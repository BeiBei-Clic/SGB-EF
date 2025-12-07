"""
EditFlow Transformer - 基于GPT的符号回归编辑流模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GPT2Model, GPT2Config
import math


class EditFlowConfig(PretrainedConfig):
    model_type = "editflow"

    def __init__(self, vocab_size=50257, hidden_dim=768, num_layers=12, num_heads=12,
                 max_seq_len=1024, dropout=0.1, pad_token_id=0, condition_dim=None,
                 base_model_name="gpt2", use_condition_injection=True,
                 time_embedding_type="sinusoidal", **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.condition_dim = condition_dim or hidden_dim
        self.base_model_name = base_model_name
        self.use_condition_injection = use_condition_injection
        self.time_embedding_type = time_embedding_type


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


class EditFlowTransformer(PreTrainedModel):
    config_class = EditFlowConfig

    def __init__(self, config):
        super().__init__(config)

        # 创建GPT模型
        gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=config.hidden_dim,
            n_layer=config.num_layers,
            n_head=config.num_heads,
            n_positions=config.max_seq_len,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
        )

        if hasattr(config, 'base_model_name') and config.base_model_name:
            try:
                self.gpt_model = GPT2Model.from_pretrained(config.base_model_name, config=gpt_config)
            except:
                self.gpt_model = GPT2Model(gpt_config)
        else:
            self.gpt_model = GPT2Model(gpt_config)

        # 自定义嵌入层（GPT已经内置位置嵌入，但我们添加额外的位置和时间嵌入）
        self.extra_position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # 时间嵌入
        if config.time_embedding_type == "sinusoidal":
            self.time_embedding = nn.Sequential(
                SinusoidalTimeEmbedding(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        else:
            self.time_embedding = nn.Sequential(
                nn.Linear(1, config.hidden_dim),
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
        self.post_init()

    def forward(self, input_ids, time_steps, condition, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # 获取GPT的token嵌入
        token_emb = self.gpt_model.wte(input_ids)

        # 额外的位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        extra_pos_emb = self.extra_position_embedding(positions)

        # 时间嵌入
        if self.config.time_embedding_type == "sinusoidal":
            time_emb = self.time_embedding(time_steps)
        else:
            time_emb = self.time_embedding(time_steps.float())
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # 组合嵌入
        inputs_embeds = token_emb + extra_pos_emb + time_emb

        # 通过GPT模型
        gpt_outputs = self.gpt_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = gpt_outputs.last_hidden_state

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
            insert_logits = insert_logits.masked_fill(~attention_mask.bool().unsqueeze(-1), -float('inf'))
            substitute_logits = substitute_logits.masked_fill(~attention_mask.bool().unsqueeze(-1), -float('inf'))

        insert_probs = F.softmax(insert_logits, dim=-1)
        substitute_probs = F.softmax(substitute_logits, dim=-1)

        return rates, insert_probs, substitute_probs


# 测试代码
if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel

    # 注册模型
    AutoConfig.register("editflow", EditFlowConfig)
    AutoModel.register(EditFlowConfig, EditFlowTransformer)

    config = EditFlowConfig(vocab_size=1000, hidden_dim=256, num_layers=4, num_heads=8)
    model = EditFlowTransformer(config)

    # 测试数据
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    time_steps = torch.rand(2, 1)
    condition = torch.randn(2, config.condition_dim)
    attention_mask = torch.ones(2, 64)

    # 前向传播
    model.eval()
    with torch.no_grad():
        rates, insert_probs, substitute_probs = model(input_ids, time_steps, condition, attention_mask)

    print(f"模型参数数量: {model.num_parameters():,}")
    print(f"输入形状: {input_ids.shape}")
    print(f"速率输出形状: {rates.shape}")
    print(f"插入概率形状: {insert_probs.shape}")
    print(f"替换概率形状: {substitute_probs.shape}")
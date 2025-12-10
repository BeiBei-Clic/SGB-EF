#!/usr/bin/env python3
"""
调试训练步骤 - 测试单个训练前向传播
"""

import sys
sys.path.append('/home/xyh/SGB-EF')

import torch
import numpy as np
from src.training.flow import tokens_to_prob, sample_conditional_path, remove_gap_tokens, KappaScheduler
from src.symbolic.data_generator import generate_flow_samples
from src.utils.special_tokens import SpecialTokensManager
from src.modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig
from src.modeling.condition_encoder import ConditionEncoder
from transformers import AutoTokenizer

def test_single_forward_pass():
    """测试单个前向传播"""
    print("=== 测试单个前向传播 ===")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)

    # 生成数据
    print("1. 生成训练数据...")
    samples = generate_flow_samples(2, max_dim=2, n_points=5, max_depth=2)

    # 初始化模型
    print("2. 初始化模型...")
    condition_encoder = ConditionEncoder("Qwen/Qwen3-Embedding-0.6B")
    config = EditFlowConfig(
        condition_dim=condition_encoder.output_dim,
        base_model_name="google-bert/bert-base-uncased",
    )
    model = EditFlowTransformer(config)

    model = model.to(device)
    condition_encoder = condition_encoder.to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 处理数据
    print("3. 处理数据...")
    sample = samples[0]

    # Tokenize表达式
    z0_tokens = []
    z1_tokens = []
    for token in sample['z0_tokens']:
        z0_tokens.extend(special_tokens_manager.tokenize_expression(token))
    for token in sample['z1_tokens']:
        z1_tokens.extend(special_tokens_manager.tokenize_expression(token))

    # 创建张量
    max_len = 128
    bos_token = tokenizer.cls_token_id
    pad_token = tokenizer.pad_token_id

    z0_tensor = torch.zeros(1, max_len, dtype=torch.long, device=device)
    z1_tensor = torch.zeros(1, max_len, dtype=torch.long, device=device)

    z0_seq = [bos_token] + z0_tokens[:max_len-1]
    z1_seq = [bos_token] + z1_tokens[:max_len-1]

    z0_tensor[0, :len(z0_seq)] = torch.tensor(z0_seq)
    z1_tensor[0, :len(z1_seq)] = torch.tensor(z1_seq)

    print(f"z0序列长度: {len(z0_seq)}")
    print(f"z1序列长度: {len(z1_seq)}")
    print(f"z0_token范围: [{z0_tensor.min()}, {z0_tensor.max()}]")
    print(f"z1_token范围: [{z1_tensor.min()}, {z1_tensor.max()}]")

    # 准备条件编码
    print("4. 准备条件编码...")
    x_values = torch.FloatTensor([sample['x_values']]).to(device)
    residuals = torch.FloatTensor([sample['residuals']]).to(device)

    condition_embeddings = condition_encoder(x_values, residuals)
    print(f"条件嵌入形状: {condition_embeddings.shape}")

    # 前向传播
    print("5. 执行前向传播...")
    batch_size = 1

    # 时间步
    t = torch.rand(batch_size, 1, device=device)
    print(f"时间步: {t}")

    # 转换为概率分布
    print("6. 转换为概率分布...")
    z0_probs = tokens_to_prob(z0_tensor, tokenizer.vocab_size)
    z1_probs = tokens_to_prob(z1_tensor, tokenizer.vocab_size)
    print(f"z0_probs形状: {z0_probs.shape}")
    print(f"z1_probs形状: {z1_probs.shape}")
    print(f"z0_probs概率和: {z0_probs.sum(dim=-1)}")
    print(f"z1_probs概率和: {z1_probs.sum(dim=-1)}")

    # 检查概率分布有效性
    print(f"z0_probs是否包含NaN: {torch.isnan(z0_probs).any()}")
    print(f"z1_probs是否包含NaN: {torch.isnan(z1_probs).any()}")
    print(f"z0_probs是否包含Inf: {torch.isinf(z0_probs).any()}")
    print(f"z1_probs是否包含Inf: {torch.isinf(z1_probs).any()}")

    # 采样条件路径
    print("7. 采样条件路径...")
    scheduler = KappaScheduler('cubic')
    z_t = sample_conditional_path(z0_probs, z1_probs, t, scheduler)
    print(f"z_t形状: {z_t.shape}")
    print(f"z_t范围: [{z_t.min()}, {z_t.max()}]")
    print(f"z_t有效范围: {(z_t >= 0).all() and (z_t < tokenizer.vocab_size).all()}")

    # 移除gap tokens
    print("8. 移除gap tokens...")
    gap_token = special_tokens_manager.get_gap_token_id()
    x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(
        z_t, tokenizer.vocab_size, gap_token
    )
    print(f"x_t形状: {x_t.shape}")
    print(f"x_pad_mask形状: {x_pad_mask.shape}")
    print(f"x_t有效范围: {(x_t >= 0).all() and (x_t < tokenizer.vocab_size).all()}")

    # 创建attention mask
    attention_mask = (~x_pad_mask).float()
    print(f"attention_mask形状: {attention_mask.shape}")
    print(f"attention_mask范围: [{attention_mask.min()}, {attention_mask.max()}]")

    # 模型前向传播
    print("9. 模型前向传播...")
    pred_rates, pred_ins_probs, pred_sub_probs = model(
        input_ids=x_t,
        time_steps=t,
        condition=condition_embeddings,
        attention_mask=attention_mask
    )

    print(f"pred_rates形状: {pred_rates.shape}")
    print(f"pred_ins_probs形状: {pred_ins_probs.shape}")
    print(f"pred_sub_probs形状: {pred_sub_probs.shape}")

    # 检查输出有效性
    print(f"pred_rates是否包含NaN: {torch.isnan(pred_rates).any()}")
    print(f"pred_ins_probs是否包含NaN: {torch.isnan(pred_ins_probs).any()}")
    print(f"pred_sub_probs是否包含NaN: {torch.isnan(pred_sub_probs).any()}")

    print(f"pred_rates是否包含Inf: {torch.isinf(pred_rates).any()}")
    print(f"pred_ins_probs是否包含Inf: {torch.isinf(pred_ins_probs).any()}")
    print(f"pred_sub_probs是否包含Inf: {torch.isinf(pred_sub_probs).any()}")

    print(f"pred_rates范围: [{pred_rates.min():.6f}, {pred_rates.max():.6f}]")
    print(f"pred_ins_probs范围: [{pred_ins_probs.min():.6f}, {pred_ins_probs.max():.6f}]")
    print(f"pred_sub_probs范围: [{pred_sub_probs.min():.6f}, {pred_sub_probs.max():.6f}]")

    print("✅ 前向传播完成，无CUDA错误!")

def main():
    print("开始调试训练步骤...")

    test_single_forward_pass()

    print("\n🎉 单步训练测试完成!")

if __name__ == "__main__":
    main()
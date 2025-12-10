#!/usr/bin/env python3
"""
调试张量操作 - 测试流匹配中的关键张量操作
"""

import sys
sys.path.append('/home/xyh/SGB-EF')

import torch
import numpy as np
from src.training.flow import tokens_to_prob, sample_conditional_path, remove_gap_tokens, KappaScheduler
from src.symbolic.data_generator import generate_flow_samples
from src.utils.special_tokens import SpecialTokensManager
from transformers import AutoTokenizer

def test_tokens_to_prob():
    """测试tokens_to_prob函数"""
    print("=== 测试tokens_to_prob函数 ===")

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # 测试正常情况
    print("\n1. 测试正常token序列...")
    normal_tokens = torch.tensor([[101, 102, 103, 104]], dtype=torch.long)  # 有效的token IDs
    probs = tokens_to_prob(normal_tokens, vocab_size)
    print(f"输入形状: {normal_tokens.shape}")
    print(f"输出形状: {probs.shape}")
    print(f"概率和: {probs.sum(dim=-1)}")
    print(f"是否包含NaN: {torch.isnan(probs).any()}")
    print(f"是否包含Inf: {torch.isinf(probs).any()}")

    # 测试边界情况
    print("\n2. 测试边界token序列...")
    boundary_tokens = torch.tensor([[0, vocab_size-1]], dtype=torch.long)
    probs_boundary = tokens_to_prob(boundary_tokens, vocab_size)
    print(f"边界token概率和: {probs_boundary.sum(dim=-1)}")

    # 测试超出范围的token
    print("\n3. 测试超出范围的token...")
    overflow_tokens = torch.tensor([[0, vocab_size+100]], dtype=torch.long)  # 故意使用超大token ID
    probs_overflow = tokens_to_prob(overflow_tokens, vocab_size)
    print(f"超出范围token处理后的概率和: {probs_overflow.sum(dim=-1)}")

def test_conditional_path_sampling():
    """测试条件路径采样"""
    print("\n=== 测试条件路径采样 ===")

    scheduler = KappaScheduler('cubic')

    # 创建简单的概率分布
    batch_size, seq_len, vocab_size = 2, 3, 10
    p0 = torch.zeros(batch_size, seq_len, vocab_size)
    p1 = torch.zeros(batch_size, seq_len, vocab_size)

    # 设置有效的概率分布
    p0[:, :, 0] = 1.0  # 第一个token
    p1[:, :, 1] = 1.0  # 第二个token

    print("1. 测试正常采样...")
    t = torch.tensor([[0.5], [0.5]], dtype=torch.float32)
    z_t = sample_conditional_path(p0, p1, t, scheduler)
    print(f"采样结果形状: {z_t.shape}")
    print(f"采样结果范围: [{z_t.min()}, {z_t.max()}]")
    print(f"采样结果是否有效: {(z_t >= 0).all() and (z_t < vocab_size).all()}")

    # 测试极端时间值
    print("\n2. 测试极端时间值...")
    t_extreme = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    z_t_extreme = sample_conditional_path(p0, p1, t_extreme, scheduler)
    print(f"极端时间采样结果: {z_t_extreme}")

def test_remove_gap_tokens():
    """测试remove_gap_tokens函数"""
    print("\n=== 测试remove_gap_tokens函数 ===")

    # 模拟包含gap token的序列
    batch_size, seq_len = 2, 5
    pad_token_id = 0
    gap_token_id = 999

    z_t = torch.tensor([
        [1, gap_token_id, 2, 3, pad_token_id],
        [gap_token_id, 4, gap_token_id, 5, 6]
    ], dtype=torch.long)

    print("输入序列:")
    print(z_t)

    x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(z_t, pad_token_id, gap_token_id)

    print("\n处理后的序列:")
    print(f"x_t: {x_t}")
    print(f"x_t形状: {x_t.shape}")
    print(f"x_pad_mask: {x_pad_mask}")
    print(f"z_gap_mask: {z_gap_mask}")
    print(f"z_pad_mask: {z_pad_mask}")

    # 检查结果有效性
    print(f"\n结果有效性检查:")
    print(f"x_t是否包含无效值: {(x_t < 0).any()}")
    print(f"pad mask是否正确: {x_pad_mask.dtype == torch.bool}")

def test_full_data_pipeline():
    """测试完整的数据处理流程"""
    print("\n=== 测试完整数据处理流程 ===")

    # 生成数据
    samples = generate_flow_samples(2, max_dim=2, n_points=5, max_depth=2)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)

    # 处理第一个样本
    sample = samples[0]
    z0_tokens = [special_tokens_manager.tokenize_expression(token) for token in sample['z0_tokens']]
    z1_tokens = [special_tokens_manager.tokenize_expression(token) for token in sample['z1_tokens']]

    # 展平token列表
    z0_flat = [item for sublist in z0_tokens for item in sublist]
    z1_flat = [item for sublist in z1_tokens for item in sublist]

    # 转换为张量
    max_len = 10
    z0_tensor = torch.zeros(1, max_len, dtype=torch.long)
    z1_tensor = torch.zeros(1, max_len, dtype=torch.long)

    z0_tensor[0, :len(z0_flat)] = torch.tensor(z0_flat[:max_len])
    z1_tensor[0, :len(z1_flat)] = torch.tensor(z1_flat[:max_len])

    print(f"z0_tensor: {z0_tensor}")
    print(f"z1_tensor: {z1_tensor}")

    # 测试tokens_to_prob
    p0 = tokens_to_prob(z0_tensor, tokenizer.vocab_size)
    p1 = tokens_to_prob(z1_tensor, tokenizer.vocab_size)

    print(f"p0形状: {p0.shape}")
    print(f"p1形状: {p1.shape}")
    print(f"p0概率和: {p0.sum(dim=-1)}")
    print(f"p1概率和: {p1.sum(dim=-1)}")

    # 测试条件路径采样
    t = torch.tensor([[0.5]], dtype=torch.float32)
    scheduler = KappaScheduler('cubic')
    z_t = sample_conditional_path(p0, p1, t, scheduler)

    print(f"z_t: {z_t}")
    print(f"z_t有效范围: {(z_t >= 0).all() and (z_t < tokenizer.vocab_size).all()}")

def main():
    print("开始调试张量操作...")

    test_tokens_to_prob()
    test_conditional_path_sampling()
    test_remove_gap_tokens()
    test_full_data_pipeline()

    print("\n✅ 所有张量操作测试完成!")

if __name__ == "__main__":
    main()
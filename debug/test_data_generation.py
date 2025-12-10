#!/usr/bin/env python3
"""
调试数据生成 - 测试SymPy表达式生成和求值
"""

import sys
sys.path.append('/home/xyh/SGB-EF')

import torch
import numpy as np
from src.symbolic.data_generator import generate_flow_samples
from src.utils.special_tokens import SpecialTokensManager
from transformers import AutoTokenizer

def test_data_generation():
    """测试数据生成过程"""
    print("=== 测试数据生成 ===")

    # 生成小批量数据
    print("1. 生成triplet样本...")
    samples = generate_flow_samples(5, max_dim=2, n_points=10, max_depth=2)
    print(f"成功生成 {len(samples)} 个样本")

    # 检查每个样本的数据完整性
    for i, sample in enumerate(samples):
        print(f"\n样本 {i}:")
        print(f"  输入维度: {sample['input_dimension']}")
        print(f"  z0_tokens: {sample['z0_tokens']}")
        print(f"  z1_tokens: {sample['z1_tokens']}")
        print(f"  x_values形状: {np.array(sample['x_values']).shape}")
        print(f"  residuals形状: {np.array(sample['residuals']).shape}")

        # 检查数值有效性
        x_vals = np.array(sample['x_values'])
        res_vals = np.array(sample['residuals'])

        print(f"  x_values范围: [{x_vals.min():.4f}, {x_vals.max():.4f}]")
        print(f"  residuals范围: [{res_vals.min():.4f}, {res_vals.max():.4f}]")
        print(f"  x_values是否包含NaN: {np.isnan(x_vals).any()}")
        print(f"  residuals是否包含NaN: {np.isnan(res_vals).any()}")
        print(f"  x_values是否包含Inf: {np.isinf(x_vals).any()}")
        print(f"  residuals是否包含Inf: {np.isinf(res_vals).any()}")

def test_tokenization():
    """测试tokenization过程"""
    print("\n=== 测试Tokenization ===")

    # 初始化tokenizer
    print("1. 初始化tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Cls token ID: {tokenizer.cls_token_id}")

    # 初始化特殊tokens管理器
    print("2. 初始化特殊tokens管理器...")
    special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)
    gap_token_id = special_tokens_manager.get_gap_token_id()
    print(f"Gap token ID: {gap_token_id}")

    # 生成数据并测试tokenization
    print("3. 测试表达式tokenization...")
    samples = generate_flow_samples(3, max_dim=2, n_points=5, max_depth=2)

    for i, sample in enumerate(samples):
        print(f"\n样本 {i}:")
        print(f"  z0表达式: {sample['z0_tokens']}")
        print(f"  z1表达式: {sample['z1_tokens']}")

        # 测试tokenization
        z0_tokenized = []
        for token in sample['z0_tokens']:
            token_ids = special_tokens_manager.tokenize_expression(token)
            z0_tokenized.extend(token_ids)
            print(f"  Token '{token}' -> {token_ids}")

        print(f"  z0完整token IDs: {z0_tokenized}")
        print(f"  Token ID范围: [{min(z0_tokenized) if z0_tokenized else 'N/A'}, {max(z0_tokenized) if z0_tokenized else 'N/A'}]")

        # 检查token ID是否有效
        invalid_tokens = [tid for tid in z0_tokenized if tid >= tokenizer.vocab_size or tid < 0]
        if invalid_tokens:
            print(f"  ❌ 发现无效token IDs: {invalid_tokens}")
        else:
            print(f"  ✅ 所有token IDs都有效")

def main():
    print("开始调试数据生成和tokenization...")

    test_data_generation()
    test_tokenization()

    print("\n✅ 所有测试完成")

if __name__ == "__main__":
    main()
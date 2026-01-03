#!/usr/bin/env python3
"""
符号回归推理示例 - 从Parquet数据集加载样本并进行推理测试
"""

import numpy as np
import re
import argparse
import pandas as pd

from src.training.editflow_manager import EditFlowManager
from src.symbolic.symbolic_utils import evaluate_expression_with_constants


def reorganize_data_by_used_variables(expression_str, x_data):
    """根据表达式实际使用的变量重新组织数据

    将原始变量索引映射到从0开始的连续索引，例如：x1, x2 → x0, x1
    """
    pattern = r'x(\d+)'
    matches = re.findall(pattern, str(expression_str))

    if not matches:
        return expression_str, x_data

    var_indices = sorted(set(int(m) for m in matches))

    # 创建变量映射并重新组织数据
    var_mapping = {f'x{old_idx}': f'x{new_idx}' for new_idx, old_idx in enumerate(var_indices)}
    new_x_data = x_data[:, var_indices]

    # 更新表达式（按长度排序避免x10→x00的问题）
    new_expression = str(expression_str)
    for old_var, new_var in sorted(var_mapping.items(), key=lambda x: len(x[0]), reverse=True):
        new_expression = new_expression.replace(old_var, new_var)

    return new_expression, new_x_data


def load_sample(parquet_path, target_expr=None, sample_idx=None):
    """从Parquet数据集加载样本"""
    df = pd.read_parquet(parquet_path)

    if sample_idx is not None:
        row = df.loc[sample_idx]
        sample_idx = sample_idx
    else:
        matched = df[df['exp_gt'] == target_expr]
        if len(matched) == 0:
            raise ValueError(f"未找到表达式: {target_expr}")
        sample_idx = matched.index[0]
        row = df.loc[sample_idx]

    # 处理x_values格式
    x_values = row['x_values']
    if x_values.ndim == 1:
        x_values = np.stack(x_values)

    return {
        'x': x_values,
        'y': np.array(row['y_target']),
        'exp_gt': row['exp_gt'],
        'exp_cur1': row['exp_cur1'],
        'input_dimension': int(row['input_dimension']),
        'sample_idx': int(sample_idx)
    }


def main():
    parser = argparse.ArgumentParser(description='符号回归推理测试')
    parser.add_argument('--target_expr', type=str, default='-x1 + log(Abs(x1) + 1/1000)')
    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('--parquet_path', type=str,
                       default='data/flow_samples_100000_3dim_100pts_6depth_24len.parquet')
    parser.add_argument('--model_path', type=str, default='checkpoints/continuous_flow_final')
    args = parser.parse_args()

    # 模型配置
    model_args = type('ModelArgs', (), {
        'seed': 42,
        'base_model_name': "google-bert/bert-base-uncased",
        'condition_model_name': "settransformer",
        'cache_dir': "models/huggingface_cache",
        'use_fp16': False,
        'gradient_accumulation_steps': 1,
        'log_with': None,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_dim': 3,
        'max_expr_length': 24,
        'num_timesteps': 1,
        'num_samples': 10000,
        'batch_size': 32,
        'num_epochs': 30,
        'test_split': 0.2,
        'eval_every': 5,
        'save_every': 5,
        'n_points': 100,
        'max_depth': 5,
        'alignment_method': 'tree_edit_distance',
        'cache_size': 1000,
        'num_workers': 4,
        'save_dir': 'checkpoints',
        'debug': True,
        'hidden_dim': 512,
        'n_layers': 8,
        'n_heads': 8,
        'dropout': 0.1,
        'use_condition_injection': True,
        'condition_max_input_dim': 3,
        'condition_dim_hidden': 768,
        'condition_num_heads': 4,
        'condition_num_inds': 32,
        'condition_num_layers': 3,
        'condition_num_seeds': 32,
        'condition_dim_output': 128,
        'condition_input_normalization': False,
    })()

    # 加载样本
    print(f"\n{'='*70}")
    print(f"加载样本: {args.sample_idx if args.sample_idx else args.target_expr}")
    print(f"{'='*70}")

    sample = load_sample(args.parquet_path, args.target_expr, args.sample_idx)

    print(f"样本 #{sample['sample_idx']}")
    print(f"  目标: {sample['exp_gt']}")
    print(f"  初始: {sample['exp_cur1']}")
    print(f"  维度: {sample['input_dimension']}, 数据形状: {sample['x'].shape}")

    # 重新组织数据
    expr_gt, x_data = reorganize_data_by_used_variables(sample['exp_gt'], sample['x'])
    expr_initial, _ = reorganize_data_by_used_variables(sample['exp_cur1'], sample['x'])

    # 推理
    print(f"\n{'='*70}")
    print(f"开始推理: {args.model_path}")
    print(f"{'='*70}\n")

    manager = EditFlowManager(model_args)
    predicted_expr = manager.symbolic_regression(
        model_path=args.model_path,
        x_data=x_data,
        y_data=sample['y'],
        n_steps=1,
        input_dim=sample['input_dimension'],
        initial_expr=expr_initial
    )

    # 评估结果
    _, _, mse_initial = evaluate_expression_with_constants(expr_initial, x_data, sample['y'])
    _, _, mse_pred = evaluate_expression_with_constants(predicted_expr, x_data, sample['y'])

    print(f"\n{'='*70}")
    print(f"推理结果")
    print(f"{'='*70}")
    print(f"样本 #{sample['sample_idx']}")
    print(f"\n目标表达式: {expr_gt}")
    print(f"当前表达式: {expr_initial}")
    print(f"预测表达式: {predicted_expr}")
    print(f"\nMSE 损失:")
    print(f"  当前 → 目标: {mse_initial:.6e}")
    print(f"  预测 → 目标: {mse_pred:.6e}")

    # 判断结果
    if mse_pred < 1e-6:
        print(f"\n✅ 推理成功！")
    elif mse_pred < mse_initial:
        print(f"\n⚠️  部分改进 ({((mse_initial - mse_pred) / mse_initial * 100):.1f}%)")
    else:
        print(f"\n❌ 推理失败")


if __name__ == "__main__":
    main()

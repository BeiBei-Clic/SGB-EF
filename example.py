#!/usr/bin/env python3
"""
符号回归推理示例 - 从Parquet数据集加载样本并进行推理测试
"""

import numpy as np
import re
import argparse
import pandas as pd
import os
from pathlib import Path

from src.training.editflow_manager import EditFlowManager


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


def list_available_datasets(data_dir='data'):
    """列出可用的数据集文件"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    parquet_files = sorted(data_path.glob('*.parquet'))
    return [str(f) for f in parquet_files]


def load_sample(parquet_path, target_expr=None, sample_idx=None):
    """从Parquet数据集加载样本"""
    if not os.path.exists(parquet_path):
        available = list_available_datasets()
        raise FileNotFoundError(
            f"数据集文件不存在: {parquet_path}\n"
            f"可用的数据集:\n" + "\n".join(f"  - {f}" for f in available)
        )

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
    parser = argparse.ArgumentParser(
        description='符号回归推理测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 通过样本索引推理
  python example.py --sample_idx 0

  # 通过目标表达式推理
  python example.py --target_expr "x0 + x1"

  # 使用特定数据集
  python example.py --parquet_path data/flow_samples_1_3dim_100pts_6depth_24len.parquet --sample_idx 0

  # 列出可用数据集
  python example.py --list_datasets
        """
    )
    parser.add_argument('--target_expr', type=str, default=None,
                       help='目标表达式（与--sample_idx二选一）')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='样本索引（与--target_expr二选一）')
    parser.add_argument('--parquet_path', type=str, default=None,
                       help='Parquet数据集路径（默认使用data目录下第一个parquet文件）')
    parser.add_argument('--model_path', type=str, default='checkpoints/continuous_flow_final',
                       help='模型检查点路径')
    parser.add_argument('--list_datasets', action='store_true',
                       help='列出所有可用的数据集并退出')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据集目录（默认: data）')
    args = parser.parse_args()

    # 列出可用数据集
    if args.list_datasets:
        available = list_available_datasets(args.data_dir)
        if not available:
            print(f"在 {args.data_dir} 目录下未找到parquet数据集")
        else:
            print("可用的数据集:")
            for f in available:
                print(f"  - {f}")
        return

    # 自动选择数据集
    if args.parquet_path is None:
        available = list_available_datasets(args.data_dir)
        if not available:
            raise ValueError(
                f"在 {args.data_dir} 目录下未找到parquet数据集。"
                f"请使用 --parquet_path 指定数据集路径，或将数据集放到 {args.data_dir} 目录下"
            )
        args.parquet_path = available[0]
        print(f"使用默认数据集: {args.parquet_path}")

    # 检查参数
    if args.target_expr is None and args.sample_idx is None:
        parser.error("必须指定 --target_expr 或 --sample_idx 之一")

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

    # 将目标表达式转换为tokens（用于对比）
    import sympy as sp
    from src.symbolic.symbolic_utils import expr_to_tree
    target_expr_sympy = sp.sympify(expr_gt) if isinstance(expr_gt, str) else expr_gt
    target_tokens_str = expr_to_tree(target_expr_sympy)
    target_tokens = target_tokens_str.split(',') if target_tokens_str else []

    # 推理
    print(f"\n{'='*70}")
    print(f"开始推理: {args.model_path}")
    print(f"{'='*70}\n")

    manager = EditFlowManager(model_args)
    result = manager.symbolic_regression(
        model_path=args.model_path,
        x_data=x_data,
        y_data=sample['y'],
        n_steps=1,
        input_dim=sample['input_dimension'],
        initial_expr=expr_initial
    )

    # 提取结果信息
    initial_tokens = result['initial_tokens']
    final_tokens = result['final_tokens']
    history = result['history']
    position_actions_history = result['position_actions_history']

    print(f"\n{'='*70}")
    print(f"推理详细信息")
    print(f"{'='*70}")

    print(f"\n初始状态:")
    print(f"  表达式: {expr_initial}")
    tokens_str = ' '.join([f"[{i}]{t}" for i, t in enumerate(initial_tokens)])
    print(f"  Tokens: {tokens_str}")

    print(f"\n目标状态:")
    print(f"  表达式: {expr_gt}")
    tokens_str = ' '.join([f"[{i}]{t}" for i, t in enumerate(target_tokens)])
    print(f"  Tokens: {tokens_str}")

    print(f"\n各位置的操作选择:")
    if position_actions_history and len(position_actions_history) > 0:
        for step_idx, step_actions in enumerate(position_actions_history):
            print(f"  步骤{step_idx + 1}:")
            for pos in sorted(step_actions.keys()):
                action_type, score = step_actions[pos]
                marker = " ← 执行" if step_idx < len(history) and f"@{pos}" in history[step_idx] else ""
                print(f"    位置{pos}: {action_type}(score={score:.4f}){marker}")

    print(f"\n执行操作: {history[0] if history else 'None'}")

    print(f"\n操作后状态:")
    tokens_str = ' '.join([f"[{i}]{t}" for i, t in enumerate(final_tokens)])
    print(f"  Tokens: {tokens_str}")

    print(f"\n结果对比:")
    # 检查是否匹配
    if final_tokens == target_tokens:
        print(f"  与目标匹配: ✅ 完全匹配")
    else:
        # 计算匹配度
        min_len = min(len(final_tokens), len(target_tokens))
        matches = sum(1 for i in range(min_len) if final_tokens[i] == target_tokens[i])

        # 分析差异
        if len(final_tokens) > len(target_tokens):
            diff = f"多了{len(final_tokens) - len(target_tokens)}个token"
        elif len(final_tokens) < len(target_tokens):
            diff = f"少了{len(target_tokens) - len(final_tokens)}个token"
        else:
            diff = f"有{min_len - matches}个token不同"

        print(f"  与目标匹配: ❌ 不匹配 ({diff}, 匹配{matches}/{min_len}个token)")


if __name__ == "__main__":
    main()

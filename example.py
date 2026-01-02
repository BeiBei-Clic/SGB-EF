#!/usr/bin/env python3
"""
带调试信息的符号回归实例 - 直接调用EditFlowManager中的symbolic_regression方法

架构说明（v2.0 - 迭代优化模式）:
- 从"连续流匹配"转变为"离散编辑预测"
- 条件编码器使用目标值y_target（北极星模式）而非残差
- 推理时条件保持恒定，提供稳定的优化方向
- 模型直接学习"从当前状态到目标状态的编辑操作"
"""

import numpy as np
import re
import json
import os

from src.training.editflow_manager import EditFlowManager


def reorganize_data_by_used_variables(expression_str, x_data, y_data):
    """根据表达式实际使用的变量重新组织数据

    Args:
        expression_str: 真实表达式字符串
        x_data: 原始输入数据 (n_samples, n_dims)
        y_data: 目标值 (n_samples,)

    Returns:
        new_expression: 重新映射后的表达式
        new_x_data: 重新组织后的输入数据
        new_used_vars: 新表达式中使用的变量名
    """
    # 查找所有使用的变量
    pattern = r'x(\d+)'
    matches = re.findall(pattern, str(expression_str))

    if not matches:
        return expression_str, x_data, []

    # 获取使用的变量索引（排序）
    var_indices = sorted(set(int(m) for m in matches))
    old_used_vars = [f'x{i}' for i in var_indices]

    # 创建变量映射：原始变量 -> 新变量
    # 例如：如果原始使用 x1, x2，则映射为 x1->x0, x2->x1
    var_mapping = {}
    for new_idx, old_idx in enumerate(var_indices):
        var_mapping[f'x{old_idx}'] = f'x{new_idx}'

    # 重新组织x_data：只保留使用的列，并按原始顺序排列
    new_x_data = x_data[:, var_indices]

    # 更新表达式字符串
    # 按照变量名长度从长到短排序，避免短变量名污染长变量名
    # 例如：先替换 x10，再替换 x1，避免 x10 被错误替换为 x00
    new_expression = str(expression_str)
    sorted_vars = sorted(var_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    for old_var, new_var in sorted_vars:
        new_expression = new_expression.replace(old_var, new_var)

    # 获取新表达式使用的变量
    new_matches = re.findall(pattern, new_expression)
    new_var_indices = sorted(set(int(m) for m in new_matches))
    new_used_vars = [f'x{i}' for i in new_var_indices]

    return new_expression, new_x_data, new_used_vars


def main():
    # 设置参数
    args = type('Args', (), {
        'seed': 42,
        'base_model_name': "google-bert/bert-base-uncased",  # 已弃用，保留用于兼容性
        'condition_model_name': "settransformer",  # 使用SetTransformer架构编码目标值
        'cache_dir': "models/huggingface_cache",  # 模型缓存目录
        'use_fp16': False,  # 推理时关闭混合精度
        'gradient_accumulation_steps': 1,  # 推理时不需要梯度累积
        'log_with': None,  # 不使用外部日志工具（如tensorboard、wandb等）
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_dim': 3,  # 支持的最大输入维度
        'max_expr_length': 24,  # 最大表达式长度
        "num_timesteps": 1,  # 已弃用：新架构固定t=0，不需要多时间步采样
        # 训练相关参数（推理时也需要，用于兼容性）
        'num_samples': 10000,  # 训练样本数（推理时不使用）
        'batch_size': 32,  # 批次大小（推理时不使用）
        'num_epochs': 30,  # 训练轮数（推理时不使用）
        'test_split': 0.2,  # 测试集比例（推理时不使用）
        'eval_every': 5,  # 评估频率（推理时不使用）
        'save_every': 5,  # 保存频率（推理时不使用）
        'n_points': 100,  # 每个样本的点数（推理时不使用）
        'max_depth': 5,  # 最大表达式深度（推理时不使用）
        'alignment_method': 'tree_edit_distance',  # 对齐方法（推理时不使用）
        'cache_size': 1000,  # 数据缓存大小（推理时不使用）
        'num_workers': 4,  # 数据加载工作进程数（推理时不使用）
        'save_dir': 'checkpoints',  # 模型保存目录（推理时不使用）
        'debug': True,  # 【启用调试模式以查看详细推理过程】
        # LLaMA模型架构参数
        'hidden_dim': 512,  # LLaMA隐藏层维度 (匹配checkpoint)
        'n_layers': 8,  # LLaMA Transformer层数 (匹配checkpoint)
        'n_heads': 8,  # LLaMA注意力头数
        'dropout': 0.1,  # Dropout比率
        'use_condition_injection': True,  # 是否使用交叉注意力条件注入
        # SetTransformer参数（条件编码器）
        'condition_max_input_dim': 3,  # 条件编码器支持的最大输入维度
        'condition_dim_hidden': 768,  # SetTransformer隐藏层维度
        'condition_num_heads': 4,  # 注意力头数
        'condition_num_inds': 32,  # 诱导点数
        'condition_num_layers': 3,  # SetTransformer层数
        'condition_num_seeds': 32,  # 输出序列长度（特征向量数量）
        'condition_dim_output': 128,  # 已弃用（现在输出序列而非单个向量）
        'condition_input_normalization': False,  # 是否对输入进行归一化
    })()

    manager = EditFlowManager(args)

    # 直接加载1样本数据集
    train_data_file = "data/flow_samples_1_3dim_100pts_6depth_24len.txt"
    sample = None
    with open(train_data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i == 1:  # 使用第1行（也是唯一一行）
                data = json.loads(line.strip())
                sample = {
                    'x': np.array(data['x_values']),
                    'y': np.array(data['y_target']),
                    'exp_gt': data['exp_gt'],
                    'exp_cur1': data['exp_cur1'],
                    'input_dimension': data['input_dimension'],
                    'z0_tokens': data['z0_tokens'],
                    'z1_tokens': data['z1_tokens']
                }
                target_expr = data['exp_gt']
                break

    # 提取数据
    x_data = sample['x']
    y_data = sample['y']
    initial_expr = sample['exp_cur1']

    print(f"\n{'='*70}")
    print(f"【数据集样本信息】")
    print(f"  目标表达式: {target_expr}")
    print(f"  初始表达式: {initial_expr}")
    print(f"  x_data 形状: {x_data.shape}")
    print(f"  y_data 形状: {y_data.shape}")
    print(f"{'='*70}\n")

    # 根据表达式实际使用的变量重新组织数据
    new_expr_gt, x_data_reorganized, new_used_vars = reorganize_data_by_used_variables(
        target_expr, x_data, y_data
    )

    # 模型路径
    model_path = "checkpoints/continuous_flow_final"

    # 检查是否有训练好的检查点
    import os

    if not os.path.exists(model_path):
        print(f"\n⚠️  检查点不存在: {model_path}")
        print("跳过推理步骤。请先训练模型：")
        print("  python train.py --num_samples 10000 --num_epochs 30")
        return

    # 执行符号回归
    predicted_expression = manager.symbolic_regression(
        model_path=model_path,
        x_data=x_data_reorganized,
        y_data=y_data,
        n_steps=1,
        initial_expr=initial_expr
    )

    print(f"\n{'='*70}")
    print(f"【推理结果】")
    print(f"  目标表达式: {new_expr_gt}")
    print(f"  预测表达式: {predicted_expression}")
    print(f"{'='*70}")

    # 验证表达式质量
    from src.symbolic.symbolic_utils import evaluate_expression_with_constants

    # 验证目标表达式
    gt_success, gt_optimized_expr, gt_mse = evaluate_expression_with_constants(
        tree_str=new_expr_gt,
        x_values=x_data_reorganized,
        y_values=y_data
    )

    print(f"\n【目标表达式质量】")
    print(f"  MSE: {gt_mse:.6f}")
    if gt_optimized_expr != new_expr_gt:
        print(f"  优化后: {gt_optimized_expr}")

    # 验证预测表达式
    pred_success, pred_optimized_expr, pred_mse = evaluate_expression_with_constants(
        tree_str=predicted_expression,
        x_values=x_data_reorganized,
        y_values=y_data
    )

    print(f"\n【预测表达式质量】")
    print(f"  MSE: {pred_mse:.6f}")
    if pred_optimized_expr != predicted_expression:
        print(f"  优化后: {pred_optimized_expr}")


if __name__ == "__main__":
    main()
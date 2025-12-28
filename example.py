#!/usr/bin/env python3
"""
带调试信息的符号回归实例 - 直接调用EditFlowManager中的symbolic_regression方法
"""

import numpy as np
import re
import sympy as sp

from src.symbolic.data_generator import generate_sample
from src.training.editflow_manager import EditFlowManager
from src.utils.logger import Logger


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

    print(f"原始表达式使用的变量: {old_used_vars}")

    # 创建变量映射：原始变量 -> 新变量
    # 例如：如果原始使用 x1, x2，则映射为 x1->x0, x2->x1
    var_mapping = {}
    for new_idx, old_idx in enumerate(var_indices):
        var_mapping[f'x{old_idx}'] = f'x{new_idx}'

    print(f"变量映射: {var_mapping}")

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
    # 创建 Logger 实例
    logger = Logger(enabled=True)

    print("=== 符号回归实例 ===")

    # 记录开始日志
    logger.log("INFERENCE_START", "开始符号回归实例", "example")

    # 设置参数
    args = type('Args', (), {
        'seed': 42,
        'base_model_name': "google-bert/bert-base-uncased",
        'condition_model_name': "settransformer",  # 现在使用SetTransformer架构
        'cache_dir': "models/huggingface_cache",  # 模型缓存目录
        'use_fp16': False,  # 推理时关闭混合精度
        'gradient_accumulation_steps': 1,  # 推理时不需要梯度累积
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_dim': 3,  # 添加最大维度参数，确保覆盖变量范围
        'max_expr_length': 12, # 最大表达式长度
        "num_timesteps": 1,
        # LLaMA模型架构参数
        'hidden_dim': 512,  # LLaMA隐藏层维度 (匹配checkpoint)
        'n_layers': 8,  # LLaMA Transformer层数 (匹配checkpoint)
        'n_heads': 8,  # LLaMA注意力头数
        'dropout': 0.1,  # Dropout比率
        'use_condition_injection': True,  # 是否使用交叉注意力条件注入
        # SetTransformer参数
        'condition_dim_hidden': 768,  # 匹配 BERT 的 hidden_size
        'condition_num_heads': 4,
        'condition_num_inds': 32,
        'condition_num_layers': 3,
        'condition_num_seeds': 32,  # 输出序列长度
        'condition_dim_output': 128,  # 已弃用
        'condition_input_normalization': False,
    })()

    manager = EditFlowManager(args)

    # 生成测试数据
    print("生成测试数据...")
    logger.log("DATA_GENERATION", "开始生成测试数据", "example")

    # 使用不同的种子生成不同的测试数据
    import sys

    sample = generate_sample(input_dimension=3, n_points=100, max_depth=5, seed=5)
    x_data = np.array(sample['x'])
    y_data = np.array(sample['y'])

    logger.log("DATA_INFO", f"真实表达式: {sample['exp_gt']} | x_data形状: {x_data.shape} | y_data形状: {y_data.shape}", "example")

    print(f"\n原始数据信息:")
    print(f"真实表达式: {sample['exp_gt']}")
    print(f"x_data 形状: {x_data.shape}")
    print(f"y_data 形状: {y_data.shape}")

    # 根据表达式实际使用的变量重新组织数据
    new_expr_gt, x_data_reorganized, new_used_vars = reorganize_data_by_used_variables(
        sample['exp_gt'], x_data, y_data
    )

    # 更新sample以使用新的表达式
    sample['exp_gt'] = new_expr_gt

    # 模型路径
    model_path = "checkpoints/checkpoint_epoch_5"

    # 执行符号回归（使用重新组织后的数据）
    # 使用束搜索来扩大搜索范围，提高发现更好表达式的概率
    logger.log("INFERENCE_START", f"开始符号回归推理 模型路径: {model_path} | 推理步数: 30 | 束大小: 5", "example")

    predicted_expression = manager.symbolic_regression(
        model_path=model_path,
        x_data=x_data_reorganized,  # 使用重新组织后的数据
        y_data=y_data,
        n_steps=30,      # 推理步数
        beam_size=5      # 束搜索宽度，每步保留5个最佳候选
    )

    logger.log("INFERENCE_COMPLETE", f"符号回归完成 | 预测表达式: {predicted_expression}", "example")
    print(f"\n最终结果对比:")
    print(f"真实表达式: {new_expr_gt}")  # 使用更新后的表达式
    print(f"预测表达式: {predicted_expression}")

    # 验证表达式质量（支持常数优化）
    if predicted_expression:
        from src.symbolic.symbolic_utils import evaluate_expression_with_constants
        try:
            # 使用常数优化评估预测表达式
            pred_success, pred_optimized_expr, pred_mse = evaluate_expression_with_constants(
                tree_str=predicted_expression,
                x_values=x_data_reorganized,
                y_values=y_data
            )

            if pred_success and pred_optimized_expr is not None:
                print(f"\n预测质量:")
                print(f"  MSE: {pred_mse:.6f}")
                print(f"  优化后的表达式: {pred_optimized_expr}")

                # 也计算真实表达式的质量作为对比
                gt_success, gt_optimized_expr, gt_mse = evaluate_expression_with_constants(
                    tree_str=new_expr_gt,
                    x_values=x_data_reorganized,
                    y_values=y_data
                )
                if gt_success and gt_optimized_expr is not None:
                    print(f"\n真实表达式质量:")
                    print(f"  MSE: {gt_mse:.6f}")
                    print(f"  优化后的表达式: {gt_optimized_expr}")
            else:
                print(f"\n验证失败: 无法计算预测表达式")
        except Exception as e:
            print(f"\n验证失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
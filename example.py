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
import sympy as sp
import json
import os

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


def load_sample_from_train_data(target_expression):
    """从训练数据集中查找包含特定表达式的样本

    Args:
        target_expression: 目标表达式字符串, 如 "sqrt(Abs(x0))"

    Returns:
        sample: 包含以下键的字典
            - 'x': numpy array, 输入数据 (n_points, input_dim)
            - 'y': numpy array, 目标值 (n_points,)
            - 'exp_gt': str, 目标表达式
            - 'exp_cur1': str, 初始corrupted表达式
            - 'input_dimension': int, 输入维度
        如果未找到则返回 None
    """
    # 确定训练数据文件路径
    # 优先使用100样本数据集(实际训练过的数据集)
    train_data_file = "data/flow_samples_100_3dim_100pts_6depth_24len.txt"

    if not os.path.exists(train_data_file):
        print(f"错误: 找不到训练数据文件")
        print(f"尝试的路径: {train_data_file}")
        return None

    print(f"从训练数据文件中查找样本: {train_data_file}")
    print(f"目标表达式: {target_expression}")

    # 遍历数据文件查找匹配的样本
    with open(train_data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                exp_gt = sample.get('exp_gt', '')

                # 去掉空格后比较
                if exp_gt.replace(' ', '') == target_expression.replace(' ', ''):
                    print(f"\n✓ 找到匹配样本 (第{line_num}行)")
                    print(f"  目标表达式: {exp_gt}")
                    print(f"  初始表达式: {sample.get('exp_cur1', '')}")
                    print(f"  输入维度: {sample.get('input_dimension', 1)}")

                    # 提取并转换数据
                    x_values = np.array(sample['x_values'])  # (n_points, input_dim)
                    y_target = np.array(sample['y_target'])  # (n_points,)

                    print(f"  x_values形状: {x_values.shape}")
                    print(f"  y_target形状: {y_target.shape}")
                    print(f"  x范围: [{x_values.min():.3f}, {x_values.max():.3f}]")
                    print(f"  y范围: [{y_target.min():.3f}, {y_target.max():.3f}]")

                    return {
                        'x': x_values,
                        'y': y_target,
                        'exp_gt': exp_gt,
                        'exp_cur1': sample.get('exp_cur1', ''),
                        'input_dimension': sample.get('input_dimension', 1)
                    }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: 第{line_num}行解析失败: {e}")
                continue

    print(f"\n未找到表达式 '{target_expression}' 的样本")
    return None


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
        # 多阈值推理参数
        'action_thresholds': None,  # 使用单最佳操作模式（每步只采纳分数最高的操作）
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

    # 创建 Logger 实例，传入debug_mode参数
    logger = Logger(enabled=True, debug_mode=args.debug)

    # 记录开始日志（使用level=3表示推理日志，受debug控制）
    logger.log("INFERENCE_START", "开始符号回归实例", "example", level=3)

    manager = EditFlowManager(args)

    # 从训练数据集加载样本 - 使用1样本数据集
    logger.log("DATA_LOADING", "从训练数据集加载样本 (目标表达式: Abs(x1), 初始: 0)", "example", level=3)

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

    if sample is None:
        print(f"\n⚠️  错误: 无法从训练数据集加载表达式 '{target_expr}' 的样本")
        print("请检查:")
        print("  1. 训练数据文件是否存在")
        print("  2. 数据集中是否包含该表达式")
        print("  3. 表达式字符串是否正确")
        return

    # 提取数据
    x_data = sample['x']
    y_data = sample['y']
    # 使用exp_cur1作为初始表达式（而不是直接使用z0_tokens）
    # 因为推理时会将表达式字符串转换为tokens，并经过tokenizer编码
    # 这样才能与训练时的处理保持一致（添加BOS和padding）
    initial_expr = sample['exp_cur1']
    initial_tokens = sample['z0_tokens']  # 仅用于日志显示

    print(f"\n{'='*70}")
    print(f"使用数据集第1行样本进行测试:")
    print(f"{'='*70}")
    print(f"【目标表达式】: {target_expr}")
    print(f"  → 这是模型需要推导出的最终表达式")
    print(f"【初始表达式】: {initial_expr}")
    print(f"  → 这是模型的起点表达式（corrupted version）")
    print(f"【z0_tokens (初始状态)】: {initial_tokens}")
    print(f"【z1_tokens (目标状态)】: {sample['z1_tokens']}")
    print(f"【正确的编辑操作】: DELETE位置0的constant, INSERT abs和x1")
    print(f"{'='*70}")

    logger.log("DATA_INFO",
               f"目标表达式: {target_expr} | 初始表达式: {initial_expr} | "
               f"x_data形状: {x_data.shape} | y_data形状: {y_data.shape}",
               "example", level=3)

    print(f"\n{'='*70}")
    print(f"【调试信息】从训练数据集加载的样本:")
    print(f"{'='*70}")
    print(f"目标表达式 (Ground Truth): {target_expr}")
    print(f"初始表达式 (Initial):      {initial_expr}")
    print(f"x_data 形状: {x_data.shape}")
    print(f"y_data 形状: {y_data.shape}")
    print(f"{'='*70}\n")

    # 根据表达式实际使用的变量重新组织数据
    new_expr_gt, x_data_reorganized, new_used_vars = reorganize_data_by_used_variables(
        target_expr, x_data, y_data
    )

    # 模型路径
    model_path = "checkpoints/continuous_flow_final"

    # 执行符号回归（使用重新组织后的数据）
    # 支持多阈值推理：每一步采纳所有高于阈值的操作
    logger.log("INFERENCE_START",
               f"开始符号回归推理 | 模型路径: {model_path} | 推理步数: {5} | "
               f"方法: 贪婪搜索 | 架构: 迭代优化（北极星模式） | "
               f"阈值: {args.action_thresholds}",
               "example", level=3)

    # 检查是否有训练好的检查点
    import os
    result = None  # 初始化result变量
    is_multi_threshold = False

    if not os.path.exists(model_path):
        print(f"\n⚠️  检查点不存在: {model_path}")
        print("跳过推理步骤。请先训练模型：")
        print("  python train.py --num_samples 10000 --num_epochs 30")
        predicted_expression = ""
    else:
        result = manager.symbolic_regression(
            model_path=model_path,
            x_data=x_data_reorganized,  # 使用重新组织后的数据
            y_data=y_data,
            n_steps=5,      # 推理步数
            initial_expr=initial_expr  # 传入表达式字符串（而非token列表）
        )

        # 判断是否为多阈值模式
        is_multi_threshold = isinstance(result, dict)

        if is_multi_threshold:
            # 多阈值模式：result 是 {threshold: expression} 字典
            predicted_expression = list(result.values())[0]  # 最低阈值的结果

            print(f"\n{'='*70}")
            print(f"【推理结果】多阈值模式:")
            print(f"  目标表达式: {new_expr_gt}")
            print(f"  预测表达式: {predicted_expression}")
            print(f"  使用阈值数: {len(result)}")
            print(f"{'='*60}")
        else:
            # 单阈值模式：result 是单个表达式字符串
            predicted_expression = result
            print(f"\n{'='*70}")
            print(f"【推理结果】单阈值模式:")
            print(f"  目标表达式: {new_expr_gt}")
            print(f"  预测表达式: {predicted_expression}")
            print(f"{'='*60}")

    logger.log("INFERENCE_COMPLETE", f"符号回归完成 | 预测表达式: {predicted_expression} | 多阈值模式: {is_multi_threshold}", "example", level=3)

    # 验证表达式质量（支持常数优化）
    if predicted_expression:
        from src.symbolic.symbolic_utils import evaluate_expression_with_constants
        try:
            # 计算真实表达式的质量作为对比
            gt_success, gt_optimized_expr, gt_mse = evaluate_expression_with_constants(
                tree_str=new_expr_gt,
                x_values=x_data_reorganized,
                y_values=y_data
            )

            if gt_success and gt_optimized_expr is not None:
                print(f"\n真实表达式质量:")
                print(f"  MSE: {gt_mse:.6f}")
                print(f"  优化后的表达式: {gt_optimized_expr}")

            if is_multi_threshold and result:
                # 多阈值模式：验证所有阈值的结果，找出最佳结果
                results_with_mse = []
                for threshold, expression in sorted(result.items(), reverse=True):
                    pred_success, pred_optimized_expr, pred_mse = evaluate_expression_with_constants(
                        tree_str=expression,
                        x_values=x_data_reorganized,
                        y_values=y_data
                    )

                    if pred_success and pred_optimized_expr is not None:
                        results_with_mse.append((threshold, expression, pred_mse, pred_optimized_expr))

                # 按MSE排序并显示最佳结果
                if results_with_mse:
                    results_with_mse.sort(key=lambda x: x[2])  # 按MSE排序
                    best_threshold, best_expr, best_mse, best_optimized = results_with_mse[0]

                    print(f"\n最佳结果:")
                    print(f"  表达式: {best_expr}")
                    print(f"  MSE: {best_mse:.6f}")
                    print(f"  优化后: {best_optimized}")

            else:
                # 单阈值模式：只验证预测表达式
                pred_success, pred_optimized_expr, pred_mse = evaluate_expression_with_constants(
                    tree_str=predicted_expression,
                    x_values=x_data_reorganized,
                    y_values=y_data
                )

                if pred_success and pred_optimized_expr is not None:
                    print(f"\n预测表达式质量:")
                    print(f"  MSE: {pred_mse:.6f}")
                    print(f"  优化后的表达式: {pred_optimized_expr}")
                else:
                    print(f"\n验证失败: 无法计算预测表达式")

        except Exception as e:
            print(f"\n验证失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
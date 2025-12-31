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
        'action_thresholds': "0.1,0.05,0.01",  # 多阈值推理的操作采纳阈值（逗号分隔），None表示单最佳操作模式
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
        'debug': False,  # 是否启用调试模式
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

    print("=== 符号回归实例 ===")

    # 记录开始日志（使用level=3表示推理日志，受debug控制）
    logger.log("INFERENCE_START", "开始符号回归实例", "example", level=3)

    manager = EditFlowManager(args)

    # 生成测试数据 - 使用 x1*x2 作为目标表达式
    print("生成测试数据...")
    logger.log("DATA_GENERATION", "开始生成测试数据 (目标表达式: x1*x2)", "example", level=3)

    # 直接构造 x1*x2 的数据
    np.random.seed(42)
    n_points = 100
    input_dimension = 3  # 需要3维，这样才有x1和x2

    # 生成随机数据
    x_data = np.random.randn(n_points, input_dimension) * 2  # 使用标准正态分布
    y_data = x_data[:, 1] * x_data[:, 2]  # x1 * x2

    # 目标表达式字符串
    target_expr = "x1*x2"

    logger.log("DATA_INFO", f"目标表达式: {target_expr} | x_data形状: {x_data.shape} | y_data形状: {y_data.shape}", "example", level=3)

    print(f"\n数据信息:")
    print(f"目标表达式: {target_expr}")
    print(f"初始表达式: {target_expr}")  # 初始表达式和目标表达式相同
    print(f"x_data 形状: {x_data.shape}")
    print(f"y_data 形状: {y_data.shape}")

    # 根据表达式实际使用的变量重新组织数据
    new_expr_gt, x_data_reorganized, new_used_vars = reorganize_data_by_used_variables(
        target_expr, x_data, y_data
    )

    # 创建一个类似sample的字典以保持兼容性
    sample = {
        'exp_gt': new_expr_gt,
        'x': x_data_reorganized,
        'y': y_data
    }

    # 模型路径
    model_path = "checkpoints/checkpoint_epoch_5"

    # 执行符号回归（使用重新组织后的数据）
    # 支持多阈值推理：每一步采纳所有高于阈值的操作
    logger.log("INFERENCE_START",
               f"开始符号回归推理 | 模型路径: {model_path} | 推理步数: {20} | "
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
            n_steps=20      # 推理步数
        )

        # 判断是否为多阈值模式
        is_multi_threshold = isinstance(result, dict)

        if is_multi_threshold:
            # 多阈值模式：result 是 {threshold: expression} 字典
            print(f"\n{'='*60}")
            print(f"多阈值推理结果 (共 {len(result)} 个阈值)")
            print(f"{'='*60}")

            # 打印所有阈值的结果
            for threshold, expression in sorted(result.items(), reverse=True):
                print(f"\n阈值 {threshold:.4f}: {expression}")

            # 选择MSE最低的结果作为最终预测
            # 这里简单选择第一个（最低阈值），实际使用中可以根据MSE排序选择
            predicted_expression = list(result.values())[0]  # 最低阈值的结果

            print(f"\n{'='*60}")
            print(f"最终结果对比 (架构v2.0 - 多阈值推理模式)")
            print(f"{'='*60}")
            print(f"真实表达式: {new_expr_gt}")
            print(f"预测表达式: {predicted_expression}")
            print(f"使用阈值数: {len(result)}")
            print(f"{'='*60}")
        else:
            # 单阈值模式：result 是单个表达式字符串
            predicted_expression = result
            print(f"\n{'='*60}")
            print(f"最终结果对比 (架构v2.0 - 单最佳操作模式)")
            print(f"{'='*60}")
            print(f"真实表达式: {new_expr_gt}")  # 使用更新后的表达式
            print(f"预测表达式: {predicted_expression}")
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
                # 多阈值模式：验证所有阈值的结果
                print(f"\n{'='*60}")
                print(f"所有阈值的结果质量评估")
                print(f"{'='*60}")

                results_with_mse = []
                for threshold, expression in sorted(result.items(), reverse=True):
                    pred_success, pred_optimized_expr, pred_mse = evaluate_expression_with_constants(
                        tree_str=expression,
                        x_values=x_data_reorganized,
                        y_values=y_data
                    )

                    if pred_success and pred_optimized_expr is not None:
                        results_with_mse.append((threshold, expression, pred_mse, pred_optimized_expr))
                        print(f"\n阈值 {threshold:.4f}:")
                        print(f"  表达式: {expression}")
                        print(f"  MSE: {pred_mse:.6f}")
                        print(f"  优化后: {pred_optimized_expr}")

                # 按MSE排序并显示最佳结果
                if results_with_mse:
                    results_with_mse.sort(key=lambda x: x[2])  # 按MSE排序
                    best_threshold, best_expr, best_mse, best_optimized = results_with_mse[0]

                    print(f"\n{'='*60}")
                    print(f"最佳结果 (MSE最低)")
                    print(f"{'='*60}")
                    print(f"阈值: {best_threshold:.4f}")
                    print(f"表达式: {best_expr}")
                    print(f"MSE: {best_mse:.6f}")
                    print(f"优化后: {best_optimized}")
                    print(f"{'='*60}")

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
#!/usr/bin/env python3
"""
带调试信息的符号回归实例 - 直接调用EditFlowManager中的symbolic_regression方法
可以直接运行，无需accelerate launch命令
"""

import os
import numpy as np
from datetime import datetime

from src.symbolic.data_generator import generate_sample
from src.training.editflow_manager import EditFlowManager


def main():
    print("=== 带调试信息的符号回归实例 (直接运行模式) ===")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 设置分布式训练环境变量（避免警告）
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"  # 推理时不需要混合精度

    # 设置参数
    args = type('Args', (), {
        'seed': 42,
        'base_model_name': "google-bert/bert-base-uncased",
        'condition_model_name': "nomic-ai/nomic-embed-text-v1.5",  # 条件嵌入模型名称
        'cache_dir': "models/huggingface_cache",  # 模型缓存目录
        'use_fp16': False,  # 推理时关闭混合精度
        'gradient_accumulation_steps': 1,  # 推理时不需要梯度累积
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_dim': 10,  # 添加最大维度参数，确保覆盖变量范围
        'max_expr_length': 12,  # 最大表达式长度
        'mixed_precision': 'no'  # 明确设置混合精度为关闭
    })()

    manager = EditFlowManager(args)

    # 生成测试数据
    print("生成测试数据...")
    sample = generate_sample(input_dimension=3, n_points=100, max_depth=5)
    x_data = np.array(sample['x'])
    y_data = np.array(sample['y'])

    print(f"\n数据信息:")
    print(f"真实表达式: {sample['exp_gt']}")
    print(f"x_data 形状: {x_data.shape}")
    print(f"y_data 形状: {y_data.shape}")

    # 模型路径
    model_path = "checkpoints/editflow_epoch_10.pth"

    # 执行符号回归（会自动推断input_dim并生成动态初始表达式）
    predicted_expression = manager.symbolic_regression(
        model_path=model_path,
        x_data=x_data,
        y_data=y_data,
        debug_mode=False,  # 开启调试模式
        n_steps=30  # 减少步数以便观察
    )

    print(f"\n最终结果对比:")
    print(f"真实表达式: {sample['exp_gt']}")
    print(f"预测表达式: {predicted_expression}")


if __name__ == "__main__":
    main()
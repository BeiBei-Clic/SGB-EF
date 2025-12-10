#!/usr/bin/env python3
"""
带调试信息的符号回归实例 - 直接调用EditFlowManager中的symbolic_regression方法
"""

import numpy as np

from src.symbolic.data_generator import generate_sample
from src.training.editflow_manager import EditFlowManager


def main():
    print("=== 带调试信息的符号回归实例 ===")

    # 设置参数
    args = type('Args', (), {
        'seed': 42,
        'base_model_name': "google-bert/bert-base-uncased",
        'condition_model_name': "Qwen/Qwen3-Embedding-0.6B",  # 条件嵌入模型名称
        'cache_dir': "models/huggingface_cache",  # 模型缓存目录
        'use_data_parallel': False,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
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
        debug_mode=True,  # 开启调试模式
        n_steps=30  # 减少步数以便观察
    )

    print(f"\n最终结果对比:")
    print(f"真实表达式: {sample['exp_gt']}")
    print(f"预测表达式: {predicted_expression}")


if __name__ == "__main__":
    main()
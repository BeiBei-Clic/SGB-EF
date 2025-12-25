#!/usr/bin/env python3
"""
带调试信息的符号回归实例 - 直接调用EditFlowManager中的symbolic_regression方法
"""

import numpy as np

from src.symbolic.data_generator import generate_sample
from src.training.editflow_manager import EditFlowManager
from src.utils.logger import Logger


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
        # SetTransformer参数
        'condition_dim_hidden': 128,
        'condition_num_heads': 4,
        'condition_num_inds': 32,
        'condition_num_layers': 3,
        'condition_num_seeds': 1,
        'condition_dim_output': 128,
        'condition_input_normalization': False,
        'condition_use_sinusoidal_encoding': False,  # 禁用正弦编码,直接使用原始残差值
    })()

    manager = EditFlowManager(args)

    # 生成测试数据
    print("生成测试数据...")
    logger.log("DATA_GENERATION", "开始生成测试数据", "example")
    # seed=None 表示每次运行生成不同的随机数据
    sample = generate_sample(input_dimension=3, n_points=100, max_depth=5, seed=2)
    x_data = np.array(sample['x'])
    y_data = np.array(sample['y'])

    logger.log("DATA_INFO", f"真实表达式: {sample['exp_gt']} | x_data形状: {x_data.shape} | y_data形状: {y_data.shape}", "example")

    print(f"\n数据信息:")
    print(f"真实表达式: {sample['exp_gt']}")
    print(f"x_data 形状: {x_data.shape}")
    print(f"y_data 形状: {y_data.shape}")

    # 模型路径
    model_path = None

    # 执行符号回归（会自动推断input_dim并生成动态初始表达式）
    logger.log("INFERENCE_START", f"开始符号回归推理 模型路径: {model_path} | 推理步数: 30", "example")
    predicted_expression = manager.symbolic_regression(
        model_path=model_path,
        x_data=x_data,
        y_data=y_data,
        n_steps=30  # 减少步数以便观察
    )

    logger.log("INFERENCE_COMPLETE", f"符号回归完成 | 预测表达式: {predicted_expression}", "example")
    print(f"\n最终结果对比:")
    print(f"真实表达式: {sample['exp_gt']}")
    print(f"预测表达式: {predicted_expression}")


if __name__ == "__main__":
    main()
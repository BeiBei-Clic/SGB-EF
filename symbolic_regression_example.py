#!/usr/bin/env python3
"""
简单的符号回归实例
"""

import numpy as np

from src.symbolic.data_generator import generate_sample
from src.training.editflow_manager import EditFlowManager


class SimpleArgs:
    def __init__(self):
        self.seed = 42
        self.base_model_name = "google-bert/bert-base-uncased"
        self.use_data_parallel = False
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5


def main():
    print("=== 符号回归实例 ===")

    # 参数设置
    args = SimpleArgs()
    manager = EditFlowManager(args)

    # 生成测试数据
    print("生成测试数据...")
    sample = generate_sample(input_dimension=2, n_points=100, max_depth=3)

    x_data = np.array(sample['x'])
    y_data = np.array(sample['y'])

    print(f"真实表达式: {sample['exp_gt']}")

    # 模型路径
    model_path = "checkpoints/editflow_epoch_10.pth"

    # 执行符号回归
    predicted_expression = manager.symbolic_regression(
        model_path=model_path,
        x_data=x_data,
        y_data=y_data
    )

    print(f"预测表达式: {predicted_expression}")


if __name__ == "__main__":
    main()
"""
EditFlow符号回归预训练主脚本
实现基于残差条件的编辑流模型训练
"""

import argparse
import os
import random
import numpy as np
import torch

from src.training.editflow_trainer import EditFlowTrainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="训练EditFlow符号回归模型")

    # 数据参数
    parser.add_argument("--num_samples", type=int, default=1000, help="训练样本数")
    parser.add_argument("--max_dim", type=int, default=10, help="最大输入维度")
    parser.add_argument("--n_points", type=int, default=100, help="数据点数量")
    parser.add_argument("--max_depth", type=int, default=4, help="表达式最大深度")

    # 模型参数
    parser.add_argument("--base_model_name", type=str, default="openai-community/gpt2", help="基础模型名称")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--save_every", type=int, default=10, help="每多少轮保存一次")

    args = parser.parse_args()

    print("=== EditFlow符号回归预训练 ===")
    print(f"样本数: {args.num_samples}")
    print(f"最大维度: {args.max_dim}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"基础模型: {args.base_model_name}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建训练器并开始训练
    trainer = EditFlowTrainer(args)
    model, condition_encoder = trainer.train()

    print("\n训练完成!")


if __name__ == "__main__":
    main()
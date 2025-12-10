"""
EditFlow连续流符号回归预训练主脚本
实现基于多步连续流匹配的编辑流模型训练
"""

import argparse
import os
import random
import numpy as np
import torch

from src.training.editflow_manager import EditFlowManager
from src.utils.gpu_monitor import display_gpu_info
from src.utils.special_tokens import SpecialTokensManager


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # 设置CUDA调试模式
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description="训练EditFlow符号回归模型")

    # 数据参数
    parser.add_argument("--num_samples", type=int, default=10, help="训练样本数")
    parser.add_argument("--max_dim", type=int, default=10, help="最大输入维度")
    parser.add_argument("--n_points", type=int, default=100, help="数据点数量")
    parser.add_argument("--max_depth", type=int, default=4, help="表达式最大深度")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例 (0.0-1.0)")
    parser.add_argument("--eval_every", type=int, default=5, help="每多少轮评估一次测试集")

    # 模型参数
    parser.add_argument("--base_model_name", type=str, default="google-bert/bert-base-uncased", help="基础模型名称")
    parser.add_argument("--condition_model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="条件嵌入模型名称")
    parser.add_argument("--cache_dir", type=str, default="models/huggingface_cache", help="模型缓存目录")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=6, help="批次大小 (每个GPU)")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--save_every", type=int, default=10, help="每多少轮保存一次")

    # 多GPU参数
    parser.add_argument("--use_data_parallel", action="store_true", default=True, help="是否使用多GPU并行训练")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")

    args = parser.parse_args()

    print("=== EditFlow符号回归预训练 ===")
    print(f"样本数: {args.num_samples}")
    print(f"最大维度: {args.max_dim}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"测试集比例: {args.test_split}")
    print(f"评估频率: 每{args.eval_every}轮")
    print(f"基础模型: {args.base_model_name}")
    print(f"条件嵌入模型: {args.condition_model_name}")
    print(f"多GPU并行: {args.use_data_parallel}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")

    # 显示GPU信息
    display_gpu_info()

    # 设置随机种子
    set_seed(args.seed)

    # 创建EditFlow管理器并开始训练
    manager = EditFlowManager(args)
    model, condition_encoder = manager.train()

    print("\nEditFlow训练完成!")


if __name__ == "__main__":
    main()
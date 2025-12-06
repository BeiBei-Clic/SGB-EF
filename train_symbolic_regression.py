"""
符号回归EditFlow训练主脚本
"""

import torch
import argparse
from torch.utils.data import DataLoader
import random
import os

from src.symbolic.data_generator import SymbolicRegressionDataGenerator
from src.training.trainer import EditFlowTrainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="训练符号回归EditFlow模型")

    # 数据参数
    parser.add_argument("--input_dim", type=int, default=1, help="输入维度")
    parser.add_argument("--max_depth", type=int, default=4, help="表达式最大深度")
    parser.add_argument("--n_points", type=int, default=100, help="数据点数量")
    parser.add_argument("--corruption_prob", type=float, default=0.5, help="表达式破坏概率")

    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="训练设备")
    parser.add_argument("--condition_model", type=str, default="distilbert-base-uncased",
                       help="条件编码器模型")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--num_samples", type=int, default=1000, help="生成的训练样本数")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print("=== 符号回归EditFlow训练 ===")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"模型维度: {args.hidden_dim}")
    print(f"Transformer层数: {args.num_layers}")

    # 创建数据生成器
    print("\n1. 创建数据生成器...")
    data_generator = SymbolicRegressionDataGenerator(
        input_dimension=args.input_dim,
        max_expression_depth=args.max_depth,
        n_points=args.n_points,
        corruption_prob=args.corruption_prob,
        random_seed=args.seed
    )

    # 生成训练数据
    print(f"2. 生成 {args.num_samples} 个训练样本...")
    train_data = data_generator.generate_batch(args.num_samples)
    print(f"   实际生成 {len(train_data)} 个有效样本")

    # 创建数据加载器
    print("3. 创建数据加载器...")
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # 创建训练器
    print("4. 创建训练器...")
    trainer = EditFlowTrainer(
        vocab_size=data_generator.vocab.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        device=args.device,
        condition_model_name=args.condition_model
    )

    # 开始训练
    print(f"5. 开始训练 ({len(dataloader)} 批次/epoch)...")
    trainer.train(dataloader, args.num_epochs, args.save_dir)

    print(f"\n训练完成！模型已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()
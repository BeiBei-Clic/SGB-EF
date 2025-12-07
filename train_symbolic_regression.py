"""
符号回归EditFlow训练主脚本
"""

import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import random
import os
import json
import numpy as np
from typing import List, Dict, Any

from src.symbolic.data_generator import generate_samples




def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_model(samples, num_epochs=10):
    """训练简单神经网络"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(1, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print(f"开始训练 {len(samples)} 个样本")

    for epoch in range(num_epochs):
        total_loss = 0
        for sample in samples:
            x = torch.tensor(sample['x'], dtype=torch.float32).unsqueeze(1).to(device)
            y = torch.tensor(sample['y'], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(samples):.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="训练符号回归模型")

    parser.add_argument("--num_samples", type=int, default=50, help="训练样本数")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--max_depth", type=int, default=4, help="表达式最大深度")
    parser.add_argument("--n_points", type=int, default=100, help="数据点数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")

    args = parser.parse_args()

    set_seed(args.seed)

    print("=== 符号回归训练 ===")
    print(f"样本数: {args.num_samples}, 轮数: {args.num_epochs}")

    # 生成数据
    print("\n1. 生成数据...")
    samples = generate_samples(args.num_samples, max_dim=5, n_points=args.n_points, max_depth=args.max_depth)
    print(f"生成了 {len(samples)} 个样本")

    # 训练模型
    print("\n2. 开始训练...")
    model = train_model(samples, args.num_epochs)

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "symbolic_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    # 显示示例
    print("\n=== 数据示例 ===")
    for i, sample in enumerate(samples[:3]):
        print(f"\n样本 {i+1}:")
        print(f"  目标表达式: {sample['exp_gt']}")
        print(f"  当前表达式: {sample['exp_cur1']}")


if __name__ == "__main__":
    main()
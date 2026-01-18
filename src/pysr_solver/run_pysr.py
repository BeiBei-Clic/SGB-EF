import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 添加当前目录到 path，支持直接运行
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_equation_data
from pysr_trainer import train_pysr, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="使用 PySR 求解 Feynman 符号回归问题")
    parser.add_argument("--equation", type=str, default="I.10.7", help="方程名称（如 I.10.7）")
    parser.add_argument("--n_samples", type=int, default=1000, help="采样数量")
    parser.add_argument("--niterations", type=int, default=100, help="PySR 迭代次数")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    print("PySR 符号回归 - Feynman 方程求解")

    # 加载数据
    print(f"\n加载数据: {args.equation}")
    X, y, meta = load_equation_data(args.equation, n_samples=args.n_samples)

    print(f"  公式: {meta['formula']}")
    print(f"  变量: {meta['var_names']}")
    print(f"  数据量: {len(X)}")
    print(f"  特征数: {X.shape[1]}")

    # 训练模型
    print(f"\n开始训练 (niterations={args.niterations})...")
    model = train_pysr(X, y, niterations=args.niterations)

    # 评估
    metrics = evaluate_model(model, X, y)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存 JSON 结果
        result = {
            "equation": args.equation,
            "formula": meta["formula"],
            "var_names": meta["var_names"],
            "best_equation": metrics["best_equation"],
            "all_equations": metrics["all_equations"],
            "metrics": {
                "r2": float(metrics["r2"]),
                "mse": float(metrics["mse"]),
                "rmse": float(metrics["rmse"]),
                "best_loss": float(metrics["best_loss"]),
            },
        }

        with open(output_path / f"{args.equation}_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存: {output_path / args.equation}_result.json")


if __name__ == "__main__":
    main()

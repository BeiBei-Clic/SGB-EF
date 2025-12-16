"""
EditFlow连续流符号回归预训练主脚本
实现基于多步连续流匹配的编辑流模型训练
使用 Hugging Face Accelerate 进行分布式训练加速
"""

import argparse
import os
import random
import numpy as np
import torch

# 设置环境变量来抑制transformers的警告输出 - 这是最有效的方法
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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
    parser.add_argument("--num_samples", type=int, default=100000000, help="训练样本数")
    parser.add_argument("--max_dim", type=int, default=10, help="最大输入维度")
    parser.add_argument("--n_points", type=int, default=100, help="数据点数量")
    parser.add_argument("--max_depth", type=int, default=4, help="表达式最大深度")
    parser.add_argument("--max_expr_length", type=int, default=12, help="表达式最大token长度")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例 (0.0-1.0)")
    parser.add_argument("--eval_every", type=int, default=5, help="每多少轮评估一次测试集")
    parser.add_argument("--read_batch_size", type=int, default=50000, help="数据读取批次大小，避免一次性加载所有数据到内存")
    parser.add_argument("--debug", action="store_true", default=False, help="是否输出调试信息")

    # 模型参数
    parser.add_argument("--base_model_name", type=str, default="google-bert/bert-base-uncased", help="基础模型名称")
    parser.add_argument("--condition_model_name", type=str, default="nomic-ai/nomic-embed-text-v1.5", help="条件嵌入模型名称")
    parser.add_argument("--cache_dir", type=str, default="models/huggingface_cache", help="模型缓存目录")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小 (每个GPU)")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--save_every", type=int, default=10, help="每多少轮保存一次")

    # 多GPU参数 - 现在由 Accelerate 自动管理
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--use_fp16", type=bool, default=True, help="是否使用FP16混合精度训练")
    parser.add_argument("--log_with", type=str, default=None, help="日志记录方式 (如 wandb, tensorboard)")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建EditFlow管理器并开始训练
    manager = EditFlowManager(args)

    try:
        model, condition_encoder = manager.train()

        # 只有在主进程才打印完成信息
        if hasattr(manager.accelerator, 'is_local_main_process') and manager.accelerator.is_local_main_process:
            print("\nEditFlow训练完成!")
    finally:
        # 确保清理分布式资源
        try:
            if hasattr(manager, 'accelerator'):
                manager.accelerator.free_memory()
                if hasattr(manager.accelerator, 'is_local_main_process') and manager.accelerator.is_local_main_process:
                    print("✓ 分布式资源已清理")
        except Exception as e:
            if hasattr(manager, 'accelerator') and hasattr(manager.accelerator, 'is_local_main_process') and manager.accelerator.is_local_main_process:
                print(f"⚠️ 资源清理时出现警告: {e}")


if __name__ == "__main__":
    main()
"""
EditFlow连续流符号回归预训练主脚本
实现基于多步连续流匹配的编辑流模型训练
使用 Hugging Face Accelerate 进行分布式训练加速
"""

import argparse
import os
import numpy as np
import torch

# 设置环境变量来抑制transformers的警告输出 - 这是最有效的方法
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 设置NCCL超时时间为无穷大，避免数据生成时的等待超时
os.environ["NCCL_TIMEOUT"] = "31536000"  # 1年（秒）
os.environ["NCCL_BLOCKING_WAIT"] = "1"   # 启用阻塞等待模式

from src.training.editflow_manager import EditFlowManager


def main():
    # 设置CUDA调试模式
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description="训练EditFlow符号回归模型")

    # 数据参数
    parser.add_argument("--num_samples", type=int, default=100000000, help="训练样本数")
    parser.add_argument("--max_dim", type=int, default=3, help="最大输入维度")
    parser.add_argument("--n_points", type=int, default=100, help="数据点数量")
    parser.add_argument("--max_depth", type=int, default=6, help="表达式最大深度")
    parser.add_argument("--max_expr_length", type=int, default=12, help="表达式最大token长度")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例 (0.0-1.0)")
    parser.add_argument("--eval_every", type=int, default=5, help="每多少轮评估一次测试集")

    # 模型参数
    parser.add_argument("--base_model_name", type=str, default="google-bert/bert-base-uncased", help="基础模型名称")
    parser.add_argument("--condition_model_name", type=str, default="settransformer", help="条件嵌入模型名称 (现在使用SetTransformer架构)")
    parser.add_argument("--cache_dir", type=str, default="models/huggingface_cache", help="模型缓存目录")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小 (每个GPU)")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--save_every", type=int, default=5, help="每多少轮保存一次")

    # 多GPU参数 - 现在由 Accelerate 自动管理
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--use_fp16", type=bool, default=True, help="是否使用FP16混合精度训练")
    parser.add_argument("--log_with", type=str, default=None, help="日志记录方式 (如 wandb, tensorboard)")

    # 多时间步采样参数
    parser.add_argument("--num_timesteps", type=int, default=1, help="每个样本的时间步采样数量")

    # SetTransformer条件编码器参数
    parser.add_argument("--condition_max_input_dim", type=int, default=6, help="SetTransformer支持的最大输入维度")
    parser.add_argument("--condition_dim_hidden", type=int, default=768, help="SetTransformer隐藏层维度（应匹配BERT的hidden_size）")
    parser.add_argument("--condition_num_heads", type=int, default=8, help="SetTransformer注意力头数 ")
    parser.add_argument("--condition_num_inds", type=int, default=32, help="SetTransformer诱导点数")
    parser.add_argument("--condition_num_layers", type=int, default=3, help="SetTransformer层数 ")
    parser.add_argument("--condition_num_seeds", type=int, default=32, help="SetTransformer种子数（输出序列长度）")
    parser.add_argument("--condition_dim_output", type=int, default=128, help="SetTransformer输出维度（已弃用，保留以兼容）")
    parser.add_argument("--condition_input_normalization", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="是否对输入进行标准化")

    # 正弦频率映射参数 (Sinusoidal Encoding)
    parser.add_argument("--condition_use_sinusoidal_encoding", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="是否使用正弦编码将数值映射到频率域")
    parser.add_argument("--condition_sinusoidal_dim", type=int, default=64, help="正弦编码维度（每个数值编码后的特征数）")
    parser.add_argument("--condition_sinusoidal_max_freq", type=float, default=10000.0, help="正弦编码最大频率（控制频率范围）")

    # 保持向后兼容性，但现在不再使用的参数
    parser.add_argument("--condition_max_length", type=int, default=1024, help="条件嵌入器的最大token长度")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建EditFlow管理器并开始训练
    manager = EditFlowManager(args)

    manager.train()

    # 只有在主进程才打印完成信息
    if hasattr(manager.accelerator, 'is_local_main_process') and manager.accelerator.is_local_main_process:
        print("\nEditFlow训练完成!")


if __name__ == "__main__":
    main()
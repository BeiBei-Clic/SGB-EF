"""
EditFlow符号回归训练主脚本 (v2.0 - 迭代优化架构)

架构更新说明：
- 从"连续流匹配"转变为"离散编辑预测"
- 移除中间状态插值，直接学习从z0到z1的编辑操作
- 条件编码器使用目标值y_target（北极星模式）而非残差
- 训练时固定t=0，不再需要多时间步采样
- 使用 Hugging Face Accelerate 进行分布式训练加速
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
    parser.add_argument("--max_expr_length", type=int, default=24, help="表达式最大token长度")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例 (0.0-1.0)")
    parser.add_argument("--eval_every", type=int, default=5, help="每多少轮评估一次测试集")

    # 模型参数
    parser.add_argument("--base_model_name", type=str, default="google-bert/bert-base-uncased", help="基础模型名称（已弃用，保留兼容性）")
    parser.add_argument("--condition_model_name", type=str, default="settransformer", help="条件嵌入模型名称 (使用SetTransformer编码目标值y_target)")
    parser.add_argument("--cache_dir", type=str, default="models/huggingface_cache", help="模型缓存目录")

    # LLaMA模型架构参数
    parser.add_argument("--hidden_dim", type=int, default=512, help="LLaMA隐藏层维度")
    parser.add_argument("--n_layers", type=int, default=8, help="LLaMA Transformer层数")
    parser.add_argument("--n_heads", type=int, default=16, help="LLaMA注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比率")
    parser.add_argument("--use_condition_injection", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="是否使用交叉注意力条件注入")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小 (每个GPU)")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--save_every", type=int, default=5, help="每多少轮保存一次")

    # 多GPU参数 - 由 Accelerate 自动管理
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--use_fp16", type=bool, default=True, help="是否使用FP16混合精度训练")
    parser.add_argument("--log_with", type=str, default=None, help="日志记录方式 (如 wandb, tensorboard)")
    parser.add_argument("--debug", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="是否启用调试模式（记录详细的训练日志）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载进程数（DataLoader的num_workers参数）")

    # 多时间步采样参数（已弃用 - v2.0架构固定t=0）
    parser.add_argument("--num_timesteps", type=int, default=1, help="每个样本的时间步采样数量（已弃用：新架构固定t=0，无需多时间步）")

    # 对齐方法参数
    parser.add_argument("--alignment_method", type=str, default='randomized',
                       choices=['levenshtein', 'randomized'],
                       help="数据对齐方法: 'levenshtein' (确定性对齐) 或 'randomized' (随机化对齐，来自Edit Flows论文)")

    # SetTransformer条件编码器参数（编码目标值y_target）
    parser.add_argument("--condition_max_input_dim", type=int, default=3, help="SetTransformer支持的最大输入维度")
    parser.add_argument("--condition_dim_hidden", type=int, default=768, help="SetTransformer隐藏层维度")
    parser.add_argument("--condition_num_heads", type=int, default=8, help="SetTransformer注意力头数")
    parser.add_argument("--condition_num_inds", type=int, default=32, help="SetTransformer诱导点数")
    parser.add_argument("--condition_num_layers", type=int, default=3, help="SetTransformer层数")
    parser.add_argument("--condition_num_seeds", type=int, default=32, help="SetTransformer种子数（输出序列长度，特征向量数量）")
    parser.add_argument("--condition_dim_output", type=int, default=128, help="SetTransformer输出维度（已弃用：现在输出序列而非单个向量）")
    parser.add_argument("--condition_input_normalization", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="是否对输入进行标准化")

    # 保持向后兼容性，但现在不再使用的参数
    parser.add_argument("--condition_max_length", type=int, default=1024, help="条件嵌入器的最大token长度")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建EditFlow管理器并开始训练
    manager = EditFlowManager(args)

    try:
        manager.train()
    finally:
        # 确保所有进程同步退出，避免资源泄漏警告
        if hasattr(manager, 'accelerator'):
            manager.accelerator.wait_for_everyone()

    # 只有在主进程才打印完成信息
    if hasattr(manager.accelerator, 'is_local_main_process') and manager.accelerator.is_local_main_process:
        print("\n" + "="*60)
        print("EditFlow训练完成! (架构v2.0 - 迭代优化模式)")
        print("="*60)


if __name__ == "__main__":
    main()
"""
极简样本过拟合测试
测试1个样本：add(x0) -> add(x0, constant)
如果模型能过拟合（loss->0），说明代码逻辑正确
"""

import torch
import sys
sys.path.append('/home/xyh/SGB-EF')

from src.training.editflow_manager import EditFlowManager
from argparse import Namespace

# 创建极简配置
args = Namespace(
    # 数据参数
    num_samples=1,
    max_dim=1,
    n_points=3,
    max_depth=3,
    max_expr_length=10,
    test_split=0.0,
    eval_every=5,
    
    # 模型参数
    base_model_name="google-bert/bert-base-uncased",
    condition_model_name="settransformer",
    cache_dir="models/huggingface_cache",
    
    # LLaMA模型参数（小模型加速测试）
    hidden_dim=128,
    n_layers=2,
    n_heads=4,
    dropout=0.0,
    use_condition_injection=True,
    
    # 训练参数
    batch_size=1,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=0.0,
    seed=42,
    save_dir="checkpoints_minimal",
    save_every=10,
    
    # 分布式参数
    gradient_accumulation_steps=1,
    use_fp16=False,
    log_with=None,
    debug=False,
    num_workers=0,
    
    # 其他参数
    action_thresholds=None,
    num_timesteps=1,
    alignment_method='randomized',
    
    # SetTransformer参数
    condition_max_input_dim=1,
    condition_dim_hidden=256,
    condition_num_heads=4,
    condition_num_inds=16,
    condition_num_layers=2,
    condition_num_seeds=16,
    condition_dim_output=128,
    condition_input_normalization=False,
    condition_max_length=1024,
)

# 使用Accelerate的单进程模式
from accelerate import Accelerator

# 模拟accelerate launch环境
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

# 创建manager并训练
print("="*60)
print("极简样本过拟合测试")
print("="*60)
print("样本: add(x0) -> add(x0, constant)")
print("期望: loss应该能降到接近0")
print("="*60)

manager = EditFlowManager(args)

try:
    manager.train()
    print("\n✅ 训练完成！")
except Exception as e:
    print(f"\n❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()

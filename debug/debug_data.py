#!/usr/bin/env python3

import sys
sys.path.append('/home/xyh/SGB-EF')

from src.symbolic.data_generator import generate_triplet_samples
from src.training.editflow_trainer import TripletDataset
import torch

# 生成小批量数据
print("生成数据...")
samples = generate_triplet_samples(4, max_dim=2, n_points=5, max_depth=2)

# 按维度分组
dim_groups = {}
for sample in samples:
    dim = sample['input_dimension']
    if dim not in dim_groups:
        dim_groups[dim] = []
    dim_groups[dim].append(sample)

print(f"维度分组: {list(dim_groups.keys())}")

for dim, dim_samples in dim_groups.items():
    print(f"\n=== 维度 {dim} ===")
    dataset = TripletDataset(dim_samples, vocab_size=1000)

    print(f"样本数量: {len(dataset)}")

    # 检查每个样本
    for i in range(len(dataset)):
        try:
            sample_data = dataset[i]
            print(f"样本 {i}:")
            print(f"  x_values: {sample_data['x_values'].shape}")
            print(f"  residuals: {sample_data['residuals'].shape}")
            print(f"  curr_token_ids: {sample_data['curr_token_ids'].shape}")
            print(f"  target_token_ids: {sample_data['target_token_ids'].shape}")
            print(f"  alignment type: {type(sample_data['alignment'])}")
            if isinstance(sample_data['alignment'], dict):
                print(f"    alignment keys: {sample_data['alignment'].keys()}")
                if 'alignment' in sample_data['alignment']:
                    print(f"    alignment length: {len(sample_data['alignment']['alignment'])}")
        except Exception as e:
            print(f"样本 {i} 出错: {e}")

    # 尝试创建DataLoader
    try:
        from src.training.editflow_trainer import custom_collate_fn
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
        )

        print("DataLoader创建成功，尝试读取第一个batch...")
        for batch in dataloader:
            print(f"Batch shapes:")
            print(f"  x_values: {batch['x_values'].shape}")
            print(f"  residuals: {batch['residuals'].shape}")
            print(f"  curr_token_ids: {batch['curr_token_ids'].shape}")
            print(f"  target_token_ids: {batch['target_token_ids'].shape}")
            break

    except Exception as e:
        print(f"DataLoader出错: {e}")
        import traceback
        traceback.print_exc()
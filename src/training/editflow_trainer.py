"""
EditFlow训练器 - 实现基于残差条件的编辑流模型训练
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

from ..symbolic.data_generator import generate_triplet_samples
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig
from ..utils.special_tokens import SpecialTokensManager


class TripletDataset(torch.utils.data.Dataset):
    """三元组数据集 (E_curr, E_target, r, z)"""

    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id if tokenizer.bos_token is not None else tokenizer.cls_token_id

        # 使用统一的特殊token管理器
        self.max_dim = 10
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=self.max_dim)

    def __len__(self):
        return len(self.samples)

    def _tokenize_expression(self, tree_str: str) -> List[int]:
        """将表达式树字符串转换为token序列，使用统一的特殊token管理器"""
        return self.special_tokens_manager.tokenize_expression(tree_str)

    def validate_data_sample(self, sample: dict, idx: int) -> dict:
        """验证数据样本的合理性"""
        print(f"\n[数据验证] 样本 {idx}:")

        # 检查x_values
        x_values = torch.FloatTensor(sample['x_values'])
        print(f"  x_values: shape={x_values.shape}, range=[{x_values.min().item():.6f}, {x_values.max().item():.6f}]")
        if torch.isnan(x_values).any():
            print(f"  ❌ x_values包含NaN!")
        if torch.isinf(x_values).any():
            print(f"  ❌ x_values包含Inf!")

        # 检查residuals
        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)
        print(f"  residuals: shape={residuals.shape}, range=[{residuals.min().item():.6f}, {residuals.max().item():.6f}]")
        if torch.isnan(residuals).any():
            print(f"  ❌ residuals包含NaN!")
        if torch.isinf(residuals).any():
            print(f"  ❌ residuals包含Inf!")

        # 检查表达式字符串
        print(f"  tree_cur1: {sample['tree_cur1'][:100]}...")
        print(f"  tree_gt: {sample['tree_gt'][:100]}...")

        # 检查对齐数据
        alignment = sample['alignment_vector']['alignment']
        print(f"  alignment: 长度={len(alignment)}")
        if alignment:
            print(f"  前3个操作: {alignment[:3]}")
            # 统计操作类型
            ops = [op for op, _, _ in alignment]
            op_counts = {op: ops.count(op) for op in set(ops)}
            print(f"  操作统计: {op_counts}")

        return sample

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 数据验证（只在前几个样本执行详细验证）
        # 注意：debug_samples 是trainer实例变量，这里先使用固定值
        if idx < 3:  # 默认只验证前3个样本
            sample = self.validate_data_sample(sample, idx)

        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)

        curr_tokens = self._tokenize_expression(sample['tree_cur1'])
        target_tokens = self._tokenize_expression(sample['tree_gt'])

        max_len = 64
        def pad_sequence(tokens):
            tokens = [self.bos_token] + tokens[:max_len-1]
            tokens.extend([self.pad_token] * (max_len - len(tokens)))
            return torch.LongTensor(tokens)

        curr_token_ids = pad_sequence(curr_tokens)
        target_token_ids = pad_sequence(target_tokens)

        # 验证token ID范围
        if idx < 3:  # 只在前几个样本验证
            print(f"  Token ID范围: curr=[{curr_token_ids.min().item()}, {curr_token_ids.max().item()}], target=[{target_token_ids.min().item()}, {target_token_ids.max().item()}]")
            print(f"  词汇表大小: {self.vocab_size}")
            if curr_token_ids.max() >= self.vocab_size:
                print(f"  ❌ 当前token ID超出词汇表范围!")
            if target_token_ids.max() >= self.vocab_size:
                print(f"  ❌ 目标token ID超出词汇表范围!")

        result = {
            'x_values': x_values,
            'residuals': residuals,
            'curr_token_ids': curr_token_ids,
            'target_token_ids': target_token_ids,
            'alignment': sample['alignment_vector']['alignment']
        }

        if idx < 3:
            print(f"[数据验证] 样本 {idx} 完成\n")

        return result


class EditFlowLoss:
    """EditFlow训练损失函数"""

    def __init__(self, scheduler_type='cubic', tokenizer=None):
        self.scheduler_type = scheduler_type
        self.tokenizer = tokenizer
        # 使用统一的特殊token管理器
        self.special_tokens_manager = SpecialTokensManager(tokenizer) if tokenizer else None
        self.func_map = self.special_tokens_manager.get_function_token_map() if self.special_tokens_manager else {}

    def print_function_mapping(self):
        """打印函数映射信息，用于调试"""
        if self.special_tokens_manager:
            self.special_tokens_manager.print_function_mapping()
        else:
            print("警告: 没有可用的特殊token管理器")

    def scheduler(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * t**2 - 2 * t**3 if self.scheduler_type == 'cubic' else t

    def scheduler_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 6 * t - 6 * t**2 if self.scheduler_type == 'cubic' else torch.ones_like(t)

    def alignment_to_target_mask(self, alignment: List[Tuple], seq_len: int, vocab_size: int) -> torch.Tensor:
        target_mask = torch.zeros(seq_len, 3 + 2 * vocab_size)

        # 调试对齐数据
        print(f"    alignment_to_target_mask: seq_len={seq_len}, vocab_size={vocab_size}")
        print(f"    func_map keys: {list(self.func_map.keys()) if self.func_map else 'None'}")

        invalid_operations = 0
        valid_operations = 0

        for i, (op, src, tgt) in enumerate(alignment[:seq_len]):
            if i >= seq_len:
                break

            # 调试每个操作
            if i < 3:  # 只打印前3个操作的详细信息
                print(f"    操作[{i}]: {op}, src={src}, tgt={tgt}")

            try:
                if op == 'keep':
                    target_mask[i, 1] = 0.1
                    valid_operations += 1
                elif op == 'insert':
                    target_mask[i, 0] = 1.0
                    if tgt in self.func_map:
                        func_idx = self.func_map[tgt]
                        if 0 <= func_idx < vocab_size:
                            target_mask[i, 3 + func_idx] = 1.0
                            valid_operations += 1
                        else:
                            print(f"    ⚠️  插入操作索引越界: {tgt} -> {func_idx}, vocab_size={vocab_size}")
                            invalid_operations += 1
                    else:
                        print(f"    ⚠️  插入操作目标函数未在映射中: {tgt}")
                        invalid_operations += 1
                elif op == 'delete':
                    target_mask[i, 2] = 1.0
                    valid_operations += 1
                elif op == 'substitute':
                    target_mask[i, 1] = 1.0
                    if tgt in self.func_map:
                        func_idx = self.func_map[tgt]
                        if 0 <= func_idx < vocab_size:
                            target_mask[i, 3 + vocab_size + func_idx] = 1.0
                            valid_operations += 1
                        else:
                            print(f"    ⚠️  替换操作索引越界: {tgt} -> {func_idx}, vocab_size={vocab_size}")
                            invalid_operations += 1
                    else:
                        print(f"    ⚠️  替换操作目标函数未在映射中: {tgt}")
                        invalid_operations += 1
                else:
                    print(f"    ⚠️  未知操作类型: {op}")
                    invalid_operations += 1

            except Exception as e:
                print(f"    ❌ 处理操作[{i}]时出错: {e}")
                invalid_operations += 1

        print(f"    有效操作: {valid_operations}, 无效操作: {invalid_operations}")

        # 检查target_mask是否包含NaN或Inf
        if torch.isnan(target_mask).any():
            print(f"    ❌ target_mask包含NaN值!")
        if torch.isinf(target_mask).any():
            print(f"    ❌ target_mask包含Inf值!")

        return target_mask

    def debug_tensor_stats(self, tensor: torch.Tensor, name: str):
        """打印张量的统计信息用于调试"""
        if torch.isnan(tensor).any():
            print(f"❌ {name}: 检测到NaN值!")
            print(f"   形状: {tensor.shape}")
            print(f"   NaN数量: {torch.isnan(tensor).sum()}")
        else:
            print(f"✓ {name}: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")

        if torch.isinf(tensor).any():
            print(f"❌ {name}: 检测到Inf值!")
            print(f"   Inf数量: {torch.isinf(tensor).sum()}")

    def __call__(self, pred_rates: torch.Tensor, pred_ins_probs: torch.Tensor,
                 pred_sub_probs: torch.Tensor, alignment: List[Tuple],
                 t: torch.Tensor, vocab_size: int, debug=False) -> torch.Tensor:
        batch_size, seq_len, _ = pred_rates.shape

        # 调试输入（可选）
        if debug:
            print(f"\n=== 损失计算调试信息 ===")
            print(f"批次大小: {batch_size}, 序列长度: {seq_len}, 词汇表大小: {vocab_size}")
            self.debug_tensor_stats(pred_rates, "pred_rates")
            self.debug_tensor_stats(pred_ins_probs, "pred_ins_probs")
            self.debug_tensor_stats(pred_sub_probs, "pred_sub_probs")
            self.debug_tensor_stats(t, "time_steps")

        # 检查预测速率是否包含异常值
        if pred_rates.max() > 100:
            if debug:
                print(f"⚠️  警告: pred_rates包含极大值: {pred_rates.max().item()}")
        if pred_rates.min() < 0:
            if debug:
                print(f"⚠️  警告: pred_rates包含负值: {pred_rates.min().item()}")

        insert_rates = pred_rates[:, :, 0:1]
        substitute_rates = pred_rates[:, :, 1:2]
        delete_rates = pred_rates[:, :, 2:3]

        if debug:
            self.debug_tensor_stats(insert_rates, "insert_rates")
            self.debug_tensor_stats(substitute_rates, "substitute_rates")
            self.debug_tensor_stats(delete_rates, "delete_rates")

        ins_probs = insert_rates * pred_ins_probs
        sub_probs = substitute_rates * pred_sub_probs
        if debug:
            self.debug_tensor_stats(ins_probs, "ins_probs")
            self.debug_tensor_stats(sub_probs, "sub_probs")

        u_cat = torch.cat([insert_rates, substitute_rates, delete_rates, ins_probs, sub_probs], dim=-1)
        if debug:
            print(f"u_cat拼接前形状: insert_rates{insert_rates.shape}, substitute_rates{substitute_rates.shape}, delete_rates{delete_rates.shape}, ins_probs{ins_probs.shape}, sub_probs{sub_probs.shape}")

        # 更严格的数值稳定性处理
        u_cat = torch.clamp(u_cat, min=1e-10, max=1e10)
        if debug:
            self.debug_tensor_stats(u_cat, "u_cat(clamped)")

        # 调试对齐数据处理
        if debug:
            print(f"处理对齐数据，样本数量: {len(alignment)}")
            if alignment:
                print(f"第一个对齐样本: {alignment[0][:3]}...")  # 只显示前3个操作

        target_masks = torch.stack([
            self.alignment_to_target_mask(align, seq_len, vocab_size)
            for align in alignment
        ]).to(pred_rates.device)
        if debug:
            self.debug_tensor_stats(target_masks, "target_masks")

        u_total = pred_rates.sum(dim=(1, 2))
        if debug:
            self.debug_tensor_stats(u_total, "u_total")

        sched_t = self.scheduler(t)
        sched_dt = self.scheduler_derivative(t)
        if debug:
            self.debug_tensor_stats(sched_t, "sched_t")
            self.debug_tensor_stats(sched_dt, "sched_dt")

        # 更安全的调度器系数计算
        denominator = (1 - sched_t + 1e-8)
        sched_coeff_raw = sched_dt / denominator
        if debug:
            self.debug_tensor_stats(denominator, "denominator(1-sched_t+1e-8)")
            self.debug_tensor_stats(sched_coeff_raw, "sched_coeff_raw")

        sched_coeff = torch.clamp(sched_coeff_raw.squeeze(-1), min=-10, max=10)
        if debug:
            self.debug_tensor_stats(sched_coeff, "sched_coeff(clamped)")

        # 更安全的交叉熵计算
        log_u_cat = torch.log(torch.clamp(u_cat, min=1e-12, max=1e12))
        if debug:
            self.debug_tensor_stats(log_u_cat, "log_u_cat")

        cross_entropy = (log_u_cat * target_masks).sum(dim=(1, 2))
        if debug:
            self.debug_tensor_stats(cross_entropy, "cross_entropy")

        loss = u_total - cross_entropy * sched_coeff
        if debug:
            self.debug_tensor_stats(loss, "loss(before_mean)")

        final_loss = loss.mean()
        if debug:
            self.debug_tensor_stats(final_loss, "final_loss")
            print(f"=== 损失计算调试结束 ===\n")

        return final_loss


def sample_conditional_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, scheduler) -> torch.Tensor:
    t = t.view(-1, 1).expand(-1, x0.size(1))
    return ((1 - scheduler(t)) * x0 + scheduler(t) * x1).long()


def custom_collate_fn(batch):
    return {
        'x_values': torch.stack([item['x_values'] for item in batch]),
        'residuals': torch.stack([item['residuals'] for item in batch]),
        'curr_token_ids': torch.stack([item['curr_token_ids'] for item in batch]),
        'target_token_ids': torch.stack([item['target_token_ids'] for item in batch]),
        'alignment': [item['alignment'] for item in batch]
    }


class EditFlowTrainer:
    """EditFlow模型训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(args.seed)
        # 调试控制：只在前N个批次/epoch输出详细调试信息
        self.debug_batches = 2
        self.debug_samples = 3

    def set_seed(self, seed: int):
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def prepare_data(self, tokenizer):
        print("生成三元组训练数据...")
        samples = generate_triplet_samples(
            num_samples=self.args.num_samples,
            max_dim=self.args.max_dim,
            n_points=self.args.n_points,
            max_depth=self.args.max_depth
        )

        dimension_groups = {}
        for sample in samples:
            dim = sample['input_dimension']
            dimension_groups.setdefault(dim, []).append(sample)

        dataloaders = {}
        datasets = {}

        for dim, dim_samples in dimension_groups.items():
            dataset = TripletDataset(dim_samples, tokenizer)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=True,
                num_workers=0, collate_fn=custom_collate_fn
            )
            dataloaders[dim] = dataloader
            datasets[dim] = dataset

        return dataloaders, datasets, dimension_groups

    def setup_models(self):
        print("初始化tokenizer和模型...")

        # 首先初始化tokenizer，获取预训练模型的词汇表
        model_name = getattr(self.args, 'base_model_name', "openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 确保tokenizer有pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

        print("初始化条件编码器...")
        condition_encoder = ConditionEncoder().to(self.device)

        print("初始化EditFlow模型...")
        config = EditFlowConfig(
            max_seq_len=64,
            condition_dim=condition_encoder.output_dim,
            use_condition_injection=True,
            base_model_name=model_name
        )
        # vocab_size 将从tokenizer动态获取
        config.vocab_size = tokenizer.vocab_size
        model = EditFlowTransformer(config).to(self.device)
        print(f"EditFlow模型参数数量: {sum(p.numel() for p in model.parameters())}")

        criterion = EditFlowLoss(scheduler_type='cubic', tokenizer=tokenizer)
        criterion.print_function_mapping()  # 打印函数映射用于调试
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        return model, condition_encoder, criterion, optimizer, tokenizer

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}")

        for batch_idx, batch in enumerate(progress_bar):
            # 批次数据验证（只在前几个批次验证）
            if batch_idx < self.debug_batches:
                print(f"\n[批次验证] 批次 {batch_idx}:")
                print(f"  批次大小: {batch['x_values'].shape[0]}")

            x_values = batch['x_values'].to(self.device)
            residuals = batch['residuals'].to(self.device)
            curr_token_ids = batch['curr_token_ids'].to(self.device)
            target_token_ids = batch['target_token_ids'].to(self.device)
            alignment = batch['alignment']

            # 检查批次数据的异常值
            if batch_idx < self.debug_batches:
                # 检查数值数据
                for name, tensor in [("x_values", x_values), ("residuals", residuals)]:
                    if torch.isnan(tensor).any():
                        print(f"  ❌ {name}包含NaN!")
                    if torch.isinf(tensor).any():
                        print(f"  ❌ {name}包含Inf!")
                    print(f"  {name}统计: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")

                # 检查token数据
                for name, tensor in [("curr_token_ids", curr_token_ids), ("target_token_ids", target_token_ids)]:
                    if tensor.min() < 0:
                        print(f"  ❌ {name}包含负值token!")
                    if tensor.max() >= model.config.vocab_size:
                        print(f"  ❌ {name}超出词汇表范围! max={tensor.max().item()}, vocab_size={model.config.vocab_size}")

                print(f"[批次验证] 批次 {batch_idx} 完成\n")

            t = torch.rand(curr_token_ids.size(0), 1, device=self.device)

            # 检查条件编码器输出
            try:
                condition = condition_encoder(x_values, residuals)
                if batch_idx < self.debug_batches:
                    print(f"条件编码器输出: shape={condition.shape}, range=[{condition.min().item():.6f}, {condition.max().item():.6f}]")
                    if torch.isnan(condition).any():
                        print(f"  ❌ 条件编码器输出包含NaN!")
                    if torch.isinf(condition).any():
                        print(f"  ❌ 条件编码器输出包含Inf!")
            except Exception as e:
                print(f"❌ 条件编码器计算错误: {e}")
                continue

            import random
            x_t = curr_token_ids if random.random() < 0.5 else sample_conditional_path(
                curr_token_ids, target_token_ids, t, criterion.scheduler
            )

            attention_mask = (x_t != dataset.pad_token).float()
            pred_rates, pred_ins_probs, pred_sub_probs = model(
                input_ids=x_t, time_steps=t, condition=condition, attention_mask=attention_mask
            )

            loss = criterion(
                pred_rates=pred_rates, pred_ins_probs=pred_ins_probs, pred_sub_probs=pred_sub_probs,
                alignment=alignment, t=t, vocab_size=model.config.vocab_size, debug=(batch_idx < self.debug_batches)
            )

            optimizer.zero_grad()

            # 增强的梯度调试
            print(f"\n--- 梯度调试信息 ---")
            print(f"损失值: {loss.item():.6f}")

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"❌ 损失为NaN，跳过反向传播")
                continue

            loss.backward()

            # 详细的梯度检查
            has_nan_gradients = False
            max_grad_norm = 0.0
            total_params = 0
            nan_params = 0
            inf_params = 0

            print("检查模型梯度:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_params += 1
                    grad_norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)

                    if torch.isnan(param.grad).any():
                        print(f"  ❌ {name}: NaN梯度 detected, norm={grad_norm}")
                        nan_params += 1
                        has_nan_gradients = True
                    elif torch.isinf(param.grad).any():
                        print(f"  ❌ {name}: Inf梯度 detected, norm={grad_norm}")
                        inf_params += 1
                        has_nan_gradients = True
                    elif grad_norm > 10:
                        print(f"  ⚠️  {name}: 大梯度值, norm={grad_norm:.4f}")

            print("检查条件编码器梯度:")
            for name, param in condition_encoder.named_parameters():
                if param.grad is not None:
                    total_params += 1
                    grad_norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)

                    if torch.isnan(param.grad).any():
                        print(f"  ❌ {name}: NaN梯度 detected, norm={grad_norm}")
                        nan_params += 1
                        has_nan_gradients = True
                    elif torch.isinf(param.grad).any():
                        print(f"  ❌ {name}: Inf梯度 detected, norm={grad_norm}")
                        inf_params += 1
                        has_nan_gradients = True
                    elif grad_norm > 10:
                        print(f"  ⚠️  {name}: 大梯度值, norm={grad_norm:.4f}")

            print(f"梯度统计: 总参数={total_params}, NaN参数={nan_params}, Inf参数={inf_params}, 最大梯度范数={max_grad_norm:.4f}")

            # 如果没有NaN梯度，进行梯度裁剪和参数更新
            if not has_nan_gradients:
                # 梯度裁剪前后的范数对比
                model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                cond_grad_norm = torch.nn.utils.clip_grad_norm_(condition_encoder.parameters(), 1.0)
                print(f"梯度裁剪: model={model_grad_norm:.4f}, condition_encoder={cond_grad_norm:.4f}")

                optimizer.step()
                print("✓ 参数更新完成")
            else:
                print("❌ 跳过参数更新 due to NaN/Inf gradients")

            print("--- 梯度调试结束 ---\n")

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

        return total_loss / num_batches, num_batches

    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config):
        import os
        checkpoint_path = os.path.join(self.args.save_dir, f"editflow_epoch_{epoch+1}.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'condition_encoder_state_dict': condition_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss,
            'config': config, 'args': self.args
        }, checkpoint_path)

        return checkpoint_path

    def train(self):
        print(f"使用设备: {self.device}")

        # 首先设置模型获取tokenizer
        model, condition_encoder, criterion, optimizer, tokenizer = self.setup_models()

        # 使用tokenizer准备数据
        dataloaders, datasets, dimension_groups = self.prepare_data(tokenizer)

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"条件编码器参数数量: {sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad):,}")
        print(f"开始训练 ({self.args.num_epochs} epochs)...")

        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            total_batches = 0

            for dim, dataloader in dataloaders.items():
                print(f"\n训练维度 {dim} 的数据...")
                dataset = datasets[dim]
                dim_loss, dim_batches = self.train_epoch(
                    model, condition_encoder, criterion, optimizer,
                    dataloader, dataset, epoch, dim
                )
                total_loss += dim_loss * dim_batches
                total_batches += dim_batches
                print(f"维度 {dim} 平均损失: {dim_loss:.4f}")

            avg_loss = total_loss / total_batches
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs} 完成, 总体平均损失: {avg_loss:.4f}")

            if (epoch + 1) % self.args.save_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch, model.config
                )
                print(f"检查点已保存到: {checkpoint_path}")

        import os
        final_model_path = os.path.join(self.args.save_dir, "editflow_final.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        torch.save({
            'epoch': self.args.num_epochs,
            'model_state_dict': model.state_dict(),
            'condition_encoder_state_dict': condition_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.config,
            'args': self.args
        }, final_model_path)

        print(f"最终模型已保存到: {final_model_path}")
        return model, condition_encoder
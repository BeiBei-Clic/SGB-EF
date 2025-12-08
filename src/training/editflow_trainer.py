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
from ..utils.gpu_monitor import get_gpu_memory_info, get_gpu_memory_usage_string


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
        # 检查x_values
        x_values = torch.FloatTensor(sample['x_values'])
        if torch.isnan(x_values).any() or torch.isinf(x_values).any():
            raise ValueError(f"样本 {idx} 的 x_values 包含 NaN 或 Inf 值")

        # 检查residuals
        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)
        if torch.isnan(residuals).any() or torch.isinf(residuals).any():
            raise ValueError(f"样本 {idx} 的 residuals 包含 NaN 或 Inf 值")

        return sample

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 数据验证
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
        if curr_token_ids.max() >= self.vocab_size or target_token_ids.max() >= self.vocab_size:
            raise ValueError(f"样本 {idx} 的 token ID 超出词汇表范围")

        result = {
            'x_values': x_values,
            'residuals': residuals,
            'curr_token_ids': curr_token_ids,
            'target_token_ids': target_token_ids,
            'alignment': sample['alignment_vector']['alignment']
        }

        return result


class EditFlowLoss:
    """EditFlow训练损失函数"""

    def __init__(self, scheduler_type='cubic', tokenizer=None):
        self.scheduler_type = scheduler_type
        self.tokenizer = tokenizer
        # 使用统一的特殊token管理器
        self.special_tokens_manager = SpecialTokensManager(tokenizer) if tokenizer else None
        self.func_map = self.special_tokens_manager.get_function_token_map() if self.special_tokens_manager else {}

    def scheduler(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * t**2 - 2 * t**3 if self.scheduler_type == 'cubic' else t

    def scheduler_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 6 * t - 6 * t**2 if self.scheduler_type == 'cubic' else torch.ones_like(t)

    def alignment_to_target_mask(self, alignment: List[Tuple], seq_len: int, vocab_size: int) -> torch.Tensor:
        target_mask = torch.zeros(seq_len, 3 + 2 * vocab_size)

        for i, (op, src, tgt) in enumerate(alignment[:seq_len]):
            if i >= seq_len:
                break

            try:
                if op == 'keep':
                    target_mask[i, 1] = 0.1
                elif op == 'insert':
                    target_mask[i, 0] = 1.0
                    if tgt in self.func_map:
                        func_idx = self.func_map[tgt]
                        if 0 <= func_idx < vocab_size:
                            target_mask[i, 3 + func_idx] = 1.0
                elif op == 'delete':
                    target_mask[i, 2] = 1.0
                elif op == 'substitute':
                    target_mask[i, 1] = 1.0
                    if tgt in self.func_map:
                        func_idx = self.func_map[tgt]
                        if 0 <= func_idx < vocab_size:
                            target_mask[i, 3 + vocab_size + func_idx] = 1.0
            except Exception:
                pass  # 静默忽略错误

        return target_mask

    def debug_tensor_stats(self, tensor: torch.Tensor, name: str):
        """打印张量的统计信息用于调试"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} 包含 NaN 或 Inf 值")

    def __call__(self, pred_rates: torch.Tensor, pred_ins_probs: torch.Tensor,
                 pred_sub_probs: torch.Tensor, alignment: List[Tuple],
                 t: torch.Tensor, vocab_size: int, debug=False) -> torch.Tensor:
        batch_size, seq_len, _ = pred_rates.shape

        # 数值稳定性检查
        self.debug_tensor_stats(pred_rates, "pred_rates")
        self.debug_tensor_stats(pred_ins_probs, "pred_ins_probs")
        self.debug_tensor_stats(pred_sub_probs, "pred_sub_probs")

        insert_rates = pred_rates[:, :, 0:1]
        substitute_rates = pred_rates[:, :, 1:2]
        delete_rates = pred_rates[:, :, 2:3]

        ins_probs = insert_rates * pred_ins_probs
        sub_probs = substitute_rates * pred_sub_probs

        u_cat = torch.cat([insert_rates, substitute_rates, delete_rates, ins_probs, sub_probs], dim=-1)

        # 数值稳定性处理
        u_cat = torch.clamp(u_cat, min=1e-10, max=1e10)

        target_masks = torch.stack([
            self.alignment_to_target_mask(align, seq_len, vocab_size)
            for align in alignment
        ]).to(pred_rates.device)

        u_total = pred_rates.sum(dim=(1, 2))

        sched_t = self.scheduler(t)
        sched_dt = self.scheduler_derivative(t)

        # 调度器系数计算
        denominator = (1 - sched_t + 1e-8)
        sched_coeff_raw = sched_dt / denominator
        sched_coeff = torch.clamp(sched_coeff_raw.squeeze(-1), min=-10, max=10)

        # 交叉熵计算
        log_u_cat = torch.log(torch.clamp(u_cat, min=1e-12, max=1e12))
        cross_entropy = (log_u_cat * target_masks).sum(dim=(1, 2))

        loss = u_total - cross_entropy * sched_coeff
        final_loss = loss.mean()

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

        # 多GPU设置
        self.use_data_parallel = getattr(args, 'use_data_parallel', False) and torch.cuda.is_available()
        if self.use_data_parallel:
            self.gpu_count = torch.cuda.device_count()
            self.device_ids = list(range(self.gpu_count))
            print(f"多GPU模式: 使用 {self.gpu_count} 块GPU")
        else:
            self.gpu_count = 1
            self.device_ids = [0] if torch.cuda.is_available() else None

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

        # 使用DataParallel包装模型以支持多GPU
        if self.use_data_parallel and self.gpu_count > 1:
            print(f"使用DataParallel包装模型...")
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            condition_encoder = torch.nn.DataParallel(condition_encoder, device_ids=self.device_ids)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"EditFlow模型参数数量: {total_params:,}")

        criterion = EditFlowLoss(scheduler_type='cubic', tokenizer=tokenizer)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # 多GPU设置优化
        if self.use_data_parallel and self.gpu_count > 1:
            # 为多GPU训练优化batch size
            effective_batch_size = self.args.batch_size * self.gpu_count
            print(f"有效批次大小: {effective_batch_size} (每个GPU: {self.args.batch_size})")
            # 如果需要，可以在这里添加梯度累积逻辑

        return model, condition_encoder, criterion, optimizer, tokenizer

    def _get_model_config(self, model):
        """获取模型配置，处理DataParallel包装"""
        if hasattr(model, 'module'):
            return model.module.config
        return model.config

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}")

        # 获取模型配置
        config = self._get_model_config(model)

        # 如果使用多GPU，添加GPU负载显示
        if self.use_data_parallel and self.gpu_count > 1:
            progress_bar.set_postfix({
                'loss': '0.0000',
                'gpu_load': get_gpu_memory_usage_string(max_gpus=3)
            })

        for batch_idx, batch in enumerate(progress_bar):
            x_values = batch['x_values'].to(self.device)
            residuals = batch['residuals'].to(self.device)
            curr_token_ids = batch['curr_token_ids'].to(self.device)
            target_token_ids = batch['target_token_ids'].to(self.device)
            alignment = batch['alignment']

            t = torch.rand(curr_token_ids.size(0), 1, device=self.device)

            # 检查条件编码器输出
            try:
                condition = condition_encoder(x_values, residuals)
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
                alignment=alignment, t=t, vocab_size=config.vocab_size
            )

            # 梯度累积
            loss = loss / gradient_accumulation_steps

            optimizer.zero_grad()

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"❌ 损失为NaN，跳过反向传播")
                continue

            loss.backward()

            # 每gradient_accumulation_steps步或最后一个batch，执行优化步骤
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):

                # 梯度检查
                has_nan_gradients = False
                # 检查模型梯度
                if hasattr(model, 'module'):
                    # 如果是DataParallel，检查module的参数
                    for name, param in model.module.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_gradients = True
                            print(f"❌ 模型参数 {name} 包含 NaN/Inf 梯度")
                            break
                else:
                    for name, param in model.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_gradients = True
                            print(f"❌ 模型参数 {name} 包含 NaN/Inf 梯度")
                            break

                # 检查条件编码器梯度
                if not has_nan_gradients:
                    if hasattr(condition_encoder, 'module'):
                        for name, param in condition_encoder.module.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_gradients = True
                                print(f"❌ 条件编码器参数 {name} 包含 NaN/Inf 梯度")
                                break
                    else:
                        for name, param in condition_encoder.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_gradients = True
                                print(f"❌ 条件编码器参数 {name} 包含 NaN/Inf 梯度")
                                break

                # 如果没有NaN梯度，进行梯度裁剪和参数更新
                if not has_nan_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(condition_encoder.parameters(), 1.0)
                    optimizer.step()
                else:
                    print("❌ 跳过参数更新 due to NaN/Inf gradients")
                    optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # 显示实时进度和GPU负载信息
            postfix_dict = {
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'
            }

            # 添加多GPU负载信息
            if self.use_data_parallel and self.gpu_count > 1:
                postfix_dict['gpu_load'] = get_gpu_memory_usage_string(max_gpus=3)

            progress_bar.set_postfix(postfix_dict)

        return total_loss / num_batches, num_batches

    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config):
        import os
        checkpoint_path = os.path.join(self.args.save_dir, f"editflow_epoch_{epoch+1}.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        # 处理DataParallel模型的状态字典
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        if hasattr(condition_encoder, 'module'):
            condition_encoder_state = condition_encoder.module.state_dict()
        else:
            condition_encoder_state = condition_encoder.state_dict()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'condition_encoder_state_dict': condition_encoder_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'args': self.args,
            'use_data_parallel': self.use_data_parallel,
            'gpu_count': self.gpu_count
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
                config = self._get_model_config(model)
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch, config
                )
                print(f"检查点已保存到: {checkpoint_path}")

        import os
        final_model_path = os.path.join(self.args.save_dir, "editflow_final.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        # 处理DataParallel模型的状态字典
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        if hasattr(condition_encoder, 'module'):
            condition_encoder_state = condition_encoder.module.state_dict()
        else:
            condition_encoder_state = condition_encoder.state_dict()

        config = self._get_model_config(model)
        torch.save({
            'epoch': self.args.num_epochs,
            'model_state_dict': model_state,
            'condition_encoder_state_dict': condition_encoder_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'args': self.args,
            'use_data_parallel': self.use_data_parallel,
            'gpu_count': self.gpu_count
        }, final_model_path)

        print(f"最终模型已保存到: {final_model_path}")
        return model, condition_encoder
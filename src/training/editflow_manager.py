"""
EditFlow连续流训练器 - 实现基于连续时间流匹配的编辑流模型训练
"""

import torch
import numpy as np
import time
import argparse
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer

from ..symbolic.data_generator import generate_flow_samples
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig
from ..utils.special_tokens import SpecialTokensManager
from ..utils.gpu_monitor import get_gpu_memory_info, get_gpu_memory_usage_string
from ..utils.misc_utils import find_latest_checkpoint


class KappaScheduler:
    """时间调度器，用于控制流的插值"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler_type = scheduler_type

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """返回时间t的调度系数"""
        return 3 * t**2 - 2 * t**3 if self.scheduler_type == 'cubic' else t

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """返回时间t的调度系数导数"""
        return 6 * t - 6 * t**2 if self.scheduler_type == 'cubic' else torch.ones_like(t)


def sample_conditional_path(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, scheduler: KappaScheduler) -> torch.Tensor:
    """在给定时间t采样条件路径"""
    batch_size, seq_len, vocab_size = p0.shape
    t = t.view(-1, 1, 1) if t.dim() == 1 else t
    kappa_t = scheduler(t).expand(-1, seq_len, -1)

    pt = (1 - kappa_t) * p0 + kappa_t * p1
    pt = pt / pt.sum(dim=-1, keepdim=True)

    pt_flat = pt.view(-1, pt.size(-1))
    sampled = torch.multinomial(pt_flat, 1)
    return sampled.view(batch_size, seq_len)


def tokens_to_prob(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """将token序列转换为概率分布"""
    batch_size, seq_len = tokens.shape
    probs = torch.zeros(batch_size, seq_len, vocab_size, device=tokens.device)

    # 只处理有效的token IDs
    valid_tokens = torch.clamp(tokens, 0, vocab_size - 1)
    probs.scatter_(2, valid_tokens.unsqueeze(-1), 1.0)
    return probs


def remove_gap_tokens(z_t: torch.Tensor, vocab_size: int, pad_token_id: int, gap_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列"""
    gap_token_id = gap_token_id or vocab_size + 1
    batch_size, z_seq_len = z_t.shape
    device = z_t.device

    z_gap_mask = (z_t == gap_token_id)
    z_pad_mask = (z_t == pad_token_id)

    # 使用掩码操作移除gap tokens
    x_t_list = []
    for i in range(batch_size):
        non_gap_mask = ~z_gap_mask[i]
        x_row = z_t[i][non_gap_mask]
        x_t_list.append(x_row)

    max_len = max(len(x) for x in x_t_list)
    x_t_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    x_pad_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i, x_row in enumerate(x_t_list):
        x_t_padded[i, :len(x_row)] = x_row
        x_pad_mask_padded[i, :len(x_row)] = (x_row == pad_token_id)

    return x_t_padded, x_pad_mask_padded, z_gap_mask, z_pad_mask


def fill_gap_tokens_with_repeats(x_ut: torch.Tensor, z_gap_mask: torch.Tensor, z_pad_mask: torch.Tensor) -> torch.Tensor:
    """用重复值填充gap token位置"""
    batch_size, z_seq_len = z_gap_mask.shape
    _, x_seq_len, vocab_size = x_ut.shape

    # 计算每个位置对应的非gap位置
    non_gap_mask = ~z_gap_mask
    indices = non_gap_mask.cumsum(dim=1) - 1
    indices = indices.clamp(min=0, max=x_seq_len-1)

    # 收集对应的特征
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]
    result[z_pad_mask] = 0

    return result


class FlowDataset(torch.utils.data.Dataset):
    """连续流数据集 (z0, z1, x_values, residuals)"""

    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id or tokenizer.cls_token_id
        self.gap_token = self.vocab_size + 100
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)

    def __len__(self):
        return len(self.samples)

    def _tokenize_expression_tokens(self, tokens: List[str]) -> List[int]:
        """将token列表转换为token ID列表"""
        token_ids = []
        for token in tokens:
            if token == "<gap>":
                token_ids.append(self.gap_token)
            else:
                tree_str = token if ',' in token else token
                tokenized = self.special_tokens_manager.tokenize_expression(tree_str)
                token_ids.extend(tokenized)
        return token_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)

        z0_tokens = self._tokenize_expression_tokens(sample['z0_tokens'])
        z1_tokens = self._tokenize_expression_tokens(sample['z1_tokens'])

        max_len = 128
        def pad_z_sequence(tokens):
            tokens = [self.bos_token] + tokens[:max_len-1]
            tokens.extend([self.pad_token] * (max_len - len(tokens)))
            return torch.LongTensor(tokens)

        return {
            'x_values': x_values,
            'residuals': residuals,
            'z0_token_ids': pad_z_sequence(z0_tokens),
            'z1_token_ids': pad_z_sequence(z1_tokens),
            'gap_token': self.gap_token
        }


class ContinuousFlowLoss:
    """连续时间流匹配损失函数"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler = KappaScheduler(scheduler_type)

    def make_ut_mask_from_z(self, z_t: torch.Tensor, z_1: torch.Tensor, vocab_size: int,
                           gap_token: int, pad_token: int) -> torch.Tensor:
        batch_size, z_seq_len = z_t.shape
        n_ops = 2 * vocab_size + 1

        z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
        z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq
        z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq
        z_sub = z_neq & ~z_ins & ~z_del

        u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
        u_mask[z_ins, z_1[z_ins]] = True
        u_mask[z_sub, z_1[z_sub] + vocab_size] = True
        u_mask[:, :, -1][z_del] = True

        return u_mask

    def __call__(self, u_cat: torch.Tensor, u_mask: torch.Tensor,
                 t: torch.Tensor, vocab_size: int) -> torch.Tensor:
        u_total = u_cat.sum(dim=(1, 2))
        sched_coeff = (self.scheduler.derivative(t) / (1 - self.scheduler(t) + 1e-8)).squeeze(-1)
        sched_coeff = torch.clamp(sched_coeff, min=-10, max=10)

        log_u_cat = torch.log(torch.clamp(u_cat, min=1e-12, max=1e12))
        cross_entropy = (log_u_cat * u_mask.float()).sum(dim=(1, 2))

        loss = u_total - cross_entropy * sched_coeff
        return loss.mean()


def custom_collate_fn(batch):
    return {
        'x_values': torch.stack([item['x_values'] for item in batch]),
        'residuals': torch.stack([item['residuals'] for item in batch]),
        'z0_token_ids': torch.stack([item['z0_token_ids'] for item in batch]),
        'z1_token_ids': torch.stack([item['z1_token_ids'] for item in batch]),
        'gap_token': batch[0]['gap_token']
    }


class EditFlowManager:
    """EditFlow模型管理器 - 支持训练和推理功能"""

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

        # 时间调度器
        self.scheduler = KappaScheduler(scheduler_type='cubic')

    def set_seed(self, seed: int):
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def prepare_data(self, tokenizer):
        print("生成连续流训练数据...")
        samples = generate_flow_samples(
            num_samples=self.args.num_samples,
            max_dim=self.args.max_dim,
            n_points=self.args.n_points,
            max_depth=self.args.max_depth
        )

        # 按维度分组
        dimension_groups = {}
        for sample in samples:
            dim = sample['input_dimension']
            dimension_groups.setdefault(dim, []).append(sample)

        dataloaders = {}
        datasets = {}

        for dim, dim_samples in dimension_groups.items():
            dataset = FlowDataset(dim_samples, tokenizer)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=True,
                num_workers=0, collate_fn=custom_collate_fn
            )
            dataloaders[dim] = dataloader
            datasets[dim] = dataset

        return dataloaders, datasets, dimension_groups

    def setup_models(self, checkpoint_path=None):
        """
        初始化模型和tokenizer，支持从检查点加载

        Args:
            checkpoint_path: 检查点文件路径，如果为None则创建新模型

        Returns:
            model, condition_encoder, criterion, optimizer, tokenizer
        """
        print("初始化tokenizer和模型...")

        # 首先初始化tokenizer，获取预训练模型的词汇表
        model_name = getattr(self.args, 'base_model_name', "google-bert/bert-base-uncased")

        # 设置模型缓存目录
        cache_dir = getattr(self.args, 'cache_dir', "models/huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"正在加载tokenizer: {model_name}")
        print(f"模型缓存目录: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"✓ Tokenizer加载完成")

        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

        print("初始化条件编码器...")
        condition_encoder = ConditionEncoder().to(self.device)

        print("初始化EditFlow模型...")
        config = EditFlowConfig(
            max_seq_len=128,
            condition_dim=condition_encoder.output_dim,
            use_condition_injection=True,
            base_model_name=model_name
        )
        config.vocab_size = tokenizer.vocab_size
        model = EditFlowTransformer(config).to(self.device)

        # 使用DataParallel包装模型以支持多GPU
        if self.use_data_parallel and self.gpu_count > 1:
            print(f"使用DataParallel包装模型...")
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            condition_encoder = torch.nn.DataParallel(condition_encoder, device_ids=self.device_ids)

        # 如果提供了检查点路径，加载预训练模型
        checkpoint = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"正在加载预训练模型: {checkpoint_path}")

            # 添加安全全局类以支持weights_only加载
            torch.serialization.add_safe_globals([
                EditFlowConfig,
                argparse.Namespace
            ])

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

            # 获取保存时的状态信息
            saved_model_was_dataparallel = checkpoint.get('model_was_dataparallel', False)
            saved_encoder_was_dataparallel = checkpoint.get('encoder_was_dataparallel', False)

            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']

                # 根据保存和当前状态的差异，调整键名
                current_model_is_dataparallel = hasattr(model, 'module')

                if saved_model_was_dataparallel and not current_model_is_dataparallel:
                    # 保存时是DataParallel，当前不是：移除module.前缀
                    model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
                elif not saved_model_was_dataparallel and current_model_is_dataparallel:
                    # 保存时不是DataParallel，当前是：添加module.前缀
                    model_state = {f'module.{key}': value for key, value in model_state.items()}

                model.load_state_dict(model_state)
                print("✓ EditFlow模型加载完成")

            # 加载条件编码器状态
            if 'condition_encoder_state_dict' in checkpoint:
                encoder_state = checkpoint['condition_encoder_state_dict']

                # 根据保存和当前状态的差异，调整键名
                current_encoder_is_dataparallel = hasattr(condition_encoder, 'module')

                if saved_encoder_was_dataparallel and not current_encoder_is_dataparallel:
                    # 保存时是DataParallel，当前不是：移除module.前缀
                    encoder_state = {key.replace('module.', ''): value for key, value in encoder_state.items()}
                elif not saved_encoder_was_dataparallel and current_encoder_is_dataparallel:
                    # 保存时不是DataParallel，当前是：添加module.前缀
                    encoder_state = {f'module.{key}': value for key, value in encoder_state.items()}

                condition_encoder.load_state_dict(encoder_state)
                print("✓ 条件编码器加载完成")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"EditFlow模型参数数量: {total_params:,}")

        criterion = ContinuousFlowLoss(scheduler_type='cubic')
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # 如果有检查点，也加载优化器状态
        if checkpoint and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ 优化器状态加载完成")

        # 多GPU设置优化
        if self.use_data_parallel and self.gpu_count > 1:
            effective_batch_size = self.args.batch_size * self.gpu_count
            print(f"有效批次大小: {effective_batch_size} (每个GPU: {self.args.batch_size})")

        return model, condition_encoder, criterion, optimizer, tokenizer

  
    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config):
        batch_size = z0_token_ids.size(0)
        t = torch.rand(batch_size, 1, device=self.device)
        effective_vocab_size = max(dataset.gap_token + 1, config.vocab_size + 200)

        z0_probs = tokens_to_prob(z0_token_ids, effective_vocab_size)
        z1_probs = tokens_to_prob(z1_token_ids, effective_vocab_size)
        z_t = sample_conditional_path(z0_probs, z1_probs, t, self.scheduler)

        gap_token = dataset.gap_token
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(
            z_t, effective_vocab_size, dataset.pad_token, gap_token
        )

        attention_mask = (~x_pad_mask).float()
        pred_rates, pred_ins_probs, pred_sub_probs = model(
            input_ids=x_t, time_steps=t, condition=condition_embeddings, attention_mask=attention_mask
        )

        return {
            'pred_rates': pred_rates,
            'pred_ins_probs': pred_ins_probs,
            'pred_sub_probs': pred_sub_probs,
            'x_t': x_t,
            'z_t': z_t,
            'z1_token_ids': z1_token_ids,
            'z_gap_mask': z_gap_mask,
            'z_pad_mask': z_pad_mask,
            't': t,
            'effective_vocab_size': effective_vocab_size,
            'gap_token': gap_token
        }

    def compute_loss(self, forward_results, criterion, dataset):
        pred_rates = forward_results['pred_rates']
        pred_ins_probs = forward_results['pred_ins_probs']
        pred_sub_probs = forward_results['pred_sub_probs']
        x_t = forward_results['x_t']
        z_t = forward_results['z_t']
        z1_token_ids = forward_results['z1_token_ids']
        z_gap_mask = forward_results['z_gap_mask']
        z_pad_mask = forward_results['z_pad_mask']
        t = forward_results['t']
        effective_vocab_size = forward_results['effective_vocab_size']
        gap_token = forward_results['gap_token']

        lambda_ins = pred_rates[:, :, 0:1]
        lambda_sub = pred_rates[:, :, 1:2]
        lambda_del = pred_rates[:, :, 2:3]

        ins_probs = lambda_ins * pred_ins_probs
        sub_probs = lambda_sub * pred_sub_probs

        extended_ins_probs = torch.zeros(x_t.size(0), x_t.size(1), effective_vocab_size, device=x_t.device)
        extended_ins_probs[:, :, :pred_ins_probs.size(-1)] = ins_probs

        extended_sub_probs = torch.zeros(x_t.size(0), x_t.size(1), effective_vocab_size, device=x_t.device)
        extended_sub_probs[:, :, :pred_sub_probs.size(-1)] = sub_probs

        u_cat = torch.cat([lambda_ins * extended_ins_probs, lambda_sub * extended_sub_probs, lambda_del], dim=-1)
        u_z = fill_gap_tokens_with_repeats(u_cat, z_gap_mask, z_pad_mask)
        u_mask = criterion.make_ut_mask_from_z(z_t, z1_token_ids, effective_vocab_size, gap_token, dataset.pad_token)

        loss = criterion(u_z, u_mask, t, effective_vocab_size)
        return loss

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}")

        config = model.module.config if hasattr(model, 'module') else model.config

        if self.use_data_parallel and self.gpu_count > 1:
            progress_bar.set_postfix({'loss': '0.0000', 'gpu_load': get_gpu_memory_usage_string(max_gpus=3)})

        for batch_idx, batch in enumerate(progress_bar):
            x_values = batch['x_values'].to(self.device)
            residuals = batch['residuals'].to(self.device)
            z0_token_ids = batch['z0_token_ids'].to(self.device)
            z1_token_ids = batch['z1_token_ids'].to(self.device)

            condition_embeddings = condition_encoder(x_values, residuals)
            forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config)
            loss = self.compute_loss(forward_results, criterion, dataset) / gradient_accumulation_steps

            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(condition_encoder.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            postfix_dict = {'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'}
            if self.use_data_parallel and self.gpu_count > 1:
                postfix_dict['gpu_load'] = get_gpu_memory_usage_string(max_gpus=3)

            progress_bar.set_postfix(postfix_dict)

        return total_loss / num_batches, num_batches

    
    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config):
        checkpoint_path = os.path.join(self.args.save_dir, f"editflow_epoch_{epoch+1}.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        # 检查模型是否被DataParallel包装
        model_is_dataparallel = hasattr(model, 'module')
        encoder_is_dataparallel = hasattr(condition_encoder, 'module')

        # 保存原始状态字典（保持DataParallel的module.前缀）
        model_state = model.state_dict()
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
            'gpu_count': self.gpu_count,
            # 添加保存时的状态信息
            'model_was_dataparallel': model_is_dataparallel,
            'encoder_was_dataparallel': encoder_is_dataparallel,
            'saved_with_dataparallel_setting': self.use_data_parallel and self.gpu_count > 1
        }, checkpoint_path)

        return checkpoint_path

    def train(self):
        print(f"使用设备: {self.device}")

        # 检查是否有可用的检查点文件
        checkpoint_path = find_latest_checkpoint(self.args)
        if checkpoint_path:
            print(f"找到检查点: {checkpoint_path}")
        else:
            print("未找到检查点，将从基础模型开始训练")

        # 使用setup_models加载模型（优先检查点，然后本地缓存，最后下载）
        model, condition_encoder, criterion, optimizer, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

        # 使用tokenizer准备数据
        dataloaders, datasets, dimension_groups = self.prepare_data(tokenizer)

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"条件编码器参数数量: {sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad):,}")
        print(f"开始连续流训练 ({self.args.num_epochs} epochs)...")

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
                config = model.module.config if hasattr(model, 'module') else model.config
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch, config
                )
                print(f"检查点已保存到: {checkpoint_path}")

        final_model_path = os.path.join(self.args.save_dir, "continuous_flow_final.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        # 检查模型是否被DataParallel包装
        model_is_dataparallel = hasattr(model, 'module')
        encoder_is_dataparallel = hasattr(condition_encoder, 'module')

        # 保存原始状态字典（保持DataParallel的module.前缀）
        model_state = model.state_dict()
        condition_encoder_state = condition_encoder.state_dict()

        config = model.module.config if hasattr(model, 'module') else model.config
        torch.save({
            'epoch': self.args.num_epochs,
            'model_state_dict': model_state,
            'condition_encoder_state_dict': condition_encoder_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'args': self.args,
            'use_data_parallel': self.use_data_parallel,
            'gpu_count': self.gpu_count,
            'scheduler_type': 'cubic',
            # 添加保存时的状态信息
            'model_was_dataparallel': model_is_dataparallel,
            'encoder_was_dataparallel': encoder_is_dataparallel,
            'saved_with_dataparallel_setting': self.use_data_parallel and self.gpu_count > 1
        }, final_model_path)

        print(f"最终模型已保存到: {final_model_path}")
        return model, condition_encoder

    def inference_multi_step(self, model, condition_encoder, tokenizer, x_values, y_values,
                           n_steps=50, device=None):
        """多步推理算法 - 从空白表达式开始构建"""
        if device is None:
            device = self.device

        model.eval()
        condition_encoder.eval()

        # 准备输入数据
        x_values = torch.FloatTensor(x_values).unsqueeze(0).to(device)  # (1, n_points, dim)
        y_values = torch.FloatTensor(y_values).unsqueeze(0).to(device)  # (1, n_points)

        # 初始残差就是目标值本身（因为从空白开始）
        residuals = y_values

        # 计算条件嵌入
        condition = condition_encoder(x_values, residuals)

        # 从x0开始，而不是空白表达式
        current_tokens = ['x0']

        # 多步迭代推理
        special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)

        for step in range(n_steps):
            print(f"推理步骤 {step + 1}/{n_steps}, 当前表达式: {','.join(current_tokens) if current_tokens else '<blank>'}")

            if current_tokens:
                tokenized_expr = special_tokens_manager.tokenize_expression(','.join(current_tokens))
            else:
                tokenized_expr = []

            max_len = 128
            if len(tokenized_expr) > max_len - 1:
                tokenized_expr = tokenized_expr[:max_len-1]

            bos_token = tokenizer.bos_token_id or tokenizer.cls_token_id
            pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id

            tokenized_expr = [bos_token] + tokenized_expr
            tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

            input_ids = torch.LongTensor([tokenized_expr]).to(device)
            seq_len = 1 + len(tokenized_expr)
            attention_mask = torch.zeros(max_len, dtype=torch.float)
            attention_mask[:seq_len] = 1.0
            attention_mask = attention_mask.unsqueeze(0).to(device)

            t = torch.tensor([[0.1 + 0.9 * step / n_steps]], dtype=torch.float32, device=device)

            with torch.no_grad():
                rates, insert_probs, substitute_probs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    time_steps=t, condition=condition
                )

                lambda_ins = rates[0, :, 0].cpu().numpy()
                lambda_sub = rates[0, :, 1].cpu().numpy()
                lambda_del = rates[0, :, 2].cpu().numpy()

                base_length = int(attention_mask[0].sum().item())
                effective_length = max(base_length, min(10, input_ids.size(1)))

                best_pos = 0
                best_action = None
                best_score = -1

                for pos in range(1, effective_length):
                    if lambda_ins[pos] > best_score:
                        best_score = lambda_ins[pos]
                        best_action = ('insert', pos-1)

                    current_token_idx = pos - 1
                    if current_token_idx < len(current_tokens) and lambda_sub[pos] > best_score:
                        best_score = lambda_sub[pos]
                        best_action = ('substitute', current_token_idx)

                    if current_token_idx < len(current_tokens) and lambda_del[pos] > best_score:
                        best_score = lambda_del[pos]
                        best_action = ('delete', current_token_idx)

                if best_action and best_score > 0.01:
                    action_type, pos = best_action

                    if action_type == 'insert':
                        best_token = torch.argmax(insert_probs[0, pos]).item()
                        token_map = special_tokens_manager.get_function_token_map()

                        for i in range(special_tokens_manager.max_dim):
                            var_name = f'x{i}'
                            tokens = special_tokens_manager.tokenizer.encode(var_name, add_special_tokens=False)
                            if tokens:
                                token_map[var_name] = tokens[0]

                        for op_name in special_tokens_manager.OPERATORS:
                            tokens = special_tokens_manager.tokenizer.encode(op_name, add_special_tokens=False)
                            if tokens:
                                token_map[op_name] = tokens[0]

                        found = False
                        for expr_elem, token_id in token_map.items():
                            if token_id == best_token:
                                current_tokens.insert(pos, expr_elem)
                                found = True
                                break

                        if not found:
                            current_tokens.insert(pos, 'x0')

                    elif action_type == 'substitute' and pos < len(current_tokens):
                        best_token = torch.argmax(substitute_probs[0, pos]).item()
                        token_map = special_tokens_manager.get_function_token_map()

                        for i in range(special_tokens_manager.max_dim):
                            var_name = f'x{i}'
                            tokens = special_tokens_manager.tokenizer.encode(var_name, add_special_tokens=False)
                            if tokens:
                                token_map[var_name] = tokens[0]

                        for expr_elem, token_id in token_map.items():
                            if token_id == best_token:
                                current_tokens[pos] = expr_elem
                                break

                    elif action_type == 'delete' and pos < len(current_tokens):
                        del current_tokens[pos]

            if len(current_tokens) > 50:
                current_tokens = current_tokens[:50]

        # 返回最终的表达式
        final_expression = ','.join(current_tokens) if current_tokens else ""
        print(f"最终表达式: {final_expression}")

        return final_expression

    def symbolic_regression(self, model_path, x_data, y_data):
        """符号回归主函数 - 接收数据点对，输出表达式"""
        print("开始符号回归推理...")
        print(f"输入数据: x形状={x_data.shape}, y形状={y_data.shape}")

        # 检查模型路径是否存在
        checkpoint_path = model_path if model_path and os.path.exists(model_path) else None
        if checkpoint_path:
            print(f"使用检查点: {checkpoint_path}")
        else:
            print("未找到检查点，将使用基础模型进行推理")

        # 使用setup_models加载所有模型组件
        model, condition_encoder, criterion, optimizer, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

        # 执行多步推理
        final_expression = self.inference_multi_step(
            model=model,
            condition_encoder=condition_encoder,
            tokenizer=tokenizer,
            x_values=x_data,
            y_values=y_data,
            n_steps=30  # 推理步数
        )

        return final_expression
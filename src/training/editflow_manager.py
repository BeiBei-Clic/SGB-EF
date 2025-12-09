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


class KappaScheduler:
    """时间调度器，用于控制流的插值"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler_type = scheduler_type

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """返回时间t的调度系数"""
        if self.scheduler_type == 'cubic':
            return 3 * t**2 - 2 * t**3
        elif self.scheduler_type == 'linear':
            return t
        else:
            return t

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """返回时间t的调度系数导数"""
        if self.scheduler_type == 'cubic':
            return 6 * t - 6 * t**2
        elif self.scheduler_type == 'linear':
            return torch.ones_like(t)
        else:
            return torch.ones_like(t)


def sample_conditional_path(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, scheduler: KappaScheduler) -> torch.Tensor:
    """在给定时间t采样条件路径"""
    # 确保p0和p1的形状一致
    if p0.shape != p1.shape:
        # 如果形状不一致，调整到较大的形状
        max_seq_len = max(p0.size(1), p1.size(1))
        max_vocab_size = max(p0.size(2), p1.size(2))
        batch_size = p0.size(0)

        # 扩展p0
        if p0.size(1) < max_seq_len or p0.size(2) < max_vocab_size:
            p0_expanded = torch.zeros(batch_size, max_seq_len, max_vocab_size, device=p0.device)
            p0_expanded[:, :p0.size(1), :p0.size(2)] = p0
            p0 = p0_expanded

        # 扩展p1
        if p1.size(1) < max_seq_len or p1.size(2) < max_vocab_size:
            p1_expanded = torch.zeros(batch_size, max_seq_len, max_vocab_size, device=p1.device)
            p1_expanded[:, :p1.size(1), :p1.size(2)] = p1
            p1 = p1_expanded

    batch_size, seq_len, vocab_size = p0.shape
    t = t.view(-1, 1, 1) if t.dim() == 1 else t
    kappa_t = scheduler(t)

    # 确保kappa_t的形状与p0和p1匹配
    if kappa_t.dim() == 3 and kappa_t.shape[1] == 1 and kappa_t.shape[2] == 1:
        # 扩展到序列长度维度
        kappa_t = kappa_t.expand(-1, seq_len, -1)
    elif kappa_t.dim() == 2:
        kappa_t = kappa_t.unsqueeze(-1).expand(-1, seq_len, -1)

    pt = (1 - kappa_t) * p0 + kappa_t * p1

    # 数值稳定性：归一化概率
    pt = pt / (pt.sum(dim=-1, keepdim=True) + 1e-8)

    # 采样
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
    if gap_token_id is None:
        # 如果没有明确的gap token，使用词汇表外的一个值
        gap_token_id = vocab_size + 1

    batch_size, z_seq_len = z_t.shape
    device = z_t.device

    # 找到gap token的位置
    gap_mask = (z_t == gap_token_id)
    z_gap_mask = gap_mask
    z_pad_mask = (z_t == pad_token_id)

    # 移除gap token
    x_t_list = []
    x_pad_mask_list = []

    for i in range(batch_size):
        z_row = z_t[i]
        non_gap_indices = ~gap_mask[i]
        x_row = z_row[non_gap_indices]
        x_t_list.append(x_row)
        x_pad_mask_list.append((x_row == pad_token_id))

    # 填充到相同长度
    max_x_len = max(len(x) for x in x_t_list)
    x_t_padded = torch.full((batch_size, max_x_len), pad_token_id, dtype=torch.long, device=device)
    x_pad_mask_padded = torch.zeros((batch_size, max_x_len), dtype=torch.bool, device=device)

    for i, x_row in enumerate(x_t_list):
        x_t_padded[i, :len(x_row)] = x_row
        x_pad_mask_padded[i, :len(x_row)] = x_pad_mask_list[i]

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
        self.pad_token = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id if tokenizer.bos_token is not None else tokenizer.cls_token_id

        # gap token - 使用词汇表外的一个ID
        self.gap_token = self.vocab_size + 100  # 确保不与正常token冲突

        # 特殊token管理器
        self.max_dim = 10
        self.special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=self.max_dim)

    def __len__(self):
        return len(self.samples)

    def _tokenize_expression_tokens(self, tokens: List[str]) -> List[int]:
        """将token列表转换为token ID列表"""
        token_ids = []
        for token in tokens:
            if token == "<gap>":
                token_ids.append(self.gap_token)
            else:
                # 使用special_tokens_manager处理其他tokens
                tree_str = token if ',' in token else token
                tokenized = self.special_tokens_manager.tokenize_expression(tree_str)
                token_ids.extend(tokenized)
        return token_ids

    def validate_data_sample(self, sample: dict, idx: int) -> dict:
        """验证数据样本的合理性"""
        x_values = torch.FloatTensor(sample['x_values'])
        if torch.isnan(x_values).any() or torch.isinf(x_values).any():
            raise ValueError(f"样本 {idx} 的 x_values 包含 NaN 或 Inf 值")

        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)
        if torch.isnan(residuals).any() or torch.isinf(residuals).any():
            raise ValueError(f"样本 {idx} 的 residuals 包含 NaN 或 Inf 值")

        return sample

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample = self.validate_data_sample(sample, idx)

        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)

        # 对齐后的Z空间序列
        z0_tokens = self._tokenize_expression_tokens(sample['z0_tokens'])
        z1_tokens = self._tokenize_expression_tokens(sample['z1_tokens'])

        # 添加BOS token并填充
        max_len = 128
        def pad_z_sequence(tokens):
            tokens = [self.bos_token] + tokens[:max_len-1]
            tokens.extend([self.pad_token] * (max_len - len(tokens)))
            return torch.LongTensor(tokens)

        z0_token_ids = pad_z_sequence(z0_tokens)
        z1_token_ids = pad_z_sequence(z1_tokens)

        return {
            'x_values': x_values,
            'residuals': residuals,
            'z0_token_ids': z0_token_ids,
            'z1_token_ids': z1_token_ids,
            'gap_token': self.gap_token
        }


class ContinuousFlowLoss:
    """连续时间流匹配损失函数"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler = KappaScheduler(scheduler_type)

    def make_ut_mask_from_z(self, z_t: torch.Tensor, z_1: torch.Tensor, vocab_size: int,
                           gap_token: int, pad_token: int) -> torch.Tensor:
        """
        创建用于计算损失的目标mask
        对于每个位置i，指示哪些操作可以将z_t[i]转换为z_1[i]
        """
        batch_size, z_seq_len = z_t.shape
        n_ops = 2 * vocab_size + 1  # insert + substitute + delete

        z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
        z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq
        z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq
        z_sub = z_neq & ~z_ins & ~z_del

        # mask: (batch_size, z_seq_len, n_ops)
        u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)

        # Insert operations: target token at gap positions
        u_mask[z_ins, z_1[z_ins]] = True

        # Substitute operations: from current to target token
        u_mask[z_sub, z_1[z_sub] + vocab_size] = True

        # Delete operations: last position in operation space
        u_mask[:, :, -1][z_del] = True

        return u_mask

    def __call__(self, u_cat: torch.Tensor, u_mask: torch.Tensor,
                 t: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        计算连续流匹配损失

        Args:
            u_cat: 模型预测的编辑操作率 (batch_size, seq_len, 2*vocab_size+1)
            u_mask: 目标操作mask (batch_size, seq_len, 2*vocab_size+1)
            t: 时间步 (batch_size, 1)
        """
        # 总操作率
        u_total = u_cat.sum(dim=(1, 2))

        # 调度器系数
        sched_coeff = (self.scheduler.derivative(t) / (1 - self.scheduler(t) + 1e-8)).squeeze(-1)
        sched_coeff = torch.clamp(sched_coeff, min=-10, max=10)

        # 交叉熵项，只计算需要的操作
        log_u_cat = torch.log(torch.clamp(u_cat, min=1e-12, max=1e12))
        cross_entropy = (log_u_cat * u_mask.float()).sum(dim=(1, 2))

        # Bregman散度损失
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
        model_name = getattr(self.args, 'base_model_name', "openai-community/gpt2")

        # 设置模型缓存目录
        cache_dir = getattr(self.args, 'cache_dir', "models/huggingface_cache")
        import os
        os.makedirs(cache_dir, exist_ok=True)

        print(f"正在加载tokenizer: {model_name}")
        print(f"模型缓存目录: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"✓ Tokenizer加载完成")

        # 确保tokenizer有pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

    def _get_model_config(self, model):
        """获取模型配置，处理DataParallel包装"""
        if hasattr(model, 'module'):
            return model.module.config
        return model.config

    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config):
        """
        执行模型前向传播，返回预测结果和损失计算所需的所有数据

        Args:
            model: EditFlow模型
            condition_embeddings: 预计算的条件嵌入向量
            z0_token_ids: 起始token序列
            z1_token_ids: 目标token序列
            dataset: 数据集对象
            config: 模型配置

        Returns:
            dict: 包含预测结果和损失计算所需数据的字典
        """
        # 随机采样时间步
        batch_size = z0_token_ids.size(0)
        t = torch.rand(batch_size, 1, device=self.device)

        # 计算词汇表大小（包括gap token）
        effective_vocab_size = max(dataset.gap_token + 1, config.vocab_size + 200)

        # 在Z空间中插值采样
        z0_probs = tokens_to_prob(z0_token_ids, effective_vocab_size)
        z1_probs = tokens_to_prob(z1_token_ids, effective_vocab_size)
        z_t = sample_conditional_path(z0_probs, z1_probs, t, self.scheduler)

        # 移除gap token，得到x_t
        gap_token = dataset.gap_token
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(
            z_t, effective_vocab_size, dataset.pad_token, gap_token
        )

        # 模型前向传播
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
        """
        根据前向传播结果计算损失

        Args:
            forward_results: forward_pass返回的结果字典
            criterion: 损失函数
            dataset: 数据集对象

        Returns:
            torch.Tensor: 计算得到的损失
        """
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

        # 构建编辑操作率张量
        lambda_ins = pred_rates[:, :, 0:1]
        lambda_sub = pred_rates[:, :, 1:2]
        lambda_del = pred_rates[:, :, 2:3]

        # 计算插入和替换的概率
        ins_probs = lambda_ins * pred_ins_probs
        sub_probs = lambda_sub * pred_sub_probs

        # 扩展到完整的词汇表大小
        extended_ins_probs = torch.zeros(x_t.size(0), x_t.size(1), effective_vocab_size, device=x_t.device)
        extended_ins_probs[:, :, :pred_ins_probs.size(-1)] = ins_probs

        extended_sub_probs = torch.zeros(x_t.size(0), x_t.size(1), effective_vocab_size, device=x_t.device)
        extended_sub_probs[:, :, :pred_sub_probs.size(-1)] = sub_probs

        # 组合所有操作率 (batch_size, seq_len, 2*vocab_size+1)
        u_cat = torch.cat([
            lambda_ins * extended_ins_probs,  # 插入操作
            lambda_sub * extended_sub_probs,  # 替换操作
            lambda_del                        # 删除操作
        ], dim=-1)

        # 将x_t的输出扩展到z_t空间
        u_z = fill_gap_tokens_with_repeats(u_cat, z_gap_mask, z_pad_mask)

        # 计算目标mask
        u_mask = criterion.make_ut_mask_from_z(
            z_t, z1_token_ids, effective_vocab_size, gap_token, dataset.pad_token
        )

        # 计算损失
        loss = criterion(u_z, u_mask, t, effective_vocab_size)
        return loss

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
            z0_token_ids = batch['z0_token_ids'].to(self.device)
            z1_token_ids = batch['z1_token_ids'].to(self.device)

            # 预计算条件嵌入（在训练循环外部一次性计算）
            condition_embeddings = condition_encoder(x_values, residuals)

            # 前向传播
            forward_results = self.forward_pass(
                model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config
            )

            # 计算损失
            loss = self.compute_loss(forward_results, criterion, dataset)

            # 梯度累积
            loss = loss / gradient_accumulation_steps
            optimizer.zero_grad()

            if torch.isnan(loss):
                continue

            loss.backward()

            # 执行优化步骤
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # 梯度检查
                has_nan_gradients = self._check_gradients(model, condition_encoder)

                if not has_nan_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(condition_encoder.parameters(), 1.0)
                    optimizer.step()
                else:
                    optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # 显示进度
            postfix_dict = {
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'
            }
            if self.use_data_parallel and self.gpu_count > 1:
                postfix_dict['gpu_load'] = get_gpu_memory_usage_string(max_gpus=3)

            progress_bar.set_postfix(postfix_dict)

        return total_loss / num_batches, num_batches

    def _check_gradients(self, model, condition_encoder):
        """检查模型和条件编码器的梯度是否包含NaN或Inf"""
        has_nan_gradients = False

        # 检查模型梯度
        if hasattr(model, 'module'):
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

        return has_nan_gradients

    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config):
        import os
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

    def _find_latest_checkpoint(self):
        """查找最新的检查点文件"""
        import os
        import glob

        save_dir = getattr(self.args, 'save_dir', 'checkpoints')

        # 查找所有epoch检查点
        pattern = os.path.join(save_dir, "editflow_epoch_*.pth")
        checkpoint_files = glob.glob(pattern)

        if checkpoint_files:
            # 提取epoch数字并排序
            def get_epoch_number(filepath):
                filename = os.path.basename(filepath)
                epoch_str = filename.replace('editflow_epoch_', '').replace('.pth', '')
                return int(epoch_str)

            # 返回最新epoch的检查点
            latest_checkpoint = max(checkpoint_files, key=get_epoch_number)
            return latest_checkpoint

        # 如果没有epoch检查点，尝试final模型
        final_model = os.path.join(save_dir, "continuous_flow_final.pth")
        if os.path.exists(final_model):
            return final_model

        return None

    def train(self):
        print(f"使用设备: {self.device}")

        # 检查是否有可用的检查点文件
        checkpoint_path = self._find_latest_checkpoint()
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
                config = self._get_model_config(model)
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch, config
                )
                print(f"检查点已保存到: {checkpoint_path}")

        import os
        final_model_path = os.path.join(self.args.save_dir, "continuous_flow_final.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        # 检查模型是否被DataParallel包装
        model_is_dataparallel = hasattr(model, 'module')
        encoder_is_dataparallel = hasattr(condition_encoder, 'module')

        # 保存原始状态字典（保持DataParallel的module.前缀）
        model_state = model.state_dict()
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
        for step in range(n_steps):
            print(f"推理步骤 {step + 1}/{n_steps}, 当前表达式: {','.join(current_tokens) if current_tokens else '<blank>'}")

            # 将当前表达式转换为token IDs
            special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=10)

            if current_tokens:
                tokenized_expr = special_tokens_manager.tokenize_expression(','.join(current_tokens))
            else:
                tokenized_expr = []

            # 添加BOS token并填充
            max_len = 128
            if len(tokenized_expr) > max_len - 1:
                tokenized_expr = tokenized_expr[:max_len-1]

            bos_token = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

            tokenized_expr = [bos_token] + tokenized_expr
            tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

            input_ids = torch.LongTensor([tokenized_expr]).to(device)

            # 正确构建attention_mask：只有实际有内容的位置才为1
            # 即使BOS和PAD是同一个token，我们也要区分哪些位置是"有意义"的
            seq_len = 1 + len(tokenized_expr)  # BOS + 实际tokens
            attention_mask = torch.zeros(max_len, dtype=torch.float)
            attention_mask[:seq_len] = 1.0  # 前seq_len个位置是有效的
            attention_mask = attention_mask.unsqueeze(0).to(device)  # (1, max_len)

            # 调试信息
            if step < 3:
                print(f"  调试: BOS={bos_token}, PAD={pad_token}, 实际seq_len={seq_len}")
                print(f"  调试: attention_mask前10个={attention_mask[0][:10].tolist()}")

            # 时间步使用更合理的分布，从0.1开始递增
            t = torch.tensor([[0.1 + 0.9 * step / n_steps]], dtype=torch.float32, device=device)

            with torch.no_grad():
                # 模型预测
                rates, insert_probs, substitute_probs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    time_steps=t,
                    condition=condition
                )

                # 调试信息：打印模型输出
                if step < 3:
                    print(f"  调试: input_ids={input_ids[0][:10].tolist()}")
                    print(f"  调试: attention_mask={attention_mask[0][:10].tolist()}")
                    print(f"  调试: time_step={t.item():.4f}")
                    print(f"  调试: condition shape={condition.shape}, condition sum={condition.sum().item():.4f}")
                    print(f"  调试: rates shape={rates.shape}, rates min={rates.min().item():.6f}, max={rates.max().item():.6f}")
                    print(f"  调试: rates前3个位置={rates[0, :3, :].cpu().numpy()}")
                    print(f"  调试: insert_probs shape={insert_probs.shape}, min={insert_probs.min().item():.6f}, max={insert_probs.max().item():.6f}")
                    print(f"  调试: substitute_probs shape={substitute_probs.shape}, min={substitute_probs.min().item():.6f}, max={substitute_probs.max().item():.6f}")

                # 解析预测结果
                lambda_ins = rates[0, :, 0].cpu().numpy()  # 插入率
                lambda_sub = rates[0, :, 1].cpu().numpy()  # 替换率
                lambda_del = rates[0, :, 2].cpu().numpy()  # 删除率

                # 根据概率决定编辑操作
                # 修复：确保至少有几个位置可用于编辑，即使原始序列很短
                base_length = int(attention_mask[0].sum().item())
                effective_length = max(base_length, min(10, input_ids.size(1)))  # 至少10个位置或最大长度

                # 找到最可能的编辑操作
                best_pos = 0
                best_action = None
                best_score = -1

                # 调试信息：打印当前rates
                if step < 5:  # 只在前5步打印调试信息
                    print(f"  调试: effective_length={effective_length}, current_tokens={current_tokens}")
                    print(f"  调试: lambda_ins前5个={lambda_ins[:min(5, len(lambda_ins))]}")
                    print(f"  调试: lambda_sub前5个={lambda_sub[:min(5, len(lambda_sub))]}")
                    print(f"  调试: lambda_del前5个={lambda_del[:min(5, len(lambda_del))]}")

                for pos in range(1, effective_length):
                    # 检查插入操作（在当前位置前插入）
                    if lambda_ins[pos] > best_score:
                        best_score = lambda_ins[pos]
                        best_action = ('insert', pos-1)  # 修正插入位置

                    # 检查替换操作（如果位置有token）
                    current_token_idx = pos - 1  # 对应current_tokens中的索引
                    if current_token_idx < len(current_tokens) and lambda_sub[pos] > best_score:
                        best_score = lambda_sub[pos]
                        best_action = ('substitute', current_token_idx)

                    # 检查删除操作（如果位置有token）
                    if current_token_idx < len(current_tokens) and lambda_del[pos] > best_score:
                        best_score = lambda_del[pos]
                        best_action = ('delete', current_token_idx)

                # 调试信息：打印找到的最佳操作
                if step < 5:
                    print(f"  调试: 找到的最佳操作: {best_action}, 分数: {best_score:.6f}")

                # 执行最佳操作
                if best_action and best_score > 0.01:  # 降低阈值允许更多编辑操作
                    action_type, pos = best_action
                    if step < 5:
                        print(f"  调试: 执行操作 {action_type} 在位置 {pos}")

                    if action_type == 'insert':
                        # 插入最高概率的token
                        best_token = torch.argmax(insert_probs[0, pos]).item()

                        if step < 5:
                            print(f"  调试: 插入操作，位置={pos}，最高概率token ID={best_token}")

                        # 获取所有可能的token映射
                        token_map = special_tokens_manager.get_function_token_map()
                        # 添加变量映射
                        for i in range(special_tokens_manager.max_dim):
                            var_name = f'x{i}'
                            tokens = special_tokens_manager.tokenizer.encode(var_name, add_special_tokens=False)
                            if tokens:
                                token_map[var_name] = tokens[0]

                        # 添加运算符映射
                        for op_name in special_tokens_manager.OPERATORS:
                            tokens = special_tokens_manager.tokenizer.encode(op_name, add_special_tokens=False)
                            if tokens:
                                token_map[op_name] = tokens[0]

                        # 查找对应的表达式元素
                        found = False
                        for expr_elem, token_id in token_map.items():
                            if token_id == best_token:
                                current_tokens.insert(pos, expr_elem)
                                found = True
                                if step < 5:
                                    print(f"  调试: 成功插入 '{expr_elem}' (token ID={best_token})")
                                break

                        if not found and step < 5:
                            print(f"  调试: 未找到token ID {best_token} 对应的表达式元素")
                            # 如果找不到，插入一个默认的变量
                            current_tokens.insert(pos, 'x0')
                            print(f"  调试: 默认插入 'x0'")

                    elif action_type == 'substitute' and pos < len(current_tokens):
                        # 替换为最高概率的token
                        best_token = torch.argmax(substitute_probs[0, pos]).item()
                        # 获取所有可能的token映射
                        token_map = special_tokens_manager.get_function_token_map()
                        # 添加变量映射
                        for i in range(special_tokens_manager.max_dim):
                            var_name = f'x{i}'
                            tokens = special_tokens_manager.tokenizer.encode(var_name, add_special_tokens=False)
                            if tokens:
                                token_map[var_name] = tokens[0]

                        # 查找对应的表达式元素
                        for expr_elem, token_id in token_map.items():
                            if token_id == best_token:
                                current_tokens[pos] = expr_elem
                                break

                    elif action_type == 'delete' and pos < len(current_tokens):
                        # 删除token
                        del current_tokens[pos]

            # 限制表达式长度
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
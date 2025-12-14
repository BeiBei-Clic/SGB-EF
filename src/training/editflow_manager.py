"""
EditFlow连续流训练器 - 实现基于连续时间流匹配的编辑流模型训练
"""

import torch
import numpy as np
import time
import argparse
import os
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer

from ..utils.special_tokens import SpecialTokensManager
from ..symbolic.data_generator import generate_flow_samples, load_dimension_index
from .flow import (
    KappaScheduler, sample_conditional_path, tokens_to_prob,
    remove_gap_tokens, fill_gap_tokens_with_repeats,
    ContinuousFlowLoss, FlowDataset, custom_collate_fn
)
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig
from ..utils.gpu_monitor import get_gpu_memory_info, get_gpu_memory_usage_string
from ..utils.misc_utils import find_latest_checkpoint, load_checkpoint

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
        """准备训练数据，优先从本地缓存文件加载，并划分训练集和测试集"""
        print("准备连续流训练数据...")

        # 数据文件路径
        cache_filename = f"data/flow_samples_{self.args.num_samples}_{self.args.max_dim}dim_{self.args.n_points}pts_{self.args.max_depth}depth.txt"

        # 生成数据（如果需要），内部会检查文件是否存在并决定是否需要生成
        generate_flow_samples(
            num_samples=self.args.num_samples,
            max_dim=self.args.max_dim,
            n_points=self.args.n_points,
            max_depth=self.args.max_depth,
            max_expr_length=self.args.max_expr_length
        )

        # 加载维度索引，如果不存在则扫描文件并保存索引
        dimension_samples = load_dimension_index(cache_filename)

        # 划分训练集和测试集
        test_split = getattr(self.args, 'test_split', 0.2)
        train_dataloaders = {}
        test_dataloaders = {}
        train_datasets = {}
        test_datasets = {}

        for dim, all_positions in dimension_samples.items():
            # 打乱样本位置索引
            np.random.shuffle(all_positions)

            # 计算划分点
            split_idx = int(len(all_positions) * (1 - test_split))
            train_positions = all_positions[:split_idx]
            test_positions = all_positions[split_idx:]

            # 创建基于位置的文件数据集
            train_dataset = FlowDataset(
                train_positions, cache_filename, tokenizer,
                max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length
            )
            test_dataset = FlowDataset(
                test_positions, cache_filename, tokenizer,
                max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length
            )

            # 创建DataLoader
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.args.batch_size, shuffle=True,
                num_workers=0, collate_fn=custom_collate_fn
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.args.batch_size, shuffle=False,
                num_workers=0, collate_fn=custom_collate_fn
            )

            train_dataloaders[dim] = train_dataloader
            test_dataloaders[dim] = test_dataloader
            train_datasets[dim] = train_dataset
            test_datasets[dim] = test_dataset

            print(f"维度 {dim}: 训练样本 {len(train_positions)}, 测试样本 {len(test_positions)}")

        print(f"数据划分完成:")
        print(f"  训练集: {sum(len(dataset) for dataset in train_datasets.values())} 个样本")
        print(f"  测试集: {sum(len(dataset) for dataset in test_datasets.values())} 个样本")
        print(f"  分布在 {len(train_datasets)} 个维度组中")

        return train_dataloaders, train_datasets, test_dataloaders, test_datasets

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
        print(f"✓ Tokenizer加载完成，原始词表大小: {tokenizer.vocab_size}")

        # 初始化特殊符号管理器并添加缺失的符号
        special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=self.args.max_dim)
        special_tokens_manager.ensure_special_tokens()

        print("初始化条件编码器...")
        condition_encoder = ConditionEncoder(model_name=self.args.condition_model_name).to(self.device)

        print("初始化EditFlow模型...")
        actual_vocab_size = len(tokenizer.get_vocab())  # 获取实际词表大小
        config = EditFlowConfig(
            max_seq_len=self.args.max_expr_length,
            condition_dim=condition_encoder.output_dim,
            base_model_name=model_name,
            vocab_size=actual_vocab_size,  # 使用更新后的词表大小
        )
        model = EditFlowTransformer(config).to(self.device)

        # 调整模型embedding层大小以匹配新的词表
        if actual_vocab_size > model.base_model.get_input_embeddings().num_embeddings:
            print(f"调整模型embedding层大小: {model.base_model.get_input_embeddings().num_embeddings} -> {actual_vocab_size}")
            model.base_model.resize_token_embeddings(actual_vocab_size)

        # 创建优化器
        criterion = ContinuousFlowLoss(scheduler_type='cubic')
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # 如果提供了检查点路径，加载预训练模型
        checkpoint = load_checkpoint(checkpoint_path, model, condition_encoder, self.device, optimizer)

        # 使用DataParallel包装模型以支持多GPU
        if self.use_data_parallel and self.gpu_count > 1:
            print(f"使用DataParallel包装模型...")
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            condition_encoder = torch.nn.DataParallel(condition_encoder, device_ids=self.device_ids)
            
            # 多GPU设置优化
            effective_batch_size = self.args.batch_size * self.gpu_count
            print(f"有效批次大小: {effective_batch_size} (每个GPU: {self.args.batch_size})")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"EditFlow模型参数数量: {total_params:,}")

        return model, condition_encoder, criterion, optimizer, tokenizer

  
    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config, debug_info=None):
        batch_size = z0_token_ids.size(0)
        t = torch.rand(batch_size, 1, device=self.device)

        # 获取debug参数
        debug_mode = getattr(self.args, 'debug', False)

        # 调试：检查z0和z1 token IDs的有效性（仅在debug模式且为第一个batch时）
        # if debug_info and debug_info.get('is_first_batch', False) and debug_mode:
        #     print(f"\n[DEBUG] {debug_info.get('context', '')} z0_token_ids统计: min={z0_token_ids.min().item()}, max={z0_token_ids.max().item()}, shape={z0_token_ids.shape}")
        #     print(f"[DEBUG] {debug_info.get('context', '')} z1_token_ids统计: min={z1_token_ids.min().item()}, max={z1_token_ids.max().item()}, shape={z1_token_ids.shape}")
        #     print(f"[DEBUG] vocab_size={config.vocab_size}")

        #     # 检查是否有越界的token IDs
        #     z0_valid = (z0_token_ids >= 0) & (z0_token_ids < config.vocab_size)
        #     z1_valid = (z1_token_ids >= 0) & (z1_token_ids < config.vocab_size)
        #     print(f"[DEBUG] z0_token_ids有效率: {z0_valid.float().mean().item():.4f}")
        #     print(f"[DEBUG] z1_token_ids有效率: {z1_valid.float().mean().item():.4f}")

        z0_probs = tokens_to_prob(z0_token_ids, config.vocab_size)
        z1_probs = tokens_to_prob(z1_token_ids, config.vocab_size)

        # 调试：检查概率分布（仅在debug模式且为第一个batch时）
        # if debug_info and debug_info.get('is_first_batch', False) and debug_mode:
        #     print(f"[DEBUG] {debug_info.get('context', '')} z0_probs: {z0_probs}")
        #     print(f"[DEBUG] {debug_info.get('context', '')} z1_probs: {z1_probs}")
        #     print(f"[DEBUG] t: min={t.min().item()}, max={t.max().item()}, mean={t.mean().item():.4f}")

        z_t = sample_conditional_path(z0_probs, z1_probs, t, self.scheduler, debug=debug_mode)

        # print(f"[DEBUG]  z_t: {z_t}")

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(
            z_t, dataset.special_tokens_manager
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
            'vocab_size': config.vocab_size,
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
        effective_vocab_size = forward_results['vocab_size']
        gap_token = dataset.special_tokens_manager.get_token_id('gap')

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
        u_mask = criterion.make_ut_mask_from_z(z_t, z1_token_ids, effective_vocab_size, gap_token, dataset.special_tokens_manager)

        loss = criterion(u_z, u_mask, t, effective_vocab_size)
        return loss

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}")

        # 处理DataParallel包装的情况
        config = model.module.config if hasattr(model, 'module') else model.config

        # 在epoch开始时清零梯度
        optimizer.zero_grad()

        if self.use_data_parallel and self.gpu_count > 1:
            progress_bar.set_postfix({'loss': '0.0000', 'gpu_load': get_gpu_memory_usage_string(max_gpus=3)})

        for batch_idx, batch in enumerate(progress_bar):
            x_values = batch['x_values'].to(self.device)
            residuals = batch['residuals'].to(self.device)
            z0_token_ids = batch['z0_token_ids'].to(self.device)
            z1_token_ids = batch['z1_token_ids'].to(self.device)

            # 调试输出：解码z0和z1的token序列（仅在debug模式且第一个batch时）
            debug_mode = getattr(self.args, 'debug', False)
            if batch_idx == 0 and debug_mode:  # 只在debug模式且第一个batch输出调试信息
                print(f"\n[DEBUG] 维度 {dimension} - 第一个batch的token解码信息:")

                # 解码z0_token_ids
                vocab = dataset.special_tokens_manager.tokenizer.get_vocab()
                id_to_token = {v: k for k, v in vocab.items()}

                print("[DEBUG] z0_token_ids解码结果 (前3个样本):")
                for i in range(min(3, z0_token_ids.size(0))):
                    z0_tokens = []
                    for token_id in z0_token_ids[i].tolist():
                        if token_id in id_to_token:
                            token = id_to_token[token_id]
                            z0_tokens.append(token)
                    z0_expression = ','.join(z0_tokens) if z0_tokens else "<empty>"
                    print(f"  样本{i}: {z0_expression}")

                print("[DEBUG] z1_token_ids解码结果 (前3个样本):")
                for i in range(min(3, z1_token_ids.size(0))):
                    z1_tokens = []
                    for token_id in z1_token_ids[i].tolist():
                        if token_id in id_to_token:
                            token = id_to_token[token_id]
                            z1_tokens.append(token)
                    z1_expression = ','.join(z1_tokens) if z1_tokens else "<empty>"
                    print(f"  样本{i}: {z1_expression}")

            condition_embeddings = condition_encoder(x_values, residuals)

            # 准备调试信息
            debug_info = None
            if batch_idx == 0:
                debug_info = {
                    'is_first_batch': True,
                    'context': f'维度{dimension}'
                }

            forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config, debug_info)
            loss = self.compute_loss(forward_results, criterion, dataset) / gradient_accumulation_steps

            grad_norm = 0.0
            if not torch.isnan(loss):
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # 统一计算梯度范数并裁剪
                    all_params = list(model.parameters()) + list(condition_encoder.parameters())
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()  # 在step后清零梯度，为下一次累积做准备

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            postfix_dict = {
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'grad_norm': f'{grad_norm:.3f}' if isinstance(grad_norm, (int, float)) else f'{grad_norm.item():.3f}'
            }
            if self.use_data_parallel and self.gpu_count > 1:
                postfix_dict['gpu_load'] = get_gpu_memory_usage_string(max_gpus=3)

            progress_bar.set_postfix(postfix_dict)

        return total_loss / num_batches, num_batches

    def evaluate(self, model, condition_encoder, criterion, test_dataloaders, test_datasets):
        """测试集评估"""
        model.eval()
        condition_encoder.eval()

        total_loss = 0.0
        num_batches = 0
        config = model.module.config if hasattr(model, 'module') else model.config

        with torch.no_grad():
            for dim, dataloader in test_dataloaders.items():
                dataset = test_datasets[dim]
                for batch in dataloader:
                    x_values = batch['x_values'].to(self.device)
                    residuals = batch['residuals'].to(self.device)
                    z0_token_ids = batch['z0_token_ids'].to(self.device)
                    z1_token_ids = batch['z1_token_ids'].to(self.device)

                    condition_embeddings = condition_encoder(x_values, residuals)
                    forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config)
                    loss = self.compute_loss(forward_results, criterion, dataset)

                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float('inf')

    
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
        train_dataloaders, train_datasets, test_dataloaders, test_datasets = self.prepare_data(tokenizer)

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"条件编码器参数数量: {sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad):,}")
        print(f"开始连续流训练 ({self.args.num_epochs} epochs)...")

        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            total_batches = 0

            for dim, dataloader in train_dataloaders.items():
                print(f"\n训练维度 {dim} 的数据...")
                dataset = train_datasets[dim]
                dim_loss, dim_batches = self.train_epoch(
                    model, condition_encoder, criterion, optimizer,
                    dataloader, dataset, epoch, dim
                )
                total_loss += dim_loss * dim_batches
                total_batches += dim_batches
                print(f"维度 {dim} 平均损失: {dim_loss:.4f}")

            avg_loss = total_loss / total_batches
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs} 完成, 训练损失: {avg_loss:.4f}")

            # 测试集评估
            eval_every = getattr(self.args, 'eval_every', 5)
            if (epoch + 1) % eval_every == 0 or epoch == self.args.num_epochs - 1:
                print("开始测试集评估...")
                test_loss = self.evaluate(model, condition_encoder, criterion, test_dataloaders, test_datasets)
                print(f"测试集损失: {test_loss:.4f}")

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

    def symbolic_regression(self, model_path, x_data, y_data, debug_mode=False, n_steps=100, input_dim=None, max_expr_length=None):
        """符号回归 - 接收数据点对，输出表达式

        Args:
            model_path: 模型检查点路径
            x_data: 输入x数据
            y_data: 目标y数据
            debug_mode: 是否显示详细调试信息
            n_steps: 推理步数
            input_dim: 输入维度，如果为None则自动推断
            max_expr_length: 表达式最大token长度，如果为None则使用args中的值
        """
        print("开始符号回归推理...")
        print(f"输入数据: x形状={x_data.shape}, y形状={y_data.shape}")

        # 加载模型
        checkpoint_path = model_path if model_path and os.path.exists(model_path) else None
        if checkpoint_path:
            print(f"使用检查点: {checkpoint_path}")
        else:
            print("未找到检查点，将使用基础模型进行推理")

        model, condition_encoder, criterion, optimizer, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

        # 设置设备和模式
        device = self.device
        model.eval()
        condition_encoder.eval()

        # 准备输入数据
        x_values = torch.FloatTensor(x_data).unsqueeze(0).to(device)
        y_values = torch.FloatTensor(y_data).unsqueeze(0).to(device)

        # 推断输入维度并生成初始表达式
        if input_dim is None:
            input_dim = x_data.shape[1] if len(x_data.shape) > 1 else 1

        # 修正：计算初始残差 (真实值 - 初始表达式的预测值)
        # 构建初始表达式
        import sympy as sp
        from ..symbolic.symbolic_utils import evaluate_expression_safe

        if input_dim == 1:
            # 一维情况：初始表达式为 x0
            initial_expr = sp.Symbol('x0')
        else:
            # 多维情况：初始表达式为 x0+x1+x2+...
            initial_expr = sum(sp.Symbol(f'x{i}') for i in range(input_dim))

        # 计算初始表达式在x_data上的预测值
        success, y_pred = evaluate_expression_safe(initial_expr, x_data)
        if not success:
            print(f"警告：无法计算初始表达式 '{initial_expr}' 的预测值，使用零残差")
            residuals = y_values
        else:
            # 计算残差：真实值 - 预测值
            residuals = y_values - torch.FloatTensor(y_pred).unsqueeze(0).to(device)

        condition = condition_encoder(x_values, residuals)

        if input_dim == 1:
            current_tokens = ['x0']
        else:
            current_tokens = []
            for i in range(input_dim):
                if i > 0:
                    current_tokens.append('+')
                current_tokens.append(f'x{i}')

        # 初始化token管理器，确保覆盖数据维度
        actual_max_dim = max(input_dim, self.args.max_dim) if hasattr(self.args, 'max_dim') else input_dim
        special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=actual_max_dim)

        # 确保所有需要的符号都在tokenizer中
        special_tokens_manager.ensure_special_tokens()

        if debug_mode:
            print(f"调试模式: {'开启' if debug_mode else '关闭'}")
            print(f"推理步数: {n_steps}")
            print(f"输入数据形状: x_values={x_values.shape}, y_values={y_values.shape}")
            print(f"条件嵌入形状: {condition.shape}")
            print(f"初始表达式: {','.join(current_tokens)}")

        for step in range(n_steps):
            if not debug_mode:
                print(f"推理步骤 {step + 1}/{n_steps}, 当前表达式: {','.join(current_tokens) if current_tokens else '<blank>'}")

            if debug_mode:
                print(f"\n[DEBUG] === 推理步骤 {step + 1}/{n_steps} ===")
                print(f"[DEBUG] 当前表达式: {','.join(current_tokens) if current_tokens else '<空白>'}")

            tokenized_expr = special_tokens_manager.tokenize_expression(','.join(current_tokens))

            max_len = getattr(self.args, 'max_expr_length', 128)
            if len(tokenized_expr) > max_len - 1:
                tokenized_expr = tokenized_expr[:max_len-1]
                if debug_mode:
                    print(f"[DEBUG] 表达式过长，截断至 {max_len-1} 个token")

            # 使用统一的特殊token管理
            bos_token = special_tokens_manager.get_token_id('bos')
            pad_token = special_tokens_manager.get_token_id('pad')

            tokenized_expr = [bos_token] + tokenized_expr
            tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

            input_ids = torch.LongTensor([tokenized_expr]).to(device)
            # 修正：基于实际token内容构建掩码，而不是位置假设
            attention_mask = (input_ids != pad_token).float().to(device)

            t = torch.tensor([[0.1 + 0.9 * step / n_steps]], dtype=torch.float32, device=device)

            if debug_mode:
                print(f"[DEBUG] 时间步t: {t[0,0]:.4f}")
                print(f"[DEBUG] tokenized_expr长度: {len(tokenized_expr)}")
                print(f"[DEBUG] 有效token数量: {attention_mask[0].sum().item()}")

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

                if debug_mode:
                    print(f"[DEBUG] 操作强度形状: lambda_ins={lambda_ins.shape}, lambda_sub={lambda_sub.shape}, lambda_del={lambda_del.shape}")
                    print(f"[DEBUG] 基础长度: {base_length}, 有效长度: {effective_length}")

                    # 显示前几个位置的操作强度
                    print(f"[DEBUG] 前5个位置的操作强度:")
                    for i in range(1, min(6, effective_length)):
                        print(f"[DEBUG]   位置{i}: INS={lambda_ins[i]:.4f}, SUB={lambda_sub[i]:.4f}, DEL={lambda_del[i]:.4f}")

                best_pos = 0
                best_action = None
                best_score = -1

                # 寻找最佳操作
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

                if debug_mode:
                    print(f"[DEBUG] 最佳操作: {best_action}, 分数: {best_score:.4f}")

                if best_action and best_score > 0.01:
                    action_type, pos = best_action

                    if debug_mode:
                        print(f"[DEBUG] 执行操作: {action_type.upper()} 位置{pos}")

                    if action_type == 'insert':
                        best_token = torch.argmax(insert_probs[0, pos]).item()
                        if debug_mode:
                            print(f"[DEBUG] 选择插入的token ID: {best_token}")

                        # 直接使用tokenizer转换token ID为token名称
                        best_token_name = tokenizer.convert_ids_to_tokens([best_token])[0]
                        current_tokens.insert(pos, best_token_name)
                        if debug_mode:
                            print(f"[DEBUG] 成功插入token: '{best_token_name}'")

                    elif action_type == 'substitute' and pos < len(current_tokens):
                        best_token = torch.argmax(substitute_probs[0, pos]).item()
                        if debug_mode:
                            print(f"[DEBUG] 选择替换的token ID: {best_token}")
                            print(f"[DEBUG] 替换前: '{current_tokens[pos]}'")

                        # 直接使用tokenizer转换token ID为token名称
                        best_token_name = tokenizer.convert_ids_to_tokens([best_token])[0]
                        current_tokens[pos] = best_token_name
                        if debug_mode:
                            print(f"[DEBUG] 成功替换为: '{best_token_name}'")

                    elif action_type == 'delete' and pos < len(current_tokens):
                        deleted_token = current_tokens[pos]
                        del current_tokens[pos]
                        if debug_mode:
                            print(f"[DEBUG] 删除token: '{deleted_token}'")
                else:
                    if debug_mode:
                        print(f"[DEBUG] 未找到有效操作 (最高分数: {best_score:.4f} <= 0.01)")

            if len(current_tokens) > 50:
                old_len = len(current_tokens)
                current_tokens = current_tokens[:50]
                if debug_mode:
                    print(f"[DEBUG] 表达式过长，从{old_len}截断至50个token")

        # 返回最终的表达式
        final_expression = ','.join(current_tokens) if current_tokens else ""
        if debug_mode:
            print(f"\n[DEBUG] 推理完成！")
            print(f"[DEBUG] 最终表达式: {final_expression}")
            print(f"[DEBUG] 最终表达式长度: {len(current_tokens)}")
        else:
            print(f"最终表达式: {final_expression}")

        return final_expression

    
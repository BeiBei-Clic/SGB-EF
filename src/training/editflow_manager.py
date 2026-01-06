"""
EditFlow迭代优化训练器 - 实现基于迭代式编辑操作的符号回归模型训练
使用 Hugging Face Accelerate 进行分布式训练加速
"""

import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from ..symbolic.data_generator import generate_flow_samples
from .flow import (
    remove_gap_tokens, fill_gap_tokens_with_repeats,
    ContinuousFlowLoss, FlowDataset, custom_collate_fn
)
from ..modeling.condition_encoder import SetTransformerConditionEncoder
from ..modeling.llama_editflow import LlamaEditFlowBackbone
from ..utils.misc_utils import find_latest_checkpoint, load_checkpoint
from ..utils.logger import Logger
from .greedy_search import SimpleSymbolicRegression


class EditFlowManager:
    """EditFlow模型管理器 - 支持训练和推理功能

    架构特点：迭代优化模式
    - 模型直接预测从z0到z1的编辑操作（插入、删除、替换）
    - 时间步固定为0，学习从起点到目标的直接编辑路径
    - 使用目标值y_target作为条件（而非残差），保持条件恒定作为"北极星"
    """

    # 类常量：训练和推理配置参数
    GRADIENT_CLIP_NORM = 10.0
    NUMERICAL_CLIP_THRESHOLD = 1e6
    MAX_EXPRESSION_LENGTH = 50
    MIN_ACTION_SCORE = 0.01

    def __init__(self, args):
        self.args = args

        # 初始化 Accelerate - 自动处理分布式训练设置
        # 注意：mixed_precision 由 accelerate launch 命令行参数控制
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.log_with
        )

        set_seed(args.seed)
        self.debug_mode = args.debug
        self.logger = Logger(self.accelerator, enabled=True, debug_mode=self.debug_mode)
        self.device = self.accelerator.device

        if self.accelerator.is_local_main_process:
            print("=== EditFlow符号回归预训练 (使用 Accelerate 加速) ===")
            print(f"样本数: {self.args.num_samples}")
            print(f"最大维度: {self.args.max_dim}")
            print(f"表达式最大长度: {self.args.max_expr_length}")
            print(f"批次大小: {self.args.batch_size}")
            print(f"训练轮数: {self.args.num_epochs}")
            print(f"学习率: {self.args.learning_rate}")
            print(f"测试集比例: {self.args.test_split}")
            print(f"评估频率: 每{self.args.eval_every}轮")
            print(f"LLaMA模型配置: hidden_dim={self.args.hidden_dim}, n_layers={self.args.n_layers}, n_heads={self.args.n_heads}")
            print(f"条件嵌入模型: {self.args.condition_model_name}")
            print(f"梯度累积步数: {self.args.gradient_accumulation_steps}")
            print(f"FP16混合精度: {self.args.use_fp16}")
            print(f"\nAccelerate 初始化完成")
            print(f"  设备: {self.device}")
            print(f"  分布式训练: {self.accelerator.distributed_type}")
            print(f"  进程数: {self.accelerator.num_processes}")
            print(f"  混合精度: {self.accelerator.mixed_precision}")
            print(f"  调试模式: {'启用' if self.debug_mode else '禁用'}")

        self.logger.training_start(self.args)

    def _gather_average_loss(self, total_loss, num_batches, default_value=0.0):
        """跨进程收集并计算平均损失"""
        self.accelerator.wait_for_everyone()
        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        num_batches_tensor = torch.tensor(num_batches, device=self.device)
        gathered_losses = self.accelerator.gather(total_loss_tensor)
        gathered_batches = self.accelerator.gather(num_batches_tensor)
        total_batches = gathered_batches.sum().item()
        return gathered_losses.sum().item() / total_batches if total_batches > 0 else default_value

    def _gather_metrics_across_processes(self, num_batches, total_loss, total_grad_norm):
        """跨进程收集训练指标（批次数、总损失、总梯度范数）

        Args:
            num_batches: 当前进程的批次数
            total_loss: 当前进程的总损失
            total_grad_norm: 当前进程的总梯度范数

        Returns:
            tuple: (gathered_batches, gathered_total_losses, gathered_total_grad_norms)
        """
        if self.accelerator.num_processes > 1:
            gathered_batches = self.accelerator.gather(
                torch.tensor(num_batches, device=self.device)
            )
            gathered_total_losses = self.accelerator.gather(
                torch.tensor(total_loss, device=self.device)
            )
            gathered_total_grad_norms = self.accelerator.gather(
                torch.tensor(total_grad_norm, device=self.device)
            )
        else:
            gathered_batches = torch.tensor([num_batches], device=self.device)
            gathered_total_losses = torch.tensor([total_loss], device=self.device)
            gathered_total_grad_norms = torch.tensor([total_grad_norm], device=self.device)

        return gathered_batches, gathered_total_losses, gathered_total_grad_norms

    def _format_gpu_metrics_summary(self, gathered_batches, gathered_total_losses,
                                    gathered_total_grad_norms, include_lr=False, lr=None):
        """格式化GPU指标统计摘要

        Args:
            gathered_batches: 各进程的批次数
            gathered_total_losses: 各进程的总损失
            gathered_total_grad_norms: 各进程的总梯度范数
            include_lr: 是否包含学习率信息
            lr: 学习率值（当include_lr=True时使用）

        Returns:
            tuple: (gpu_metrics_list, global_total_batches, global_avg_loss, global_avg_grad_norm)
        """
        gpu_metrics = []
        global_total_batches = 0
        global_total_loss = 0.0
        global_total_grad_norm = 0.0

        for gpu_idx in range(self.accelerator.num_processes):
            gpu_batches = gathered_batches[gpu_idx].item()
            gpu_total_loss = gathered_total_losses[gpu_idx].item()
            gpu_total_grad_norm = gathered_total_grad_norms[gpu_idx].item()

            gpu_avg_loss = gpu_total_loss / gpu_batches if gpu_batches > 0 else 0.0
            gpu_avg_grad_norm = gpu_total_grad_norm / gpu_batches if gpu_batches > 0 else 0.0

            gpu_metrics.append(
                f"  [GPU {gpu_idx}] batches={gpu_batches} | "
                f"total_loss={gpu_total_loss:.2f} | avg_loss={gpu_avg_loss:.6f} | "
                f"avg_grad_norm={gpu_avg_grad_norm:.3f}"
            )

            global_total_batches += gpu_batches
            global_total_loss += gpu_total_loss
            global_total_grad_norm += gpu_total_grad_norm

        global_avg_loss = global_total_loss / global_total_batches if global_total_batches > 0 else 0.0
        global_avg_grad_norm = global_total_grad_norm / self.accelerator.num_processes if self.accelerator.num_processes > 0 else 0.0

        return gpu_metrics, global_total_batches, global_avg_loss, global_avg_grad_norm


    def prepare_data(self, tokenizer):
        """准备训练数据，使用 Hugging Face datasets 加载"""
        cache_filename = f"data/flow_samples_{self.args.num_samples}_{self.args.max_dim}dim_{self.args.n_points}pts_{self.args.max_depth}depth_{self.args.max_expr_length}len.parquet"

        # 主进程负责数据生成
        if self.accelerator.is_local_main_process:
            print(f"准备连续流训练数据 (单进程生成模式)...")
            print(f"使用对齐方法: {self.args.alignment_method}")
            generate_flow_samples(
                num_samples=self.args.num_samples,
                max_dim=self.args.max_dim,
                n_points=self.args.n_points,
                max_depth=self.args.max_depth,
                max_expr_length=self.args.max_expr_length,
                verbose=True,
                alignment_method=self.args.alignment_method,
            )
        else:
            print(f"[Rank {self.accelerator.process_index}] 跳过数据生成，等待主进程完成...")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            print("[主进程] 数据生成完成，开始加载训练数据")

        print(f"[Rank {self.accelerator.process_index}] 准备开始训练阶段...")
        self.accelerator.wait_for_everyone()

        # 加载数据
        use_stream = self.args.dataset_stream
        num_proc = self.args.dataset_num_proc

        if self.accelerator.is_local_main_process:
            print(f"使用 Hugging Face datasets 加载数据 (stream={use_stream})...")

        # 分割训练集和测试集
        train_dataset, test_dataset, train_size_estimate, test_size_estimate = self._split_train_test(
            cache_filename, tokenizer, use_stream, num_proc
        )

        # 创建DataLoader
        train_dataloader, test_dataloader = self._create_dataloaders(
            train_dataset, test_dataset
        )

        # 准备分布式训练
        if self.accelerator.is_local_main_process:
            print(f"正在准备分布式训练 (accelerator.prepare)...")

        train_dataloader, test_dataloader = self.accelerator.prepare(
            train_dataloader, test_dataloader
        )

        if self.accelerator.is_local_main_process:
            is_stream_mode = getattr(train_dataset, 'stream', False)
            train_shuffle = not is_stream_mode
            num_workers = 0 if is_stream_mode else self.accelerator.num_processes
            expected_train_batches = train_size_estimate // self.args.batch_size
            expected_test_batches = test_size_estimate // self.args.batch_size

            print(f"✓ 分布式训练准备完成")
            print(f"数据准备完成: 训练集约 {train_size_estimate} 样本, 测试集约 {test_size_estimate} 样本")

            self.logger.log(
                "DATALOADER_VERIFY",
                f"DataLoader创建完成 | 预期训练批次数={expected_train_batches} | "
                f"预期测试批次数={expected_test_batches} | "
                f"num_workers={num_workers} | is_stream_mode={is_stream_mode} | "
                f"train_shuffle={train_shuffle} | "
                f"支持set_epoch={hasattr(train_dataset, 'set_epoch')}",
                "data_loading",
                level=1
            )

        return train_dataloader, train_dataset, test_dataloader, test_dataset

    def _split_train_test(self, cache_filename, tokenizer, use_stream, num_proc):
        """分割训练集和测试集"""
        # 当样本数很少时，让所有样本同时用于训练和测试
        if self.args.num_samples <= self.args.batch_size:
            return self._create_full_datasets(cache_filename, tokenizer, use_stream, num_proc)

        # 正常分割逻辑
        if use_stream:
            return self._split_stream_mode(cache_filename, tokenizer, num_proc)
        else:
            return self._split_nonstream_mode(cache_filename, tokenizer, num_proc)

    def _create_full_datasets(self, cache_filename, tokenizer, use_stream, num_proc):
        """样本数很少时，创建完整的训练集和测试集"""
        if self.accelerator.is_local_main_process:
            mode_str = "流式" if use_stream else "非流式"
            print(f"{mode_str}模式: 样本数({self.args.num_samples}) ≤ batch_size({self.args.batch_size})")
            print(f"        所有样本将同时用于训练和测试")

        full_dataset = FlowDataset(
            data_file=cache_filename, tokenizer=tokenizer,
            max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length,
            stream=use_stream, num_proc=num_proc, logger=self.logger
        )
        return full_dataset, full_dataset, self.args.num_samples, self.args.num_samples

    def _split_stream_mode(self, cache_filename, tokenizer, num_proc):
        """流式模式下的数据分割"""
        split_ratio = 1 - self.args.test_split
        train_size = int(self.args.num_samples * split_ratio)
        test_size = self.args.num_samples - train_size

        if self.accelerator.is_local_main_process:
            print(f"流式模式: 训练集约 {train_size} 样本, 测试集约 {test_size} 样本")

        train_dataset = FlowDataset(
            data_file=cache_filename, tokenizer=tokenizer,
            max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length,
            stream=True, num_proc=num_proc, logger=self.logger,
            skip=0, take=train_size
        )
        test_dataset = FlowDataset(
            data_file=cache_filename, tokenizer=tokenizer,
            max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length,
            stream=True, num_proc=num_proc, logger=self.logger,
            skip=train_size, take=test_size
        )
        return train_dataset, test_dataset, train_size, test_size

    def _split_nonstream_mode(self, cache_filename, tokenizer, num_proc):
        """非流式模式下的数据分割"""
        full_dataset = FlowDataset(
            data_file=cache_filename, tokenizer=tokenizer,
            max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length,
            stream=False, num_proc=num_proc, logger=self.logger
        )

        total_size = len(full_dataset)
        train_size = int(total_size * (1 - self.args.test_split))

        from torch.utils.data import Subset
        indices = list(range(total_size))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        if self.accelerator.is_local_main_process:
            print(f"非流式模式: 训练集 {len(train_indices)} 样本, 测试集 {len(test_indices)} 样本")

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)
        return train_dataset, test_dataset, len(train_indices), len(test_indices)

    def _create_dataloaders(self, train_dataset, test_dataset):
        """创建训练和测试DataLoader"""
        is_stream_mode = getattr(train_dataset, 'stream', False)
        train_size = len(train_dataset)
        test_size = len(test_dataset)
        train_drop_last = train_size >= self.args.batch_size
        test_drop_last = test_size >= self.args.batch_size

        if self.accelerator.is_local_main_process:
            if not train_drop_last:
                print(f"警告: 训练集大小({train_size}) < batch_size({self.args.batch_size})，禁用drop_last")
            if not test_drop_last:
                print(f"警告: 测试集大小({test_size}) < batch_size({self.args.batch_size})，禁用drop_last")

        train_shuffle = not is_stream_mode
        num_workers = 0 if is_stream_mode else self.accelerator.num_processes

        if self.accelerator.is_local_main_process:
            print(f"正在创建 DataLoader (batch_size={self.args.batch_size}, num_workers={num_workers}, shuffle={train_shuffle})...")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=train_shuffle,
            num_workers=num_workers, collate_fn=custom_collate_fn,
            drop_last=train_drop_last, pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=test_drop_last
        )

        if self.accelerator.is_local_main_process:
            print(f"✓ DataLoader 创建完成")

        return train_dataloader, test_dataloader

    def setup_models(self, checkpoint_path=None):
        """初始化模型和tokenizer，支持从检查点加载"""
        if self.accelerator.is_local_main_process:
            print("初始化tokenizer和模型...")

        # 初始化tokenizer
        from ..utils.special_tokens import SymbolicRegressionTokenizer, SymbolicVocab
        tokenizer = SymbolicRegressionTokenizer(max_dim=self.args.max_dim)

        if self.accelerator.is_local_main_process:
            print(f"✓ 符号回归Tokenizer初始化完成")
            print(f"  词汇表大小: {tokenizer.vocab_size} (符号回归专属小词汇表)")
            print(f"  最大维度: {self.args.max_dim}")
            print(f"  运算符: {len(SymbolicVocab.OPERATORS)}个 - {', '.join(SymbolicVocab.OPERATORS)}")
            print(f"  函数: {len(SymbolicVocab.FUNCTIONS)}个 - {', '.join(SymbolicVocab.FUNCTIONS)}")
            print(f"  特殊token: {len(SymbolicVocab.SPECIAL_TOKENS)}个")
            print(f"  变量token: x0 ~ x{self.args.max_dim-1} (共{self.args.max_dim}个)")

        # 初始化条件编码器
        if self.accelerator.is_local_main_process:
            print("初始化条件编码器...")
        condition_encoder = SetTransformerConditionEncoder(
            max_input_dim=self.args.condition_max_input_dim,
            dim_hidden=self.args.condition_dim_hidden,
            num_heads=self.args.condition_num_heads,
            num_inds=self.args.condition_num_inds,
            num_layers=self.args.condition_num_layers,
            num_seeds=self.args.condition_num_seeds,
            dim_output=self.args.condition_dim_output,
            verbose=self.accelerator.is_local_main_process
        ).to(self.device)

        # 初始化LLaMA EditFlow模型
        if self.accelerator.is_local_main_process:
            print("初始化LLaMA EditFlow模型（自定义架构，不加载预训练权重）...")

        model = LlamaEditFlowBackbone(
            vocab_size=len(tokenizer.get_vocab()),
            hidden_dim=self.args.hidden_dim,
            n_layers=self.args.n_layers,
            n_heads=self.args.n_heads,
            condition_dim=self.args.condition_dim_hidden,
            dropout=self.args.dropout,
            max_seq_len=self.args.max_expr_length,
            use_condition_injection=self.args.use_condition_injection,
            verbose=self.accelerator.is_local_main_process
        ).to(self.device)

        # 创建优化器和损失函数
        criterion = ContinuousFlowLoss(debug_mode=self.debug_mode)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # 添加学习率调度器（余弦退火）
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs, eta_min=1e-6)

        # 加载检查点
        load_checkpoint(checkpoint_path, model, condition_encoder, self.device, optimizer, verbose=self.accelerator.is_local_main_process)

        # 使用 Accelerate 准备模型和优化器
        if self.accelerator.is_local_main_process:
            print(f"使用 Accelerate 准备模型和优化器...")
            print(f"  进程数: {self.accelerator.num_processes}")
            print(f"  设备: {self.accelerator.device}")
            print(f"  混合精度: {self.accelerator.mixed_precision}")

        model, condition_encoder, optimizer = self.accelerator.prepare(model, condition_encoder, optimizer)

        # 如果有checkpoint，使用Accelerate的load_state方法加载完整状态
        if checkpoint_path:
            if self.accelerator.is_local_main_process:
                print(f"Loading complete training state from {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.accelerator.is_local_main_process:
            print(f"✓ LLaMA EditFlow模型参数数量: {total_params:,}")

        self.tokenizer = tokenizer

        return model, condition_encoder, criterion, optimizer, scheduler, tokenizer

  
    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, debug_info=None):
        """前向传播：直接预测从z0到z1的编辑操作"""
        batch_size = z0_token_ids.size(0)
        vocab_size = self.tokenizer.vocab_size
        batch_size, seq_len = z0_token_ids.shape

        # 移除gap token得到输入序列x_t（原始序列空间，无gap重复）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(z0_token_ids, self.tokenizer)
        attention_mask = (~x_pad_mask).float()

        # 调用模型
        output = model(input_ids=x_t, condition=condition_embeddings, attention_mask=attention_mask)
        pred_rates = output['rates_logits']

        # 记录debug信息（使用Logger的通用函数）
        if debug_info:
            debug_info['sample_idx'] = 0
            self.logger.log_forward_debug(
                debug_info,
                self.tokenizer,
                z0_token_ids=z0_token_ids,
                z1_token_ids=z1_token_ids,
                condition_embeddings=condition_embeddings,
                x_t=x_t,
                attention_mask=attention_mask,
                pred_rates=pred_rates,
                insert_logits=output['insert_logits'],
                substitute_logits=output['substitute_logits']
            )

        return {
            'pred_rates': pred_rates,
            'pred_ins_logits': output['insert_logits'],
            'pred_sub_logits': output['substitute_logits'],
            'x_t': x_t,
            'z0': z0_token_ids,
            'z1_token_ids': z1_token_ids,
            'z_gap_mask': z_gap_mask,
            'z_pad_mask': z_pad_mask,
            'attention_mask': attention_mask,
            'vocab_size': vocab_size,
        }

    def compute_loss(self, forward_results, criterion, debug_info=None):
        pred_rates = forward_results['pred_rates']
        x_t = forward_results['x_t']
        z0 = forward_results['z0']
        z1_token_ids = forward_results['z1_token_ids']
        z_gap_mask = forward_results['z_gap_mask']
        z_pad_mask = forward_results['z_pad_mask']
        effective_vocab_size = forward_results['vocab_size']
        gap_token = self.tokenizer.convert_tokens_to_ids('<gap>')

        # 拆分操作logits：ins, del, sub, keep
        ins_logits_rate = pred_rates[:, :, 0:1]
        del_logits_rate = pred_rates[:, :, 1:2]
        sub_logits_rate = pred_rates[:, :, 2:3]
        keep_logits_rate = pred_rates[:, :, 3:4]

        # 在logit空间相加
        ins_logits = forward_results['pred_ins_logits'] + ins_logits_rate
        sub_logits = forward_results['pred_sub_logits'] + sub_logits_rate
        del_logits = del_logits_rate
        keep_logits = keep_logits_rate

        # 拼接所有操作logits：ins | del | sub | keep
        u_cat_x = torch.cat([ins_logits, del_logits, sub_logits, keep_logits], dim=-1)

        # 将X空间的预测扩展到Z空间
        u_z = fill_gap_tokens_with_repeats(u_cat_x, z_gap_mask, z_pad_mask)

        # 生成编辑操作掩码
        u_mask_x = criterion.make_ut_mask_from_z(z0, z1_token_ids, effective_vocab_size, gap_token, self.tokenizer, x_t)
        u_mask = fill_gap_tokens_with_repeats(u_mask_x, z_gap_mask, z_pad_mask)

        # 记录debug信息（使用Logger的方法）
        if self.accelerator.is_local_main_process and self.debug_mode:
            self.logger.log_compute_loss_debug(debug_info, z0, z1_token_ids, x_t, pred_rates, u_cat_x, u_mask_x, effective_vocab_size, self.tokenizer)

        loss = criterion(u_cat_x, u_z, u_mask, effective_vocab_size,
                        accelerator=self.accelerator, logger=self.logger)

        return loss

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        # 关键修复：对于流式数据集，在每个 epoch 开始时调用 set_epoch
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)
            if self.accelerator.is_local_main_process:
                self.logger.log(
                    "DATASET_SET_EPOCH",
                    f"调用 dataset.set_epoch({epoch}) | 迭代器已重置，数据已重新洗牌",
                    f"epoch{epoch+1}_data_reset",
                    level=1
                )

        total_loss = 0.0
        num_batches = 0
        local_total_grad_norm = 0.0

        # 计算数据集信息
        dataset_size = len(dataset)
        num_batches_estimate = dataset_size // self.args.batch_size

        # 记录epoch开始和数据集信息
        if self.accelerator.is_local_main_process:
            self.logger.log(
                "EPOCH_START",
                f"开始 Epoch {epoch+1}/{self.args.num_epochs} | 维度={dimension} | "
                f"数据集大小={dataset_size} | 预计批次数={num_batches_estimate} | "
                f"批次大小={self.args.batch_size}",
                f"epoch{epoch+1}_dim{dimension}",
                level=1
            )

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}",
                          disable=not self.accelerator.is_local_main_process)

        if self.accelerator.is_local_main_process:
            progress_bar.set_postfix({'loss': '0.0000', 'grad_norm': '0.000'})

        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()

            # 记录数据加载进度（不受debug控制，始终记录）
            if self.accelerator.is_local_main_process:
                progress_pct = (batch_idx + 1) / num_batches_estimate * 100 if num_batches_estimate > 0 else 0
                self.logger.log(
                    "BATCH_LOAD_START",
                    f"开始加载 Batch {batch_idx+1}/{num_batches_estimate} | "
                    f"进度={progress_pct:.1f}% | timestamp={time.time():.2f}",
                    f"epoch{epoch+1}_dim{dimension}_batch{batch_idx}",
                    level=1
                )

            loss, grad_norm = self._process_batch(
                model, condition_encoder, criterion, optimizer, batch, batch_idx, epoch, dimension
            )

            total_loss += loss
            num_batches += 1
            local_total_grad_norm += grad_norm

            batch_total_time = time.time() - batch_start_time

            # 更新进度条
            if self.accelerator.is_local_main_process:
                postfix_dict = {
                    'loss': f'{loss:.4f}',
                    'grad_norm': f'{grad_norm:.3f}',
                    'time': f'{batch_total_time:.2f}s' if self.debug_mode else ''
                }
                progress_bar.set_postfix(postfix_dict)

            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("BATCH_COMPLETE", f"Batch {batch_idx} 完成 | 总耗时={batch_total_time:.3f}s | timestamp={time.time():.2f}",
                                f"维度{dimension}_batch{batch_idx}", level=2)

        # 跨进程收集并计算平均损失
        avg_loss = self._gather_average_loss(total_loss, num_batches, default_value=0.0)

        # 在所有进程上收集训练指标
        num_processes = self.accelerator.num_processes
        gathered_batches, gathered_total_losses, gathered_total_grad_norms = \
            self._gather_metrics_across_processes(num_batches, total_loss, local_total_grad_norm)

        # 数据消耗监控：记录实际处理的 batch 数（只在主进程）
        if self.accelerator.is_local_main_process:
            expected_batches = dataset_size // self.args.batch_size
            actual_batches = num_batches
            total_batches_all_processes = gathered_batches.sum().item()

            # 计算样本覆盖率
            total_samples_processed = total_batches_all_processes * self.args.batch_size
            coverage_rate = (total_samples_processed / dataset_size * 100) if dataset_size > 0 else 0.0

            # 根据是否分布式训练，显示不同的日志格式
            if num_processes > 1:
                # 使用辅助方法格式化GPU指标（返回的total_batches_all_processes应该与之前计算的一致）
                gpu_metrics, _, global_avg_loss, global_avg_grad_norm = \
                    self._format_gpu_metrics_summary(gathered_batches, gathered_total_losses, gathered_total_grad_norms)

                # 构建完整的日志消息
                gpu_metrics_summary = "\n" + "\n".join(gpu_metrics)
                data_allocation_summary = (
                    f"\n--- 数据分配 --- | 进程数={num_processes} | 数据集大小={dataset_size} | "
                    f"批次大小={self.args.batch_size} | 预期单进程批次数={expected_batches} | "
                    f"覆盖率={coverage_rate:.1f}%"
                )
                global_summary = (
                    f"\n--- 全局汇总 --- | 总批次数={total_batches_all_processes} | "
                    f"avg_loss={global_avg_loss:.6f} | avg_grad_norm={global_avg_grad_norm:.3f}"
                )

                self.logger.log(
                    "EPOCH_BATCH_COUNT",
                    f"Epoch {epoch+1} 完成 [分布式训练详细] |" +
                    gpu_metrics_summary +
                    data_allocation_summary +
                    global_summary,
                    f"epoch{epoch+1}_dim{dimension}_detailed",
                    level=1
                )
            else:
                # 单GPU训练
                avg_grad_norm = local_total_grad_norm / num_batches if num_batches > 0 else 0.0
                self.logger.log(
                    "EPOCH_BATCH_COUNT",
                    f"Epoch {epoch+1} 完成 | 预期批次数={expected_batches} | "
                    f"实际批次数={actual_batches} | 总损失={total_loss:.2f} | "
                    f"平均损失={avg_loss:.6f} | 平均梯度范数={avg_grad_norm:.3f} | "
                    f"数据集大小={dataset_size} | 批次大小={self.args.batch_size}",
                    f"epoch{epoch+1}_dim{dimension}_monitor",
                    level=1
                )

            # 警告：如果实际批次数远少于预期，可能是数据加载问题
            if epoch > 0 and actual_batches == 0:
                self.logger.error(
                    "NO_DATA_LOADED",
                    f"严重错误：Epoch {epoch+1} 没有处理任何 batch！"
                    f"这通常意味着 IterableDataset 迭代器已耗尽且未正确重置。",
                    f"epoch{epoch+1}_critical"
                )
            elif epoch > 0 and actual_batches < expected_batches * 0.5:
                self.logger.error(
                    "INSUFFICIENT_DATA",
                    f"警告：Epoch {epoch+1} 实际批次数({actual_batches}) "
                    f"远少于预期({expected_batches})，可能存在数据加载问题。",
                    f"epoch{epoch+1}_warning"
                )

        # 返回平均损失、批次数、总损失和总梯度范数（用于GPU级别信息汇总）
        return avg_loss, num_batches, total_loss, local_total_grad_norm

    def _process_batch(self, model, condition_encoder, criterion, optimizer, batch, batch_idx, epoch, dimension):
        """处理单个训练batch"""
        if self.accelerator.is_local_main_process and self.debug_mode:
            self.logger.log("BATCH_START", f"开始处理 Batch {batch_idx} | timestamp={time.time():.2f}",
                            f"维度{dimension}_batch{batch_idx}", level=2)

        with self.accelerator.accumulate([model, condition_encoder]):
            data_load_start = time.time()
            x_values = batch['x_values'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            z0_token_ids = batch['z0_token_ids'].to(self.device)
            z1_token_ids = batch['z1_token_ids'].to(self.device)
            point_mask = batch['point_mask'].to(self.device) if 'point_mask' in batch else None

            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("DATA_LOAD", f"数据加载完成 | 耗时={time.time() - data_load_start:.3f}s",
                                f"维度{dimension}_batch{batch_idx}", level=2)

            # 编码条件
            condition_start = time.time()
            condition_embeddings = condition_encoder(x_values, y_target, point_mask)
            condition_time = time.time() - condition_start

            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("CONDITION_ENCODE", f"条件编码完成 | 耗时={condition_time:.3f}s",
                                f"维度{dimension}_batch{batch_idx}", level=2)
                context = f'维度{dimension}'
                self.logger.tensor_values(f"x_values_batch{batch_idx}", x_values[0],
                                         context=context, level=2, max_elements=50)
                self.logger.tensor_values(f"y_target_batch{batch_idx}", y_target[0],
                                         context=context, level=2, max_elements=50)

            debug_info = {'batch_idx': batch_idx, 'context': f'维度{dimension}'}

            # 前向传播
            forward_start = time.time()
            forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, debug_info)
            forward_time = time.time() - forward_start

            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("FORWARD_PASS", f"前向传播完成 | 耗时={forward_time:.3f}s",
                                f"维度{dimension}_batch{batch_idx}", level=2)

            # NaN检查
            nan_check_start = time.time()
            if self.accelerator.distributed_type != "NO":
                pred_rates = forward_results['pred_rates']
                local_has_nan = torch.isnan(pred_rates).any().float()
                gathered_nan_results = self.accelerator.gather(local_has_nan)
                global_has_nan = gathered_nan_results.sum()

                if global_has_nan.item() > 0:
                    self.logger.error("FORWARD_NAN", f"维度{dimension} 检测到前向传播NaN", f"batch_idx:{batch_idx}")

            nan_check_time = time.time() - nan_check_start

            # 计算损失
            loss_compute_start = time.time()
            loss = self.compute_loss(forward_results, criterion, debug_info)
            loss_compute_time = time.time() - loss_compute_start

            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("LOSS_COMPUTED", f"loss={loss.item():.6f} | 耗时={loss_compute_time:.3f}s | NaN检查耗时={nan_check_time:.3f}s",
                                f"维度{dimension}_batch{batch_idx}", level=2)

            # 反向传播
            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("BACKWARD_START", f"开始反向传播 | loss={loss.item():.6f} | timestamp={time.time():.2f}",
                                f"维度{dimension}_batch{batch_idx}", level=2)

            backward_start = time.time()
            try:
                self.accelerator.backward(loss)
                backward_time = time.time() - backward_start
                if self.accelerator.is_local_main_process and self.debug_mode:
                    self.logger.log("BACKWARD_SUCCESS", f"反向传播成功 | 耗时={backward_time:.3f}s",
                                    f"维度{dimension}_batch{batch_idx}", level=2)
            except Exception as e:
                self.logger.log_crash(
                    step_name="BACKWARD", batch_idx=batch_idx, dimension=dimension,
                    error=e, extra_info=f"loss={loss.item():.6f}"
                )
                raise

            # 梯度裁剪和优化器更新
            all_params = list(model.parameters()) + list(condition_encoder.parameters())
            self.accelerator.clip_grad_norm_(all_params, self.GRADIENT_CLIP_NORM)

            grad_norm = 0.0
            for param in all_params:
                if param.grad is not None:
                    grad_norm += float(param.grad.data.norm().item() ** 2)
            grad_norm = float(grad_norm ** 0.5)

            optimizer.step()
            optimizer.zero_grad()

            return loss.item(), grad_norm

    def evaluate(self, model, condition_encoder, criterion, test_dataloader, test_dataset):
        """测试集评估"""
        model.eval()
        condition_encoder.eval()

        # 关键修复：对于流式测试集，也调用 set_epoch 重置迭代器
        if hasattr(test_dataset, 'set_epoch'):
            test_dataset.set_epoch(0)  # 测试集使用固定 epoch
            if self.accelerator.is_local_main_process:
                self.logger.log(
                    "TEST_DATASET_RESET",
                    "测试集迭代器已重置",
                    "evaluation",
                    level=1
                )

        total_loss = 0.0
        num_batches = 0

        # 计算测试集信息
        test_size = len(test_dataset)
        test_num_batches_estimate = test_size // self.args.batch_size

        # 记录测试开始
        if self.accelerator.is_local_main_process:
            self.logger.log(
                "EVAL_START",
                f"开始测试集评估 | 测试集大小={test_size} | "
                f"预计批次数={test_num_batches_estimate} | 批次大小={self.args.batch_size}",
                "evaluation",
                level=1
            )

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # 记录测试数据加载进度
                if self.accelerator.is_local_main_process:
                    progress_pct = (batch_idx + 1) / test_num_batches_estimate * 100 if test_num_batches_estimate > 0 else 0
                    self.logger.log(
                        "TEST_BATCH_LOAD",
                        f"测试 Batch {batch_idx+1}/{test_num_batches_estimate} | "
                        f"进度={progress_pct:.1f}% | timestamp={time.time():.2f}",
                        f"evaluation_batch{batch_idx}",
                        level=1
                    )
                x_values = batch['x_values'].to(self.device)
                y_target = batch['y_target'].to(self.device)
                z0_token_ids = batch['z0_token_ids'].to(self.device)
                z1_token_ids = batch['z1_token_ids'].to(self.device)
                point_mask = batch['point_mask'].to(self.device)

                condition_embeddings = condition_encoder(x_values, y_target, point_mask)
                forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids)
                loss = self.compute_loss(forward_results, criterion)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = self._gather_average_loss(total_loss, num_batches, default_value=float('inf'))

        return avg_loss


    def save_checkpoint(self, model, condition_encoder, loss, epoch, is_final=False):
        self.accelerator.wait_for_everyone()

        checkpoint_dir = os.path.join(
            self.args.save_dir,
            "continuous_flow_final" if is_final else f"checkpoint_epoch_{epoch+1}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 使用 Accelerate 的 save_state 方法
        self.accelerator.save_state(checkpoint_dir)

        # 保存模型配置信息
        if self.accelerator.is_local_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_encoder = self.accelerator.unwrap_model(condition_encoder)

            model_config = {
                'vocab_size': unwrapped_model.vocab_size,
                'hidden_dim': unwrapped_model.hidden_dim,
                'n_layers': unwrapped_model.n_layers,
                'n_heads': unwrapped_model.n_heads,
                'condition_dim': unwrapped_model.condition_dim,
                'dropout': unwrapped_model.dropout,
                'max_seq_len': unwrapped_model.max_seq_len,
                'use_condition_injection': unwrapped_model.use_condition_injection,
            }

            config_data = {
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'condition_encoder_state_dict': unwrapped_encoder.state_dict(),
                'loss': loss,
                'model_config': model_config,
                'args': self.args,
                'accelerate_config': {
                    'distributed_type': str(self.accelerator.distributed_type),
                    'num_processes': self.accelerator.num_processes,
                    'mixed_precision': str(self.accelerator.mixed_precision),
                }
            }

            config_path = os.path.join(checkpoint_dir, "training_config.json")
            torch.save(config_data, config_path)

        return checkpoint_dir

    def train(self):
        checkpoint_path = find_latest_checkpoint(self.args)
        if self.accelerator.is_local_main_process:
            print(f"使用设备: {self.device}")
            print(f"{'找到检查点' if checkpoint_path else '未找到检查点，将从基础模型开始训练'}: {checkpoint_path or ''}")

        model, condition_encoder, criterion, optimizer, scheduler, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)
        train_dataloader, train_dataset, test_dataloader, test_dataset = self.prepare_data(tokenizer)

        model_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad)
        if self.accelerator.is_local_main_process:
            print(f"模型参数数量: {model_params:,}, 条件编码器参数数量: {encoder_params:,}")
            print(f"开始连续流训练 ({self.args.num_epochs} epochs)...")
            self.logger.log("TRAINING_START", f"开始训练 | num_epochs={self.args.num_epochs} | model_params={model_params:,} | encoder_params={encoder_params:,}", level=1)

            # 分布式训练说明
            if self.accelerator.num_processes > 1:
                train_dataset_size = len(train_dataset)
                test_dataset_size = len(test_dataset)
                samples_per_process = train_dataset_size // self.accelerator.num_processes
                batches_per_process = samples_per_process // self.args.batch_size
                total_batches_all_processes = batches_per_process * self.accelerator.num_processes
                coverage_rate = (total_batches_all_processes * self.args.batch_size / train_dataset_size * 100) if train_dataset_size > 0 else 0.0

                print("\n" + "="*70)
                print("📊 分布式训练配置说明")
                print("="*70)
                print(f"进程数 (GPU数):        {self.accelerator.num_processes}")
                print(f"训练集总样本数:        {train_dataset_size}")
                print(f"每个进程分配样本数:    {samples_per_process} (整数除法)")
                print(f"每个进程预期批次数:    {batches_per_process}")
                print(f"所有进程总批次数:      {total_batches_all_processes}")
                print(f"样本覆盖率:            {coverage_rate:.1f}%")
                print(f"\n注意：由于整数除法，约 {train_dataset_size % self.accelerator.num_processes} 个样本")
                print(f"      ({train_dataset_size - total_batches_all_processes * self.args.batch_size} 个) 不会被训练")
                print("="*70 + "\n")

                self.logger.log(
                    "DISTRIBUTED_TRAINING_INFO",
                    f"分布式训练配置 | 进程数={self.accelerator.num_processes} | "
                    f"训练集大小={train_dataset_size} | 每进程样本数={samples_per_process} | "
                    f"每进程批次数={batches_per_process} | 总批次数={total_batches_all_processes} | "
                    f"覆盖率={coverage_rate:.1f}%",
                    "distributed_setup",
                    level=1
                )

        eval_every = self.args.eval_every

        for epoch in range(self.args.num_epochs):
            avg_loss, num_batches, total_loss, total_grad_norm = self.train_epoch(
                model, condition_encoder, criterion, optimizer,
                train_dataloader, train_dataset, epoch, "Mixed"
            )

            # 在所有进程上收集训练指标
            gathered_batches, gathered_total_losses, gathered_total_grad_norms = \
                self._gather_metrics_across_processes(num_batches, total_loss, total_grad_norm)

            # 只在主进程上打印和记录日志
            if self.accelerator.is_local_main_process:
                current_lr = optimizer.param_groups[0]['lr']

                if self.accelerator.num_processes > 1:
                    # 使用辅助方法格式化GPU指标
                    gpu_details, global_total_batches, global_avg_loss, _ = \
                        self._format_gpu_metrics_summary(gathered_batches, gathered_total_losses, gathered_total_grad_norms)

                    # 构建完整的日志消息
                    gpu_summary = "\n" + "\n".join(gpu_details) + "\n--- 全局汇总 --- | " + \
                                 f"total_batches={global_total_batches} | avg_train_loss={global_avg_loss:.6f} | " + \
                                 f"lr={current_lr:.2e}"

                    # 控制台输出（简化版）
                    print(f"Epoch {epoch+1}/{self.args.num_epochs} 完成 | avg_train_loss={global_avg_loss:.4f} | total_batches={global_total_batches} | lr={current_lr:.2e}")

                    # 日志文件记录（包含详细的GPU信息）
                    self.logger.log(
                        "EPOCH_COMPLETE",
                        f"Epoch {epoch+1}/{self.args.num_epochs} [分布式训练详细] |\n" + gpu_summary,
                        f"epoch{epoch+1}_complete",
                        level=1
                    )
                else:
                    # 单GPU训练
                    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
                    print(f"Epoch {epoch+1}/{self.args.num_epochs} 完成, 训练损失: {avg_loss:.4f}, 梯度范数: {avg_grad_norm:.3f}, 学习率: {current_lr:.2e}")
                    self.logger.log(
                        "EPOCH_COMPLETE",
                        f"Epoch {epoch+1}/{self.args.num_epochs} | train_loss={avg_loss:.4f} | "
                        f"avg_grad_norm={avg_grad_norm:.3f} | lr={current_lr:.2e} | batches={num_batches}",
                        level=1
                    )

            scheduler.step()

            if (epoch + 1) % eval_every == 0 or epoch == self.args.num_epochs - 1:
                test_loss = self.evaluate(model, condition_encoder, criterion, test_dataloader, test_dataset)
                if self.accelerator.is_local_main_process:
                    print(f"测试集损失: {test_loss:.4f}")
                    self.logger.log("EVALUATION", f"Epoch {epoch+1}/{self.args.num_epochs} | test_loss={test_loss:.4f}", level=1)

            if (epoch + 1) % self.args.save_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, avg_loss, epoch
                )
                if self.accelerator.is_local_main_process:
                    print(f"检查点已保存到: {checkpoint_path}")
                    self.logger.log("CHECKPOINT_SAVED", f"Epoch {epoch+1}/{self.args.num_epochs} | path={checkpoint_path} | train_loss={avg_loss:.4f}", level=1)

        # 保存最终模型
        final_path = self.save_checkpoint(
            model, condition_encoder, avg_loss, self.args.num_epochs - 1, is_final=True
        )
        if self.accelerator.is_local_main_process:
            print(f"最终模型已保存到: {final_path}")
            self.logger.log("TRAINING_COMPLETE", f"训练完成 | final_path={final_path} | final_train_loss={avg_loss:.4f} | total_epochs={self.args.num_epochs}", level=1)

        return model, condition_encoder

    # ============= 推理辅助方法 =============
    def _load_inference_model(self, model_path):
        """加载推理模型并处理检查点警告"""
        checkpoint_path = model_path if model_path and os.path.exists(model_path) else None

        if checkpoint_path:
            self.logger.log("MODEL_LOAD", f"使用检查点: {checkpoint_path}", "inference", level=3)
        else:
            if self.accelerator.is_local_main_process:
                print("\n" + "="*60)
                print("⚠️  警告：未找到检查点！")
                print("="*60)
                print("模型将使用随机初始化的权重进行推理。")
                print("这会导致推理质量很差，可能陷入无限循环。")
                print("\n建议操作：")
                print("1. 先训练模型：python train.py --num_epochs 30")
                print("2. 或指定已有检查点：python example.py --model_path checkpoints/your_checkpoint")
                print("="*60 + "\n")
            self.logger.log("MODEL_LOAD", "⚠️ 未找到检查点，使用随机初始化权重（警告：推理质量会很差）", "inference", level=3)

        return self.setup_models(checkpoint_path=checkpoint_path)

    def _prepare_initial_expression(self, initial_expr, x_data, y_data_len):
        """准备初始表达式和tokens"""
        import sympy as sp
        from ..symbolic.symbolic_utils import evaluate_expression_safe, expr_to_tree

        # 处理初始表达式
        if isinstance(initial_expr, list):
            current_tokens = initial_expr
            initial_expr = None
        else:
            if initial_expr is None:
                initial_expr = sp.Symbol('x0')
            elif isinstance(initial_expr, str):
                initial_expr = sp.sympify(initial_expr)

            initial_expr_str = expr_to_tree(initial_expr)
            current_tokens = initial_expr_str.split(',') if initial_expr_str else ['x0']

        # 计算初始表达式的预测值
        if initial_expr is not None:
            success, y_pred = evaluate_expression_safe(initial_expr, x_data)
            if not success:
                self.logger.log("INITIAL_EXPR_WARN", "无法计算初始表达式的预测值，使用零初始化", "inference", level=3)
                y_pred = [0.0] * y_data_len
        else:
            y_pred = [0.0] * y_data_len

        return initial_expr, current_tokens, y_pred

    # ============= 主推理方法 =============
    def symbolic_regression(self, model_path, x_data, y_data, n_steps=100, input_dim=None, max_expr_length=None, initial_expr=None):
        """符号回归 - 使用简单推理(贪婪搜索)接收数据点对，输出表达式

        Args:
            model_path: 模型检查点路径
            x_data: 输入x数据
            y_data: 目标y数据
            n_steps: 推理步数
            input_dim: 输入维度，如果为None则自动推断
            max_expr_length: 表达式最大token长度，如果为None则使用args中的值
            initial_expr: 初始表达式（sympy表达式或字符串），如果为None则使用x0
        """
        self.logger.log("SYMBOLIC_REGRESSION_START",
                       f"输入数据: x形状={x_data.shape}, y形状={y_data.shape} | n_steps={n_steps}",
                       "inference", level=3)

        model, condition_encoder, _, _, _, tokenizer = self._load_inference_model(model_path)
        device = self.device
        model.eval()
        condition_encoder.eval()

        # 准备输入数据
        x_values = torch.FloatTensor(x_data).unsqueeze(0).to(device)
        y_values = torch.FloatTensor(y_data).unsqueeze(0).to(device)

        if input_dim is None:
            input_dim = x_data.shape[1] if len(x_data.shape) > 1 else 1

        # 准备初始表达式
        initial_expr, current_tokens, y_pred = self._prepare_initial_expression(initial_expr, x_data, len(y_data))

        if self.accelerator.is_local_main_process:
            print(f"初始表达式: {initial_expr}")
            print(f"初始tokens: {current_tokens}")

        # 计算残差并编码条件
        residuals = y_values - torch.FloatTensor(y_pred).unsqueeze(0).to(device)
        point_mask = torch.ones_like(y_values)
        condition = condition_encoder(x_values, y_values, point_mask)

        # 记录条件信息（内联_encode_condition_and_log的逻辑）
        self.logger.log("INITIAL_DATA",
                       f"x_values: shape={x_values.shape} range=[{x_values.min():.4f},{x_values.max():.4f}] | "
                       f"y_target: shape={y_values.shape} range=[{y_values.min():.4f},{y_values.max():.4f}] | "
                       f"residuals: shape={residuals.shape} range=[{residuals.min():.4f},{residuals.max():.4f}] | "
                       f"initial_expr: {initial_expr} | initial_tokens: {current_tokens}",
                       "inference", level=3)
        self.logger.log("ARCHITECTURE_INFO",
                       "使用目标值y_target作为条件（架构改进：北极星模式）",
                       "inference", level=3)

        # 打印条件嵌入的前10个维度
        condition_cpu = condition.cpu().squeeze(0)
        condition_values = condition_cpu.detach().numpy()
        condition_preview = condition_values.flatten()[:10] if condition_values.ndim == 2 else condition_values[:10]
        self.logger.log("INITIAL_CONDITION",
                       f"condition: shape={condition.shape} range=[{condition.min():.4f},{condition.max():.4f}] | "
                       f"前10维: [{', '.join([f'{float(v):.6f}' for v in condition_preview])}]",
                       "inference", level=3)

        # 创建搜索器（内联_create_searcher的逻辑）
        self.logger.log("SIMPLE_SEARCH_INIT", f"初始化简单推理器 | n_steps={n_steps}", "inference", level=3)
        searcher = SimpleSymbolicRegression(
            model=model, condition_encoder=condition_encoder, tokenizer=tokenizer,
            device=device, args=self.args, logger=self.logger,
            min_action_score=self.MIN_ACTION_SCORE,
            max_expression_length=self.MAX_EXPRESSION_LENGTH,
            numerical_clip_threshold=self.NUMERICAL_CLIP_THRESHOLD
        )

        # 执行推理（内联_run_single_threshold_search的逻辑）
        if self.accelerator.is_local_main_process:
            print(f"\n执行单最佳操作推理...")

        residuals_np = residuals.cpu().squeeze(0).numpy()
        best_candidate = searcher.greedy_search(
            initial_tokens=current_tokens, initial_condition=condition,
            initial_residuals=residuals_np, x_data=x_data, y_data=y_data,
            x_values=x_values, n_steps=n_steps
        )

        final_expression = ','.join(best_candidate.tokens) if best_candidate and best_candidate.tokens else ""

        if best_candidate and self.accelerator.is_local_main_process:
            mse_score = best_candidate.mse_score
            mse_str = f'{mse_score:.6f}' if mse_score is not None else 'N/A'
            self.logger.log("SIMPLE_SEARCH_RESULT",
                           f"MSE分数: {mse_str} | "
                           f"操作历史: {' -> '.join(best_candidate.history[-5:]) if best_candidate.history else 'N/A'}",
                           "inference", level=3)

        self.logger.log("INFERENCE_COMPLETE", f"最终表达式: {final_expression}", "inference", level=3)

        return {
            'final_expression': final_expression,
            'initial_tokens': current_tokens,
            'final_tokens': best_candidate.tokens if best_candidate else [],
            'history': best_candidate.history if best_candidate else [],
            'position_actions_history': best_candidate.position_actions_history if best_candidate else [],
            'mse_score': best_candidate.mse_score if best_candidate else None
        }


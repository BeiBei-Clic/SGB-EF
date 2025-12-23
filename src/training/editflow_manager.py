"""
EditFlow连续流训练器 - 实现基于连续时间流匹配的编辑流模型训练
使用 Hugging Face Accelerate 进行分布式训练加速
"""

import torch
import numpy as np
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

from ..utils.special_tokens import SpecialTokensManager
from ..symbolic.data_generator import generate_flow_samples, load_dimension_index
from .flow import (
    KappaScheduler, sample_conditional_path,
    remove_gap_tokens, fill_gap_tokens_with_repeats,
    ContinuousFlowLoss, FlowDataset, custom_collate_fn
)
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig
from ..utils.gpu_monitor import get_gpu_memory_usage_string
from ..utils.misc_utils import find_latest_checkpoint, load_checkpoint
from ..utils.log_utils import (
    log_training_start, log_training_step, log_tensor_info,
    log_gpu_memory_usage, log_training_error
)

class EditFlowManager:
    """EditFlow模型管理器 - 支持训练和推理功能

    新增功能：多时间步采样训练
    - num_timesteps参数控制每个样本采样的时间步数量
    - 默认值为5（在train.py中定义），可以大幅提升训练数据利用效率
    - 每个原始样本将生成num_timesteps个训练实例
    - 在训练过程中会自动进行损失聚合，确保梯度计算正确
    """

    def __init__(self, args):
        self.args = args

        # 初始化 Accelerate - 自动处理分布式训练设置
        self.accelerator = Accelerator(
            mixed_precision='fp16' if getattr(args, 'use_fp16', True) else 'no',
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            log_with=getattr(args, 'log_with', None)
        )

        # 设置随机种子
        set_seed(args.seed)

        # 设备信息
        self.device = self.accelerator.device
        if self.accelerator.is_local_main_process:
            print("=== EditFlow符号回归预训练 (使用 Accelerate 加速) ===")
            print(f"样本数: {getattr(self.args, 'num_samples', 'N/A')}")
            print(f"最大维度: {getattr(self.args, 'max_dim', 'N/A')}")
            print(f"表达式最大长度: {getattr(self.args, 'max_expr_length', 'N/A')}")
            print(f"批次大小: {getattr(self.args, 'batch_size', 'N/A')}")
            print(f"训练轮数: {getattr(self.args, 'num_epochs', 'N/A')}")
            print(f"学习率: {getattr(self.args, 'learning_rate', 'N/A')}")
            print(f"测试集比例: {getattr(self.args, 'test_split', 'N/A')}")
            print(f"评估频率: 每{getattr(self.args, 'eval_every', 'N/A')}轮")
            print(f"基础模型: {getattr(self.args, 'base_model_name', 'N/A')}")
            print(f"条件嵌入模型: {getattr(self.args, 'condition_model_name', 'N/A')}")
            print(f"梯度累积步数: {getattr(self.args, 'gradient_accumulation_steps', 'N/A')}")
            print(f"FP16混合精度: {getattr(self.args, 'use_fp16', 'N/A')}")
            print(f"时间步采样数: {self.args.num_timesteps} (每个样本生成的时间步训练数量)")

            print(f"\nAccelerate 初始化完成")
            print(f"  设备: {self.device}")
            print(f"  分布式训练: {self.accelerator.distributed_type}")
            print(f"  进程数: {self.accelerator.num_processes}")
            print(f"  混合精度: {self.accelerator.mixed_precision}")

            # 显示GPU信息
            from ..utils.gpu_monitor import display_gpu_info
            display_gpu_info()

        # 记录训练开始日志
        log_training_start(self.args)
        log_gpu_memory_usage("初始化完成")

        # 时间调度器
        self.scheduler = KappaScheduler(scheduler_type='cubic')

    def set_seed(self, seed: int):
        """设置随机种子 - 现在使用 Accelerate 的 set_seed"""
        set_seed(seed)

    def prepare_data(self, tokenizer):
        """准备训练数据，支持多进程并行生成"""

        # 设置NCCL超时时间为无穷大，避免等待时超时
        import os
        os.environ["NCCL_TIMEOUT"] = "31536000"  # 1年（秒）

        # 1. 数据生成阶段：只使用主进程（单进程）
        cache_filename = f"data/flow_samples_{self.args.num_samples}_{self.args.max_dim}dim_{self.args.n_points}pts_{self.args.max_depth}depth_{self.args.max_expr_length}len.txt"

        # 只有主进程负责数据生成，避免NCCL通信问题
        if self.accelerator.is_local_main_process:
            print(f"准备连续流训练数据 (单进程生成模式)...")

            # 调用数据生成函数
            generate_flow_samples(
                num_samples=self.args.num_samples,
                max_dim=self.args.max_dim,
                n_points=self.args.n_points,
                max_depth=self.args.max_depth,
                max_expr_length=self.args.max_expr_length,
                verbose=True,  # 显示详细日志
            )
        else:
            # 非主进程跳过数据生成，等待主进程完成
            print(f"[Rank {self.accelerator.process_index}] 跳过数据生成，等待主进程完成...")

        # 2. 同步屏障：等待主进程完成数据生成
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            print("[主进程] 数据生成完成，开始训练阶段")

        # 3. 同步屏障：确保所有进程都能访问到完整的数据文件
        print(f"[Rank {self.accelerator.process_index}] 准备开始训练阶段...")
        self.accelerator.wait_for_everyone()

        # 加载索引（此时文件已完整）
        dimension_samples = load_dimension_index(cache_filename, verbose=self.accelerator.is_local_main_process)

        # === 修改开始：合并所有维度的位置索引 ===
        all_train_positions = []
        all_test_positions = []
        test_split = getattr(self.args, 'test_split', 0.2)

        for dim, positions in dimension_samples.items():
            # 这里的 shuffle 配合 set_seed 保证所有进程打乱顺序一致
            np.random.shuffle(positions)
            split_idx = int(len(positions) * (1 - test_split))

            all_train_positions.extend(positions[:split_idx])
            all_test_positions.extend(positions[split_idx:])

        # 再次整体打乱，让不同维度的样本混合，有助于模型泛化
        np.random.shuffle(all_train_positions)
        np.random.shuffle(all_test_positions)

        # 创建单一的训练和测试数据集
        train_dataset = FlowDataset(
            all_train_positions, cache_filename, tokenizer,
            max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length
        )
        test_dataset = FlowDataset(
            all_test_positions, cache_filename, tokenizer,
            max_dim=self.args.max_dim, max_expr_length=self.args.max_expr_length
        )

        # 关键参数：num_workers 和 drop_last
        # num_workers > 0 可以防止IO阻塞导致的GPU等待
        # drop_last=True 保证每个进程的 batch 数量严格一致，防止 DDP 卡死
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,  # 恢复多进程数据加载
            collate_fn=custom_collate_fn,
            drop_last=True, # 防止尾部batch不齐导致的死锁
            pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
            drop_last=False # 测试集通常不需要 drop_last，除非 evaluate 也有同步逻辑
        )

        # 使用 Accelerate 准备
        train_dataloader, test_dataloader = self.accelerator.prepare(
            train_dataloader, test_dataloader
        )

        if self.accelerator.is_local_main_process:
            print(f"数据准备完成: 训练集 {len(all_train_positions)} 样本, 测试集 {len(all_test_positions)} 样本")

        return train_dataloader, train_dataset, test_dataloader, test_dataset

    def setup_models(self, checkpoint_path=None):
        """
        初始化模型和tokenizer，支持从检查点加载

        Args:
            checkpoint_path: 检查点文件路径，如果为None则创建新模型

        Returns:
            model, condition_encoder, criterion, optimizer, tokenizer
        """
        if self.accelerator.is_local_main_process:
            print("初始化tokenizer和模型...")

        # 初始化tokenizer
        model_name = getattr(self.args, 'base_model_name', "google-bert/bert-base-uncased")
        cache_dir = getattr(self.args, 'cache_dir', "models/huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        if self.accelerator.is_local_main_process:
            print(f"正在加载tokenizer: {model_name}")
            print(f"模型缓存目录: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.accelerator.is_local_main_process:
            print(f"✓ Tokenizer加载完成，原始词表大小: {tokenizer.vocab_size}")

        # 初始化特殊符号管理器并添加缺失的符号
        special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=self.args.max_dim)
        special_tokens_manager.ensure_special_tokens(verbose=self.accelerator.is_local_main_process)

        if self.accelerator.is_local_main_process:
            print("初始化条件编码器...")
        condition_encoder = ConditionEncoder(
            model_name=self.args.condition_model_name,  # 保持兼容性，但实际不使用
            verbose=self.accelerator.is_local_main_process,
            max_length=getattr(self.args, 'condition_max_length', 512),  # 保持兼容性，但实际不使用
            args=self.args  # 传递args对象以使用SetTransformer参数
        ).to(self.device)

        if self.accelerator.is_local_main_process:
            print("初始化EditFlow模型...")
        config = EditFlowConfig(
            max_seq_len=self.args.max_expr_length,
            condition_dim=condition_encoder.output_dim,
            base_model_name=model_name,
            vocab_size=len(tokenizer.get_vocab()),
        )
        model = EditFlowTransformer(config, verbose=self.accelerator.is_local_main_process).to(self.device)

        # 创建优化器和损失函数
        criterion = ContinuousFlowLoss(scheduler_type='cubic')
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate * 0.1,  # 降低学习率以防止梯度爆炸
            weight_decay=self.args.weight_decay,
            eps=1e-8  # 增加数值稳定性
        )

        # 如果提供了检查点路径，加载预训练模型
        load_checkpoint(checkpoint_path, model, condition_encoder, self.device, optimizer, verbose=self.accelerator.is_local_main_process)

        # 使用 Accelerate 准备模型、优化器和数据加载器
        if self.accelerator.is_local_main_process:
            print(f"使用 Accelerate 准备模型和优化器...")
            print(f"  进程数: {self.accelerator.num_processes}")
            print(f"  设备: {self.accelerator.device}")
            print(f"  混合精度: {self.accelerator.mixed_precision}")

        model, condition_encoder, optimizer = self.accelerator.prepare(
            model, condition_encoder, optimizer
        )

        # 如果有checkpoint，使用Accelerate的load_state方法加载完整状态
        if checkpoint_path:
            if self.accelerator.is_local_main_process:
                print(f"Loading complete training state from {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.accelerator.is_local_main_process:
            print(f"EditFlow模型参数数量: {total_params:,}")

        return model, condition_encoder, criterion, optimizer, tokenizer

  
    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config, debug_info=None):
        # 多时间步采样参数
        num_timesteps = self.args.num_timesteps  # 使用命令行参数值

        original_batch_size = z0_token_ids.size(0)

        
        # 为每个样本采样多个时间步
        t = torch.rand(original_batch_size, num_timesteps, 1, device=self.device)  # [B, K, 1]

        # 扩展条件嵌入到多时间步
        # condition_embeddings: [B, D] -> [B, K, D]
        condition_embeddings = condition_embeddings.unsqueeze(1).expand(-1, num_timesteps, -1)

        # 扩展token序列到多时间步
        # z0_token_ids: [B, L] -> [B*K, L]
        z0_token_ids_expanded = z0_token_ids.unsqueeze(1).expand(-1, num_timesteps, -1).contiguous()
        z1_token_ids_expanded = z1_token_ids.unsqueeze(1).expand(-1, num_timesteps, -1).contiguous()

        # 重塑为标准批次格式
        z0_token_ids = z0_token_ids_expanded.reshape(original_batch_size * num_timesteps, -1)
        z1_token_ids = z1_token_ids_expanded.reshape(original_batch_size * num_timesteps, -1)
        t = t.reshape(original_batch_size * num_timesteps, -1)
        condition_embeddings = condition_embeddings.reshape(original_batch_size * num_timesteps, -1)

        batch_size = z0_token_ids.size(0)  # 更新batch_size为扩展后的大小

        # z0 token序列转换为概率分布
        batch_size, seq_len = z0_token_ids.shape
        z0_probs = torch.zeros(batch_size, seq_len, config.vocab_size, device=z0_token_ids.device)
        z0_probs.scatter_(2, z0_token_ids.unsqueeze(-1), 1.0)

        # z1 token序列转换为概率分布
        batch_size, seq_len = z1_token_ids.shape
        z1_probs = torch.zeros(batch_size, seq_len, config.vocab_size, device=z1_token_ids.device)
        z1_probs.scatter_(2, z1_token_ids.unsqueeze(-1), 1.0)

        # 记录多时间步采样信息（仅在第一个batch时）
        if debug_info and debug_info.get('is_first_batch', False) and self.accelerator.is_local_main_process:
            context = debug_info.get('context', '')
            log_training_step("MULTI_TIMESTEP_SAMPLE",
                            f"num_timesteps={num_timesteps} | original_batch={original_batch_size} | expanded_batch={batch_size}",
                            context, debug_level=1)
            log_tensor_info("z0_probs", z0_probs, level=2, context=context)
            log_tensor_info("z1_probs", z1_probs, level=2, context=context)

            # 记录token ID和时间步信息
            for i in range(min(3, z0_token_ids.size(0))):
                sample_idx = i // num_timesteps  # 原始样本索引
                t_idx = i % num_timesteps       # 时间步索引
                log_training_step("TOKEN_SAMPLE_CHECK",
                                f"样本{sample_idx} 时间步{t_idx} z0={z0_token_ids[i].tolist()} z1={z1_token_ids[i].tolist()} t={t[i].item():.4f}",
                                context, debug_level=3)
            log_training_step("TIMESTEP_STATS",
                            f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}",
                            context, debug_level=2)


        z_t = sample_conditional_path(z0_probs, z1_probs, t, self.scheduler)

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
        gap_token = dataset.special_tokens_manager.tokenizer.convert_tokens_to_ids('<gap>')

        # 获取时间步采样数量
        num_timesteps = self.args.num_timesteps

        lambda_ins = pred_rates[:, :, 0:1]
        lambda_sub = pred_rates[:, :, 1:2]
        lambda_del = pred_rates[:, :, 2:3]

        ins_probs = lambda_ins * pred_ins_probs
        sub_probs = lambda_sub * pred_sub_probs

        # 简化：如果词汇表已经完整，直接使用
        extended_ins_probs = ins_probs
        extended_sub_probs = sub_probs

        u_cat = torch.cat([lambda_ins * extended_ins_probs, lambda_sub * extended_sub_probs, lambda_del], dim=-1)

        u_z = fill_gap_tokens_with_repeats(u_cat, z_gap_mask, z_pad_mask)

        u_mask = criterion.make_ut_mask_from_z(z_t, z1_token_ids, effective_vocab_size, gap_token, dataset.special_tokens_manager)

        loss = criterion(u_z, u_mask, t, effective_vocab_size, accelerator=self.accelerator)

        return loss

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)

        # 显示进度条 - 只在主进程显示
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}",
                          disable=not self.accelerator.is_local_main_process)

        # 处理模型配置
        config = model.module.config if hasattr(model, 'module') else model.config

        # 在epoch开始时清零梯度
        optimizer.zero_grad()

        # 只在主进程设置初始进度条显示
        if self.accelerator.is_local_main_process:
            progress_bar.set_postfix({'loss': '0.0000'})

        for batch_idx, batch in enumerate(progress_bar):
            x_values = batch['x_values'].to(self.device)
            residuals = batch['residuals'].to(self.device)
            z0_token_ids = batch['z0_token_ids'].to(self.device)
            z1_token_ids = batch['z1_token_ids'].to(self.device)

            # 记录token解码信息（仅在第一个batch时）
            if batch_idx == 0 and self.accelerator.is_local_main_process:
                log_training_step("TOKEN_DECODE", f"维度 {dimension} - 第一个batch的token解码信息", debug_level=2)

                # 解码z0_token_ids
                vocab = dataset.special_tokens_manager.tokenizer.get_vocab()
                id_to_token = {v: k for k, v in vocab.items()}

                z0_expressions = []
                for i in range(min(3, z0_token_ids.size(0))):
                    z0_tokens = []
                    for token_id in z0_token_ids[i].tolist():
                        if token_id in id_to_token:
                            token = id_to_token[token_id]
                            z0_tokens.append(token)
                    z0_expression = ','.join(z0_tokens) if z0_tokens else "<empty>"
                    z0_expressions.append(f"样本{i}: {z0_expression}")

                z1_expressions = []
                for i in range(min(3, z1_token_ids.size(0))):
                    z1_tokens = []
                    for token_id in z1_token_ids[i].tolist():
                        if token_id in id_to_token:
                            token = id_to_token[token_id]
                            z1_tokens.append(token)
                    z1_expression = ','.join(z1_tokens) if z1_tokens else "<empty>"
                    z1_expressions.append(f"样本{i}: {z1_expression}")

                log_training_step("TOKEN_DECODE_Z0", "\n".join(z0_expressions), f"维度{dimension}", debug_level=3)
                log_training_step("TOKEN_DECODE_Z1", "\n".join(z1_expressions), f"维度{dimension}", debug_level=3)

            condition_embeddings = condition_encoder(x_values, residuals)

            # 记录条件嵌入信息（每个batch都记录，用于调试NaN来源）
            if self.accelerator.is_local_main_process:
                log_tensor_info("condition_embeddings", condition_embeddings, level=2, context=f"维度{dimension}_batch{batch_idx}")
                log_tensor_info("x_values", x_values, level=2, context=f"维度{dimension}_batch{batch_idx}")
                log_tensor_info("residuals", residuals, level=2, context=f"维度{dimension}_batch{batch_idx}")

            # 准备调试信息
            debug_info = None
            if batch_idx == 0:
                debug_info = {
                    'is_first_batch': True,
                    'context': f'维度{dimension}'
                }

            forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, config, debug_info)

            # 记录前向传播输出（每个batch都记录，用于调试NaN来源）
            if self.accelerator.is_local_main_process:
                log_tensor_info("pred_rates", forward_results['pred_rates'], level=2, context=f"维度{dimension}_batch{batch_idx}")
                log_tensor_info("pred_ins_probs", forward_results['pred_ins_probs'], level=2, context=f"维度{dimension}_batch{batch_idx}")
                log_tensor_info("pred_sub_probs", forward_results['pred_sub_probs'], level=2, context=f"维度{dimension}_batch{batch_idx}")

            # 分布式健康检查：确保前向传播结果在不同进程间合理
            if self.accelerator.distributed_type != "NO":
                pred_rates = forward_results['pred_rates']

                # 检查是否有任何进程的模型输出包含NaN
                local_has_nan = torch.isnan(pred_rates).any().float()
                gathered_nan_results = self.accelerator.gather(local_has_nan)
                global_has_nan = gathered_nan_results.sum()

                if global_has_nan.item() > 0:
                    if self.accelerator.is_local_main_process:
                        log_training_error("FORWARD_NAN", f"维度{dimension} 检测到前向传播NaN，跳过批次", f"batch_idx:{batch_idx}")
                    # 继续执行到criterion中的分布式NaN检测，那里会统一跳过

            # 尝试计算损失，如果包含NaN则跳过该批次
            try:
                loss = self.compute_loss(forward_results, criterion, dataset) / gradient_accumulation_steps
                # 记录损失值（每个batch都记录）
                if self.accelerator.is_local_main_process:
                    log_training_step("LOSS_COMPUTED", f"loss={loss.item():.6f}", f"维度{dimension}_batch{batch_idx}", debug_level=2)
            except ValueError as e:
                if "批次包含异常值" in str(e):
                    log_training_error("BATCH_SKIP", f"跳过批次 {batch_idx} (包含NaN/Inf): {e}", f"维度{dimension}")

                    # 分布式同步：确保所有进程都到达这个点
                    log_training_step("SYNC_SKIP", f"所有进程同步跳过批次 {batch_idx}", f"维度{dimension}", debug_level=2)

                    # 等待所有进程都准备好跳过
                    self.accelerator.wait_for_everyone()

                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # 再次同步确保所有进程都完成了清理
                    self.accelerator.wait_for_everyone()

                    continue  # 跳过此批次
                else:
                    raise  # 重新抛出其他类型的ValueError

            grad_norm = 0.0
            if not torch.isnan(loss):
                # 使用 Accelerate 的 backward 而不是直接调用 loss.backward()
                self.accelerator.backward(loss)

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # 统一计算梯度范数并裁剪
                    all_params = list(model.parameters()) + list(condition_encoder.parameters())

                    # 记录梯度统计信息（用于调试NaN来源）
                    if self.accelerator.is_local_main_process:
                        grad_max = 0.0
                        grad_min = 0.0
                        grad_has_nan = False
                        for param in all_params:
                            if param.grad is not None:
                                if torch.isnan(param.grad).any():
                                    grad_has_nan = True
                                    break
                                grad_max = max(grad_max, param.grad.abs().max().item())
                                grad_min = max(grad_min, param.grad.abs().min().item())
                        log_training_step("GRAD_STATS", f"max={grad_max:.6f} min={grad_min:.6f} has_nan={grad_has_nan}",
                                        f"维度{dimension}_batch{batch_idx}", debug_level=2)

                    # 检查是否有NaN梯度 - 在gradient unscaling之前检查
                    has_nan_grad = False
                    for param in all_params:
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break

                    if has_nan_grad:
                        if self.accelerator.is_local_main_process:
                            log_training_error("GRAD_NAN", f"检测到NaN梯度，跳过此次更新", f"维度{dimension}_batch{batch_idx}")
                        optimizer.zero_grad()
                        continue

                    # 使用Accelerate的梯度裁剪（会自动处理混合精度）
                    grad_norm = self.accelerator.clip_grad_norm_(all_params, 1.0)  # 恢复正常的梯度裁剪阈值

                    # 记录梯度范数
                    if self.accelerator.is_local_main_process:
                        log_training_step("GRAD_NORM", f"grad_norm={grad_norm:.4f}",
                                        f"维度{dimension}_batch{batch_idx}", debug_level=2)

                    optimizer.step()
                    optimizer.zero_grad()  # 在step后清零梯度，为下一次累积做准备

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # 只在主进程更新进度条显示
            if self.accelerator.is_local_main_process:
                postfix_dict = {
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'grad_norm': f'{grad_norm:.3f}' if isinstance(grad_norm, (int, float)) else f'{grad_norm.item():.3f}'
                }
                progress_bar.set_postfix(postfix_dict)

        # 等待所有进程完成
        self.accelerator.wait_for_everyone()

        # 收集平均损失（跨进程）
        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        num_batches_tensor = torch.tensor(num_batches, device=self.device)

        # 使用 Accelerate 收集所有进程的损失
        gathered_losses = self.accelerator.gather(total_loss_tensor)
        gathered_batches = self.accelerator.gather(num_batches_tensor)

        total_batches = gathered_batches.sum().item()
        avg_loss = gathered_losses.sum().item() / total_batches if total_batches > 0 else 0.0

        return avg_loss, num_batches

    def evaluate(self, model, condition_encoder, criterion, test_dataloader, test_dataset):
        """测试集评估"""
        model.eval()
        condition_encoder.eval()

        total_loss = 0.0
        num_batches = 0
        config = model.module.config if hasattr(model, 'module') else model.config

        with torch.no_grad():
            # === 修改：不再循环 dim，直接遍历 dataloader ===
            for batch in test_dataloader:
                x_values = batch['x_values'].to(self.device)
                residuals = batch['residuals'].to(self.device)
                z0_token_ids = batch['z0_token_ids'].to(self.device)
                z1_token_ids = batch['z1_token_ids'].to(self.device)

                condition_embeddings = condition_encoder(x_values, residuals)
                forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, test_dataset, config)

                # 尝试计算损失，如果包含NaN则跳过该批次
                try:
                    loss = self.compute_loss(forward_results, criterion, test_dataset)
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1
                except ValueError as e:
                    if "批次包含异常值" in str(e):
                        # 评估时跳过包含NaN的批次，不输出错误信息避免干扰
                        continue
                    else:
                        raise

        # 等待所有进程完成
        self.accelerator.wait_for_everyone()

        # 使用 Accelerate 收集所有进程的损失
        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        num_batches_tensor = torch.tensor(num_batches, device=self.device)

        gathered_losses = self.accelerator.gather(total_loss_tensor)
        gathered_batches = self.accelerator.gather(num_batches_tensor)

        total_batches = gathered_batches.sum().item()
        avg_loss = gathered_losses.sum().item() / total_batches if total_batches > 0 else float('inf')

        return avg_loss


    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config, is_final=False):
        # 等待所有进程同步
        self.accelerator.wait_for_everyone()

        # 创建checkpoint目录
        checkpoint_dir = os.path.join(
            self.args.save_dir,
            "continuous_flow_final" if is_final else f"checkpoint_epoch_{epoch+1}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 使用 Accelerate 的 save_state 方法（推荐的正确方式）
        self.accelerator.save_state(checkpoint_dir)

        # 另外保存模型配置信息
        if self.accelerator.is_local_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_encoder = self.accelerator.unwrap_model(condition_encoder)

            config_data = {
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'condition_encoder_state_dict': unwrapped_encoder.state_dict(),
                'loss': loss,
                'config': config,
                'args': self.args,
                'accelerate_config': {
                    'distributed_type': str(self.accelerator.distributed_type),
                    'num_processes': self.accelerator.num_processes,
                    'mixed_precision': str(self.accelerator.mixed_precision),
                }
            }

            if is_final:
                config_data['scheduler_type'] = 'cubic'

            # 保存配置信息
            config_path = os.path.join(checkpoint_dir, "training_config.json")
            torch.save(config_data, config_path)

        return checkpoint_dir

    def train(self):
        # 检查检查点并加载模型
        checkpoint_path = find_latest_checkpoint(self.args)
        if self.accelerator.is_local_main_process:
            print(f"使用设备: {self.device}")
            print(f"{'找到检查点' if checkpoint_path else '未找到检查点，将从基础模型开始训练'}: {checkpoint_path or ''}")

        model, condition_encoder, criterion, optimizer, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

        # 注意这里接收返回值的变化
        train_dataloader, train_dataset, test_dataloader, test_dataset = self.prepare_data(tokenizer)

        model_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad)
        if self.accelerator.is_local_main_process:
            print(f"模型参数数量: {model_params:,}, 条件编码器参数数量: {encoder_params:,}")
            print(f"开始连续流训练 ({self.args.num_epochs} epochs)...")
            # 记录训练开始到 training.log
            log_training_step("TRAINING_START", f"开始训练 | num_epochs={self.args.num_epochs} | model_params={model_params:,} | encoder_params={encoder_params:,}", debug_level=1)

        config = model.module.config if hasattr(model, 'module') else model.config
        eval_every = getattr(self.args, 'eval_every', 5)

        for epoch in range(self.args.num_epochs):
            # === 修改开始：不再循环 dim，直接传整个 dataloader ===
            # 这里传入 "Mixed" 作为维度名称仅用于显示
            avg_loss, num_batches = self.train_epoch(
                model, condition_encoder, criterion, optimizer,
                train_dataloader, train_dataset, epoch, "Mixed"
            )

            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs} 完成, 训练损失: {avg_loss:.4f}")
                # 记录训练成果到 training.log
                log_training_step("EPOCH_COMPLETE", f"Epoch {epoch+1}/{self.args.num_epochs} | train_loss={avg_loss:.4f} | batches={num_batches}", debug_level=1)

            # 修改 evaluate 调用，传入单个 dataloader
            if (epoch + 1) % eval_every == 0 or epoch == self.args.num_epochs - 1:
                test_loss = self.evaluate(model, condition_encoder, criterion, test_dataloader, test_dataset)
                if self.accelerator.is_local_main_process:
                    print(f"测试集损失: {test_loss:.4f}")
                    # 记录评估结果到 training.log
                    log_training_step("EVALUATION", f"Epoch {epoch+1}/{self.args.num_epochs} | test_loss={test_loss:.4f}", debug_level=1)

            # 保存检查点
            if (epoch + 1) % self.args.save_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch, config
                )
                if self.accelerator.is_local_main_process:
                    print(f"检查点已保存到: {checkpoint_path}")
                    # 记录检查点保存到 training.log
                    log_training_step("CHECKPOINT_SAVED", f"Epoch {epoch+1}/{self.args.num_epochs} | path={checkpoint_path} | train_loss={avg_loss:.4f}", debug_level=1)

        # 保存最终模型
        final_path = self.save_checkpoint(
            model, condition_encoder, optimizer, avg_loss, self.args.num_epochs - 1, config, is_final=True
        )
        if self.accelerator.is_local_main_process:
            print(f"最终模型已保存到: {final_path}")
            # 记录训练完成到 training.log
            log_training_step("TRAINING_COMPLETE", f"训练完成 | final_path={final_path} | final_train_loss={avg_loss:.4f} | total_epochs={self.args.num_epochs}", debug_level=1)

        # 显式清理分布式资源
        try:
            self.accelerator.free_memory()
            if self.accelerator.is_local_main_process:
                print("✓ 分布式资源已清理")
        except Exception as e:
            if self.accelerator.is_local_main_process:
                print(f"⚠️ 资源清理时出现警告: {e}")

        return model, condition_encoder

    def symbolic_regression(self, model_path, x_data, y_data, n_steps=100, input_dim=None, max_expr_length=None):
        """符号回归 - 接收数据点对，输出表达式

        Args:
            model_path: 模型检查点路径
            x_data: 输入x数据
            y_data: 目标y数据
            n_steps: 推理步数
            input_dim: 输入维度，如果为None则自动推断
            max_expr_length: 表达式最大token长度，如果为None则使用args中的值
        """
        log_training_step("SYMBOLIC_REGRESSION_START", f"开始符号回归推理 | 输入数据: x形状={x_data.shape}, y形状={y_data.shape}", "inference", debug_level=1)

        # 加载模型
        checkpoint_path = model_path if model_path and os.path.exists(model_path) else None
        if checkpoint_path:
            log_training_step("MODEL_LOAD", f"使用检查点: {checkpoint_path}", "inference", debug_level=1)
        else:
            log_training_step("MODEL_LOAD", "未找到检查点，将使用基础模型进行推理", "inference", debug_level=1)

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
        from ..symbolic.symbolic_utils import evaluate_expression_safe, evaluate_expression_with_constants

        if input_dim == 1:
            # 一维情况：初始表达式为 x0
            initial_expr = sp.Symbol('x0')
        else:
            # 多维情况：初始表达式为 x0+x1+x2+...
            initial_expr = sum(sp.Symbol(f'x{i}') for i in range(input_dim))

        # 计算初始表达式在x_data上的预测值
        success, y_pred = evaluate_expression_safe(initial_expr, x_data)
        if not success:
            if self.accelerator.is_local_main_process:
                log_training_error("INITIAL_EXPR_FAILED", f"无法计算初始表达式 '{initial_expr}' 的预测值，使用零残差", "inference")
            residuals = y_values
        else:
            # 计算残差：真实值 - 预测值
            residuals = y_values - torch.FloatTensor(y_pred).unsqueeze(0).to(device)

        # 记录初始数据状态（验证数据是否正确传递）
        log_training_step("INITIAL_DATA", f"x_values: shape={x_values.shape} range=[{x_values.min():.4f},{x_values.max():.4f}] | y_values: shape={y_values.shape} range=[{y_values.min():.4f},{y_values.max():.4f}] | residuals: shape={residuals.shape} range=[{residuals.min():.4f},{residuals.max():.4f}] mean={residuals.mean():.4f} std={residuals.std():.4f}", "inference", debug_level=1)

        condition = condition_encoder(x_values, residuals)

        # 记录初始条件编码状态
        log_training_step("INITIAL_CONDITION", f"condition: shape={condition.shape} range=[{condition.min():.4f},{condition.max():.4f}] mean={condition.mean():.4f} std={condition.std():.4f}", "inference", debug_level=1)

        # 构建初始前缀表达式（与训练格式一致）
        if input_dim == 1:
            # 一维情况：初始表达式为 x0
            current_tokens = ['x0']
        else:
            # 多维情况：使用嵌套的add前缀表达式，例如 add,x0,x1 表示 (x0 + x1)
            # 对于三个变量：add,add,x0,x1,x2 表示 ((x0 + x1) + x2)
            current_tokens = []
            for i in range(input_dim - 1):
                current_tokens.append('add')

            # 添加所有变量
            for i in range(input_dim):
                current_tokens.append(f'x{i}')

            # 对于3个变量：add,add,x0,x1,x2
            # 对于2个变量：add,x0,x1

        # 初始化token管理器，确保覆盖数据维度
        actual_max_dim = max(input_dim, self.args.max_dim) if hasattr(self.args, 'max_dim') else input_dim
        special_tokens_manager = SpecialTokensManager(tokenizer, max_dim=actual_max_dim)

        # 确保所有需要的符号都在tokenizer中
        special_tokens_manager.ensure_special_tokens(verbose=self.accelerator.is_local_main_process)

        log_training_step("INFERENCE_SETUP", f"推理步数: {n_steps} | 输入数据形状: x_values={x_values.shape}, y_values={y_values.shape} | 条件嵌入形状: {condition.shape} | 初始表达式: {','.join(current_tokens)}", "inference", debug_level=1)

        for step in range(n_steps):
            if self.accelerator.is_local_main_process:
                print(f"推理步骤 {step + 1}/{n_steps}, 当前表达式: {','.join(current_tokens) if current_tokens else '<blank>'}")

            log_training_step("INFERENCE_STEP", f"推理步骤 {step + 1}/{n_steps} | 当前表达式: {','.join(current_tokens) if current_tokens else '<空白>'}", "inference", debug_level=2)

            tokenized_expr = special_tokens_manager.tokenize_expression(','.join(current_tokens))

            # 验证输入ID的正确性
            log_training_step("TOKEN_ID_VERIFY", f"当前表达式tokens: {current_tokens} | tokenized_expr IDs: {tokenized_expr}", "inference", debug_level=3)

            # 验证每个token ID的有效性
            vocab = special_tokens_manager.tokenizer.get_vocab()
            valid_ids = []
            invalid_ids = []
            for i, token_id in enumerate(tokenized_expr):
                token_name = special_tokens_manager.tokenizer.convert_ids_to_tokens([token_id])[0]
                if token_id < len(vocab) and token_name in vocab:
                    valid_ids.append((i, token_id, token_name, vocab[token_name]))
                else:
                    invalid_ids.append((i, token_id, token_name))

            log_training_step("TOKEN_VALIDATION", f"有效token IDs: {len(valid_ids)} | 无效token IDs: {len(invalid_ids)}", "inference", debug_level=3)
            if invalid_ids:
                log_training_step("TOKEN_INVALID", f"发现无效token IDs: {invalid_ids}", "inference", debug_level=2)

            # 检查是否有重复或UNK token
            unk_token_id = vocab.get('<unk>', None)
            if unk_token_id in tokenized_expr:
                log_training_step("TOKEN_UNK_WARNING", f"发现UNK token (ID={unk_token_id})", "inference", debug_level=2)

            max_len = getattr(self.args, 'max_expr_length', 128)
            if len(tokenized_expr) > max_len - 1:
                tokenized_expr = tokenized_expr[:max_len-1]
                log_training_step("EXPRESSION_TRUNCATE", f"表达式过长，截断至 {max_len-1} 个token", "inference", debug_level=2)

            # 使用统一的特殊token管理
            bos_token = special_tokens_manager.tokenizer.convert_tokens_to_ids('<s>')
            pad_token = special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')

            tokenized_expr = [bos_token] + tokenized_expr
            tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

            input_ids = torch.LongTensor([tokenized_expr]).to(device)
            # 修正：基于实际token内容构建掩码，而不是位置假设
            attention_mask = (input_ids != pad_token).float().to(device)

            t = torch.tensor([[0.1 + 0.9 * step / n_steps]], dtype=torch.float32, device=device)

            # 记录当前条件嵌入状态（每步都记录，用于调试）
            log_training_step("CONDITION_STATE", f"步骤{step+1} condition: 范围=[{condition.min():.4f},{condition.max():.4f}] mean={condition.mean():.4f} std={condition.std():.4f} norm={condition.norm():.4f}", "inference", debug_level=2)

            log_training_step("MODEL_INPUT", f"时间步t: {t[0,0]:.4f} | input_ids长度: {len(tokenized_expr)} | 有效token数量: {attention_mask[0].sum().item()}", "inference", debug_level=2)

            # 解码完整的input_ids序列
            decoded_tokens = []
            for token_id in input_ids[0].tolist():
                token_name = special_tokens_manager.tokenizer.convert_ids_to_tokens([token_id])[0]
                decoded_tokens.append(token_name)
            log_training_step("TOKEN_DECODE_FULL", f"解码的token序列: {decoded_tokens}", "inference", debug_level=3)

            # 检查input_ids的统计信息
            input_ids_np = input_ids[0].cpu().numpy()

            # 确认没有无效的token ID
            if input_ids_np.max() >= len(vocab):
                log_training_step("TOKEN_INVALID_ERROR", f"发现无效的token ID: {input_ids_np.max()}, vocab大小: {len(vocab)}", "inference", debug_level=1)
            else:
                log_training_step("TOKEN_VALID_CHECK", "所有token ID都有效", "inference", debug_level=3)

            with torch.no_grad():
                rates, insert_probs, substitute_probs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    time_steps=t, condition=condition
                )

                # 记录模型输出的原始值（仅第一步）
                if step == 0:
                    log_training_step("MODEL_OUTPUT_ANALYSIS", f"模型输出形状: {rates.shape} | 位置数: {rates.size(1)}", "inference", debug_level=2)

                    # 检查是否所有位置都完全相同
                    rates_flat = rates[0, :, :].cpu().numpy()
                    unique_rows = np.unique(rates_flat, axis=0)
                    if unique_rows.shape[0] < rates_flat.shape[0]:
                        log_training_step("MODEL_OUTPUT_UNIQUE", f"发现重复的输出行，唯一行数: {unique_rows.shape[0]}/{rates_flat.shape[0]}", "inference", debug_level=2)

                    # 检查insert_probs的分布
                    top_insert_tokens = torch.topk(insert_probs[0, 1], 5)  # 检查位置1的top5
                    log_training_step("MODEL_INSERT_PROBS", f"位置1的top5插入tokens: {top_insert_tokens}", "inference", debug_level=3)

                lambda_ins = rates[0, :, 0].cpu().numpy()
                lambda_sub = rates[0, :, 1].cpu().numpy()
                lambda_del = rates[0, :, 2].cpu().numpy()

                base_length = int(attention_mask[0].sum().item())
                effective_length = max(base_length, min(10, input_ids.size(1)))

                if self.accelerator.is_local_main_process:
                    # 显示前几个位置的操作强度
                    for i in range(1, min(6, effective_length)):
                        log_training_step("OPERATION_STRENGTH", f"位置{i}: ins={lambda_ins[i]:.4f} sub={lambda_sub[i]:.4f} del={lambda_del[i]:.4f}", "inference", debug_level=3)

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

                if self.accelerator.is_local_main_process:
                    log_training_step("BEST_ACTION", f"最佳操作: {best_action} | 分数: {best_score:.4f}", "inference", debug_level=2)

                if best_action and best_score > 0.01:
                    action_type, pos = best_action

                    if self.accelerator.is_local_main_process:
                        log_training_step("EXECUTE_ACTION", f"执行操作: {action_type} | 位置: {pos}", "inference", debug_level=2)

                    if action_type == 'insert':
                        best_token = torch.argmax(insert_probs[0, pos]).item()
                        if self.accelerator.is_local_main_process:
                            log_training_step("INSERT_TOKEN", f"插入token: {best_token}", "inference", debug_level=3)

                        # 直接使用tokenizer转换token ID为token名称
                        best_token_name = tokenizer.convert_ids_to_tokens([best_token])[0]
                        current_tokens.insert(pos, best_token_name)
                        if self.accelerator.is_local_main_process:
                            log_training_step("INSERT_COMPLETE", f"插入完成: {best_token_name} -> 位置{pos}", "inference", debug_level=2)

                    elif action_type == 'substitute' and pos < len(current_tokens):
                        best_token = torch.argmax(substitute_probs[0, pos]).item()
                        if self.accelerator.is_local_main_process:
                            log_training_step("SUBSTITUTE_TOKEN", f"替换token: {best_token}", "inference", debug_level=3)

                        # 直接使用tokenizer转换token ID为token名称
                        best_token_name = tokenizer.convert_ids_to_tokens([best_token])[0]
                        current_tokens[pos] = best_token_name
                        if self.accelerator.is_local_main_process:
                            log_training_step("SUBSTITUTE_COMPLETE", f"替换完成: {best_token_name} -> 位置{pos}", "inference", debug_level=2)

                    elif action_type == 'delete' and pos < len(current_tokens):
                        deleted_token = current_tokens[pos]
                        del current_tokens[pos]
                        if self.accelerator.is_local_main_process:
                            log_training_step("DELETE_COMPLETE", f"删除完成: {deleted_token} <- 位置{pos}", "inference", debug_level=2)

                    # 在每次修改后评估表达式
                    current_expr_str = ','.join(current_tokens)
                    if self.accelerator.is_local_main_process:
                        log_training_step("EXPRESSION_UPDATE", f"表达式更新: {current_expr_str}", "inference", debug_level=2)

                    # 评估表达式并进行常数优化
                    log_training_step("BEFORE_EVAL", f"开始评估表达式: {current_expr_str}", "inference", debug_level=2)
                    eval_success, optimized_expr, loss = evaluate_expression_with_constants(
                        current_expr_str, x_data, y_data
                    )
                    log_training_step("AFTER_EVAL", f"评估完成: eval_success={eval_success}, loss={loss:.6f}", "inference", debug_level=2)

                    if eval_success:
                        # 更新残差为基于优化后表达式的残差
                        try:
                            # 计算优化后表达式的预测值
                            from ..symbolic.symbolic_utils import evaluate_expr
                            y_pred_optimized = evaluate_expr(optimized_expr, x_data)

                            # 裁剪超大值到合理范围，防止数值溢出
                            # 使用很大的阈值（1e6），让大部分有效值通过
                            CLIP_THRESHOLD = 1e6
                            y_pred_clipped = np.clip(y_pred_optimized, -CLIP_THRESHOLD, CLIP_THRESHOLD)

                            # 检查是否被裁剪
                            was_clipped = np.any(y_pred_optimized != y_pred_clipped)
                            n_clipped = np.sum(y_pred_optimized != y_pred_clipped) if was_clipped else 0

                            # 更新残差：真实值 - 裁剪后的预测值
                            new_residuals = y_data - y_pred_clipped

                            # 同样裁剪残差
                            new_residuals = np.clip(new_residuals, -CLIP_THRESHOLD, CLIP_THRESHOLD)

                            # 重新计算条件
                            condition = condition_encoder(x_values, torch.FloatTensor(new_residuals).unsqueeze(0).to(device))

                            # 检查新条件是否有效
                            if torch.any(torch.isnan(condition)) or torch.any(torch.isinf(condition)):
                                if self.accelerator.is_local_main_process:
                                    log_training_error("CONDITION_INVALID", f"新条件包含无效值，跳过更新", "inference")
                                continue

                            if self.accelerator.is_local_main_process:
                                clip_info = f", 裁剪了{n_clipped}个值" if was_clipped else ""
                                log_training_step("EVAL_SUCCESS", f"评估成功: {optimized_expr} | loss: {loss:.6f}{clip_info}", "inference", debug_level=2)
                                log_training_step("RESIDUAL_UPDATE", f"新残差: 范围=[{new_residuals.min():.6f},{new_residuals.max():.6f}] mean={new_residuals.mean():.6f} std={new_residuals.std():.6f} | 新condition: 范围=[{condition.min():.4f},{condition.max():.4f}] norm={condition.norm():.4f}", "inference", debug_level=2)
                        except Exception as e:
                            if self.accelerator.is_local_main_process:
                                log_training_error("RESIDUAL_UPDATE_ERROR", f"残差更新失败: {e}", "inference")
                    else:
                        if self.accelerator.is_local_main_process:
                            log_training_error("EVAL_FAILED", f"表达式评估失败: {current_expr_str}", "inference")
                else:
                    if self.accelerator.is_local_main_process:
                        log_training_step("NO_ACTION", "没有找到合适的操作，跳过此步骤", "inference", debug_level=3)

            if len(current_tokens) > 50:
                old_len = len(current_tokens)
                current_tokens = current_tokens[:50]
                if self.accelerator.is_local_main_process:
                    log_training_step("EXPRESSION_TRUNCATE", f"表达式过长，截断从{old_len}到50", "inference", debug_level=2)

        # 返回最终的表达式
        final_expression = ','.join(current_tokens) if current_tokens else ""
        if self.accelerator.is_local_main_process:
            log_training_step("INFERENCE_COMPLETE", f"符号回归完成 | 最终表达式: {final_expression}", "inference", debug_level=1)

        return final_expression

    
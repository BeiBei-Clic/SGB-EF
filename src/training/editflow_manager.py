"""
EditFlow迭代优化训练器 - 实现基于迭代式编辑操作的符号回归模型训练
使用 Hugging Face Accelerate 进行分布式训练加速

重构说明：EditFlowManager 现在作为协调者，委托具体任务给：
- EditFlowTrainer: 负责训练循环和评估
- InferenceEngine: 负责符号回归推理
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
    ContinuousFlowLoss, prepare_dataset_hf, custom_collate_fn
)
from ..modeling.condition_encoder import SetTransformerConditionEncoder
from ..modeling.llama_editflow import LlamaEditFlowBackbone
from ..utils.misc_utils import find_latest_checkpoint, load_checkpoint
from ..utils.logger import Logger
from .greedy_search import SimpleSymbolicRegression

# 新导入：训练器和推理引擎
from .trainers.editflow_trainer import EditFlowTrainer
from .inference.inference_engine import InferenceEngine


class EditFlowManager:
    """EditFlow模型管理器 - 协调训练和推理功能

    重构后的职责：
    - 数据准备和管理
    - 模型设置和检查点管理
    - 训练流程协调（委托给 EditFlowTrainer）
    - 推理流程协调（委托给 InferenceEngine）

    架构特点：迭代优化模式
    - 模型直接预测从z0到z1的编辑操作（插入、删除、替换）
    - 时间步固定为0，学习从起点到目标的直接编辑路径
    - 使用目标值y_target作为条件（而非残差），保持条件恒定作为"北极星"
    """

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

    # ============= 数据管理方法 =============
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

        import time
        prepare_start = time.time()
        train_dataloader, test_dataloader = self.accelerator.prepare(
            train_dataloader, test_dataloader
        )
        prepare_time = time.time() - prepare_start

        if self.accelerator.is_local_main_process:
            print(f"✓ 分布式训练准备完成 (耗时: {prepare_time:.2f}秒)")

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
        """分割训练集和测试集（统一方法）"""
        import time

        # 当样本数很少时，让所有样本同时用于训练和测试
        if self.args.num_samples <= self.args.batch_size:
            if self.accelerator.is_local_main_process:
                mode_str = "流式" if use_stream else "非流式"
                print(f"{mode_str}模式: 样本数({self.args.num_samples}) ≤ batch_size({self.args.batch_size})")
                print(f"        所有样本将同时用于训练和测试")

            full_dataset = prepare_dataset_hf(
                data_file=cache_filename, tokenizer=tokenizer,
                max_expr_length=self.args.max_expr_length,
                stream=use_stream, num_proc=num_proc,
                logger=self.logger
            )
            return full_dataset, full_dataset, self.args.num_samples, self.args.num_samples

        # 正常分割逻辑
        split_ratio = 1 - self.args.test_split
        train_size = int(self.args.num_samples * split_ratio)
        test_size = self.args.num_samples - train_size

        if use_stream:
            # 流式模式：使用skip和take
            if self.accelerator.is_local_main_process:
                print(f"流式模式: 训练集约 {train_size} 样本, 测试集约 {test_size} 样本")

            train_dataset = prepare_dataset_hf(
                data_file=cache_filename, tokenizer=tokenizer,
                max_expr_length=self.args.max_expr_length,
                stream=True, num_proc=num_proc,
                skip=0, take=train_size,
                logger=self.logger
            )
            test_dataset = prepare_dataset_hf(
                data_file=cache_filename, tokenizer=tokenizer,
                max_expr_length=self.args.max_expr_length,
                stream=True, num_proc=num_proc,
                skip=train_size, take=test_size,
                logger=self.logger
            )
            return train_dataset, test_dataset, train_size, test_size
        else:
            # 非流式模式：使用Subset索引
            if self.accelerator.is_local_main_process:
                print(f"[性能] 开始创建完整数据集...")

            dataset_start = time.time()
            full_dataset = prepare_dataset_hf(
                data_file=cache_filename, tokenizer=tokenizer,
                max_expr_length=self.args.max_expr_length,
                stream=False, num_proc=num_proc,
                logger=self.logger
            )
            dataset_time = time.time() - dataset_start

            if self.accelerator.is_local_main_process:
                print(f"[性能] Dataset 创建耗时: {dataset_time:.2f}秒")
                print(f"[性能] 开始创建训练/测试集索引 (total_size={self.args.num_samples})...")

            shuffle_start = time.time()
            from torch.utils.data import Subset
            indices = list(range(self.args.num_samples))
            np.random.shuffle(indices)
            shuffle_time = time.time() - shuffle_start

            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            if self.accelerator.is_local_main_process:
                print(f"[性能] 创建和打乱索引耗时: {shuffle_time:.2f}秒")
                print(f"非流式模式: 训练集 {len(train_indices)} 样本, 测试集 {len(test_indices)} 样本")

            train_dataset = Subset(full_dataset, train_indices)
            test_dataset = Subset(full_dataset, test_indices)
            return train_dataset, test_dataset, len(train_indices), len(test_indices)

    def _create_dataloaders(self, train_dataset, test_dataset):
        """创建训练和测试DataLoader"""
        import time

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

        dataloader_start = time.time()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=train_shuffle,
            num_workers=num_workers, collate_fn=custom_collate_fn,
            drop_last=train_drop_last, pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=test_drop_last
        )
        dataloader_time = time.time() - dataloader_start

        if self.accelerator.is_local_main_process:
            print(f"✓ DataLoader 创建完成 (耗时: {dataloader_time:.2f}秒)")

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

    # ============= 检查点管理 =============
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

    # ============= 训练接口（委托给 EditFlowTrainer）============
    def train(self):
        """训练模型 - 委托给 EditFlowTrainer"""
        checkpoint_path = find_latest_checkpoint(self.args)
        if self.accelerator.is_local_main_process:
            print(f"使用设备: {self.device}")
            print(f"{'找到检查点' if checkpoint_path else '未找到检查点，将从基础模型开始训练'}: {checkpoint_path or ''}")

        # 准备数据和模型
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

        # 创建训练器
        trainer = EditFlowTrainer(
            model=model,
            condition_encoder=condition_encoder,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            args=self.args,
            logger=self.logger,
            accelerator=self.accelerator
        )

        eval_every = self.args.eval_every

        # 训练循环
        for epoch in range(self.args.num_epochs):
            avg_loss, num_batches, total_loss, total_grad_norm = trainer.train_epoch(
                train_dataloader, train_dataset, epoch, "Mixed"
            )

            # 在所有进程上收集训练指标（使用合并后的方法）
            metrics = trainer.gather_and_format_metrics(num_batches, total_loss, total_grad_norm)

            # 只在主进程上打印和记录日志
            if self.accelerator.is_local_main_process:
                current_lr = optimizer.param_groups[0]['lr']

                if self.accelerator.num_processes > 1:
                    # 使用合并后的方法返回的指标
                    gpu_details = metrics['gpu_metrics']
                    global_total_batches = metrics['global_total_batches']
                    global_avg_loss = metrics['global_avg_loss']

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

            # 评估
            if (epoch + 1) % eval_every == 0 or epoch == self.args.num_epochs - 1:
                test_loss = trainer.evaluate(test_dataloader, test_dataset)
                if self.accelerator.is_local_main_process:
                    print(f"测试集损失: {test_loss:.4f}")
                    self.logger.log("EVALUATION", f"Epoch {epoch+1}/{self.args.num_epochs} | test_loss={test_loss:.4f}", level=1)

            # 保存检查点
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

    # ============= 推理接口（委托给 InferenceEngine）============
    def symbolic_regression(self, model_path, x_data, y_data, n_steps=100, input_dim=None, max_expr_length=None, initial_expr=None):
        """符号回归 - 委托给 InferenceEngine

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

        # 加载模型
        model, condition_encoder, _, _, _, tokenizer = self.setup_models(checkpoint_path=model_path)
        device = self.device

        # 准备输入数据
        x_values = torch.FloatTensor(x_data).unsqueeze(0).to(device)
        y_values = torch.FloatTensor(y_data).unsqueeze(0).to(device)

        if input_dim is None:
            input_dim = x_data.shape[1] if len(x_data.shape) > 1 else 1

        # 编码条件
        point_mask = torch.ones_like(y_values)
        condition = condition_encoder(x_values, y_values, point_mask)

        # 创建推理引擎
        inference_engine = InferenceEngine(
            model=model,
            condition_encoder=condition_encoder,
            tokenizer=tokenizer,
            args=self.args,
            logger=self.logger,
            device=device
        )

        # 执行推理
        return inference_engine.symbolic_regression(
            x_data=x_data,
            y_data=y_data,
            condition=condition,
            x_values=x_values,
            y_values=y_values,
            n_steps=n_steps,
            initial_expr=initial_expr
        )


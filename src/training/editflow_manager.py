"""
EditFlow迭代优化训练器 - 实现基于迭代式编辑操作的符号回归模型训练
使用 Hugging Face Accelerate 进行分布式训练加速
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

# from ..utils.special_tokens import SpecialTokensManager  # 已移除：使用小词表后不再需要
from ..symbolic.data_generator import generate_flow_samples, load_dimension_index
from .flow import (
    # KappaScheduler, sample_conditional_path,  # 已移除：不再需要时间调度器
    remove_gap_tokens, fill_gap_tokens_with_repeats,
    ContinuousFlowLoss, FlowDataset, custom_collate_fn
)
from ..modeling.condition_encoder import SetTransformerConditionEncoder
# 使用新的LLaMA EditFlow模型替代BERT
from ..modeling.llama_editflow import LlamaEditFlowBackbone
from ..utils.misc_utils import find_latest_checkpoint, load_checkpoint
from ..utils.logger import Logger
from .greedy_search import Candidate, SimpleSymbolicRegression


class EditFlowManager:
    """EditFlow模型管理器 - 支持训练和推理功能

    架构特点：迭代优化模式
    - 模型直接预测从z0到z1的编辑操作（插入、删除、替换）
    - 时间步固定为0，学习从起点到目标的直接编辑路径
    - 使用目标值y_target作为条件（而非残差），保持条件恒定作为"北极星"
    """

    # 类常量：训练和推理配置参数
    GRADIENT_CLIP_NORM = 5.0  # 提高到5.0以适应6750万参数的大模型
    NUMERICAL_CLIP_THRESHOLD = 1e6
    MAX_EXPRESSION_LENGTH = 50
    LEARNING_RATE_SCALE = 0.1  # 降低学习率以防止梯度爆炸
    MIN_ACTION_SCORE = 0.01  # 最小操作分数阈值

    def __init__(self, args):
        self.args = args

        # 初始化 Accelerate - 自动处理分布式训练设置
        # 注意：mixed_precision 由 accelerate launch 命令行参数控制
        # 不要在代码中硬编码，否则会覆盖命令行的 --mixed_precision=bf16 设置
        self.accelerator = Accelerator(
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            log_with=getattr(args, 'log_with', None)
        )

        # 设置随机种子
        set_seed(args.seed)

        # 初始化统一日志管理器
        self.logger = Logger(self.accelerator, enabled=True)

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
            print(f"LLaMA模型配置: hidden_dim={getattr(self.args, 'hidden_dim', 256)}, n_layers={getattr(self.args, 'n_layers', 6)}, n_heads={getattr(self.args, 'n_heads', 8)}")
            print(f"条件嵌入模型: {getattr(self.args, 'condition_model_name', 'N/A')}")
            print(f"梯度累积步数: {getattr(self.args, 'gradient_accumulation_steps', 'N/A')}")
            print(f"FP16混合精度: {getattr(self.args, 'use_fp16', 'N/A')}")

            print(f"\nAccelerate 初始化完成")
            print(f"  设备: {self.device}")
            print(f"  分布式训练: {self.accelerator.distributed_type}")
            print(f"  进程数: {self.accelerator.num_processes}")
            print(f"  混合精度: {self.accelerator.mixed_precision}")

        # 记录训练开始日志
        self.logger.training_start(self.args)

    def set_seed(self, seed: int):
        """设置随机种子 - 现在使用 Accelerate 的 set_seed"""
        set_seed(seed)

    def prepare_data(self, tokenizer):
        """准备训练数据，支持多进程并行生成"""

        # 1. 数据生成阶段：只使用主进程（单进程）
        cache_filename = f"data/flow_samples_{self.args.num_samples}_{self.args.max_dim}dim_{self.args.n_points}pts_{self.args.max_depth}depth_{self.args.max_expr_length}len.txt"

        # 只有主进程负责数据生成，避免NCCL通信问题
        if self.accelerator.is_local_main_process:
            print(f"准备连续流训练数据 (单进程生成模式)...")

            # 获取对齐方法配置
            alignment_method = getattr(self.args, 'alignment_method', 'randomized')
            print(f"使用对齐方法: {alignment_method}")

            # 调用数据生成函数
            generate_flow_samples(
                num_samples=self.args.num_samples,
                max_dim=self.args.max_dim,
                n_points=self.args.n_points,
                max_depth=self.args.max_depth,
                max_expr_length=self.args.max_expr_length,
                verbose=True,  # 显示详细日志
                alignment_method=alignment_method,
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

        # 使用符号回归专属的小词汇表分词器
        # 不再依赖BERT的大词汇表，使用自定义的紧凑词汇表
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

        if self.accelerator.is_local_main_process:
            print("初始化条件编码器...")
        condition_encoder = SetTransformerConditionEncoder(
            max_input_dim=getattr(self.args, 'condition_max_input_dim', 3),
            dim_hidden=getattr(self.args, 'condition_dim_hidden', 128),
            num_heads=getattr(self.args, 'condition_num_heads', 4),
            num_inds=getattr(self.args, 'condition_num_inds', 32),
            num_layers=getattr(self.args, 'condition_num_layers', 3),
            num_seeds=getattr(self.args, 'condition_num_seeds', 1),
            dim_output=getattr(self.args, 'condition_dim_output', 128),
            verbose=self.accelerator.is_local_main_process
        ).to(self.device)

        if self.accelerator.is_local_main_process:
            print("初始化LLaMA EditFlow模型（自定义架构，不加载预训练权重）...")

        # 获取条件编码器的隐藏层维度
        # 现在条件编码器输出 (batch_size, num_seeds, dim_hidden) 格式
        # 所以 condition_dim 应该等于 dim_hidden
        condition_hidden_dim = getattr(self.args, 'condition_dim_hidden', 128)

        # 直接实例化 LlamaEditFlowBackbone
        model = LlamaEditFlowBackbone(
            vocab_size=len(tokenizer.get_vocab()),  # 符号回归专用小词表
            hidden_dim=getattr(self.args, 'hidden_dim', 256),  # LLaMA隐藏层维度
            n_layers=getattr(self.args, 'n_layers', 6),  # Transformer层数
            n_heads=getattr(self.args, 'n_heads', 8),  # 注意力头数
            condition_dim=condition_hidden_dim,
            dropout=getattr(self.args, 'dropout', 0.1),
            max_seq_len=self.args.max_expr_length,
            use_condition_injection=getattr(self.args, 'use_condition_injection', True),
            verbose=self.accelerator.is_local_main_process
        ).to(self.device)

        # 创建优化器和损失函数
        criterion = ContinuousFlowLoss(scheduler_type='cubic')
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate * self.LEARNING_RATE_SCALE,
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
            print(f"✓ LLaMA EditFlow模型参数数量: {total_params:,}")

        return model, condition_encoder, criterion, optimizer, tokenizer

  
    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, debug_info=None):
        """
        修改后的前向传播：移除中间状态插值，直接预测从z0到z1的编辑操作
        这将模型从"连续流匹配"转变为"迭代优化"架构
        """
        batch_size = z0_token_ids.size(0)

        # 获取 vocab_size（从tokenizer获取，避免访问被DDP包装的model属性）
        vocab_size = dataset.tokenizer.vocab_size

        # 迭代优化模式：使用z0作为当前状态的输入（z0 -> z1的编辑操作）
        batch_size, seq_len = z0_token_ids.shape
        z0_probs = torch.zeros(batch_size, seq_len, vocab_size, device=z0_token_ids.device)
        z0_probs.scatter_(2, z0_token_ids.unsqueeze(-1), 1.0)

        # z1 token序列用于计算目标编辑操作
        z1_probs = torch.zeros(batch_size, seq_len, vocab_size, device=z1_token_ids.device)
        z1_probs.scatter_(2, z1_token_ids.unsqueeze(-1), 1.0)

        # 移除过度调试的z0_probs和z1_probs记录（第一个batch也不需要）
        # 只保留架构变更说明
        if debug_info and debug_info.get('is_first_batch', False) and self.accelerator.is_local_main_process:
            context = debug_info.get('context', '')
            self.logger.log("DIRECT_EDIT_PREDICTION",
                            f"直接编辑模式 | batch_size={batch_size} | t固定为0",
                            context, level=1)
            self.logger.log("ARCHITECTURE_CHANGE",
                            "从'连续流匹配'转变为'迭代优化'架构",
                            context, level=1)

        # 迭代优化模式：直接使用z0作为当前状态，不再进行时间插值
        # 移除gap token得到输入序列x_t（原始序列空间，无gap重复）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(
            z0_token_ids, dataset.tokenizer
        )

        # 调试：验证训练时的序列格式（仅验证第一个批次）
        if not hasattr(self, '_train_sequence_format_logged'):
            bos_token = dataset.tokenizer.convert_tokens_to_ids('<s>')
            sample_idx = 0
            if x_t[sample_idx, 0] == bos_token:
                if self.accelerator.is_local_main_process:
                    self.logger.log("TRAIN_SEQUENCE_FORMAT",
                                   f"训练序列格式验证 | x_t[{sample_idx}, 0:5]={x_t[sample_idx, :5].tolist()} | "
                                   f"BOS token在位置0={x_t[sample_idx, 0]}",
                                   "train", level=2)
                self._train_sequence_format_logged = True

        attention_mask = (~x_pad_mask).float()

        # 调用 LlamaEditFlowBackbone，返回字典格式
        output = model(
            input_ids=x_t, condition=condition_embeddings, attention_mask=attention_mask
        )

        # 合并三个速率为一个tensor（与旧接口保持一致）
        ins_rate, del_rate, sub_rate = output['rates']
        pred_rates = torch.cat([ins_rate, del_rate, sub_rate], dim=-1)

        return {
            'pred_rates': pred_rates,
            'pred_ins_probs': output['insert_probs'],
            'pred_sub_probs': output['substitute_probs'],
            'x_t': x_t,
            'z0': z0_token_ids,  # 当前状态（起点），用于损失计算
            'z1_token_ids': z1_token_ids,  # 目标状态（终点）
            'z_gap_mask': z_gap_mask,
            'z_pad_mask': z_pad_mask,
            'vocab_size': vocab_size,
        }

    def compute_loss(self, forward_results, criterion, dataset):
        pred_rates = forward_results['pred_rates']
        pred_ins_probs = forward_results['pred_ins_probs']
        pred_sub_probs = forward_results['pred_sub_probs']
        x_t = forward_results['x_t']
        z0 = forward_results['z0']  # 当前状态（起点）
        z1_token_ids = forward_results['z1_token_ids']  # 目标状态（终点）
        z_gap_mask = forward_results['z_gap_mask']
        z_pad_mask = forward_results['z_pad_mask']
        effective_vocab_size = forward_results['vocab_size']
        gap_token = dataset.tokenizer.convert_tokens_to_ids('<gap>')

        # 获取时间步采样数量
        num_timesteps = self.args.num_timesteps

        # 修复索引错位bug：模型输出顺序是 [ins_rate, del_rate, sub_rate]
        # 因此索引 0=插入, 1=删除, 2=替换
        lambda_ins = pred_rates[:, :, 0:1]  # 插入速率
        lambda_del = pred_rates[:, :, 1:2]  # 删除速率（修复：原来是 lambda_sub）
        lambda_sub = pred_rates[:, :, 2:3]  # 替换速率（修复：原来是 lambda_del）

        ins_probs = lambda_ins * pred_ins_probs
        sub_probs = lambda_sub * pred_sub_probs

        # u_cat_x 是 X 空间（原始序列空间）的预测速率
        # 形状: [batch, x_seq_len, 2*vocab_size+1]
        u_cat_x = torch.cat([ins_probs, sub_probs, lambda_del], dim=-1)

        # u_z 是 Z 空间（扩展空间，含gap重复）的预测速率
        # 形状: [batch, z_seq_len, 2*vocab_size+1]
        u_z = fill_gap_tokens_with_repeats(u_cat_x, z_gap_mask, z_pad_mask)

        # 生成编辑操作掩码：比较z0（当前）和z1（目标），找出需要编辑的位置
        u_mask = criterion.make_ut_mask_from_z(z0, z1_token_ids, effective_vocab_size, gap_token, dataset.tokenizer)

        # 关键修复：传入 u_cat_x (X空间) 用于正确的 u_total 计算
        # u_z 仍用于 cross_entropy 计算（监督带路径编辑操作）
        # 传入 logger 用于记录详细的损失统计信息
        loss = criterion(u_cat_x, u_z, u_mask, effective_vocab_size,
                        accelerator=self.accelerator, logger=self.logger)

        return loss

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0

        # 显示进度条 - 只在主进程显示
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}",
                          disable=not self.accelerator.is_local_main_process)

        # 只在主进程设置初始进度条显示
        if self.accelerator.is_local_main_process:
            progress_bar.set_postfix({'loss': '0.0000', 'grad_norm': '0.000'})

        for batch_idx, batch in enumerate(progress_bar):
            # 使用 Accelerate 的梯度累积上下文管理器
            # 自动处理梯度同步、累积步数判断、优化器更新
            with self.accelerator.accumulate([model, condition_encoder]):
                x_values = batch['x_values'].to(self.device)
                y_target = batch['y_target'].to(self.device)  # 修改：使用y_target而非residuals
                residuals = batch.get('residuals', torch.zeros_like(y_target)).to(self.device)  # 保留用于日志
                z0_token_ids = batch['z0_token_ids'].to(self.device)
                z1_token_ids = batch['z1_token_ids'].to(self.device)

                # 移除过度的token解码日志以提高性能

                point_mask = batch['point_mask'].to(self.device) if 'point_mask' in batch else None
                # 关键修改：使用y_target作为条件而非residuals（架构改进）
                condition_embeddings = condition_encoder(x_values, y_target, point_mask)

                # 移除每个batch的tensor记录以减少IO开销
                # 只在第一个batch记录一次用于验证
                if batch_idx == 0 and self.accelerator.is_local_main_process:
                    self.logger.tensor("condition_embeddings", condition_embeddings, level=2, context=f"维度{dimension}_batch0")
                    self.logger.tensor("x_values", x_values, level=2, context=f"维度{dimension}_batch0")
                    self.logger.tensor("residuals", residuals, level=2, context=f"维度{dimension}_batch0")

                # 准备调试信息
                debug_info = None
                if batch_idx == 0:
                    debug_info = {
                        'is_first_batch': True,
                        'context': f'维度{dimension}'
                    }

                forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, dataset, debug_info)

                # 移除每个batch的前向传播tensor记录
                # 只在第一个batch记录一次用于验证
                if batch_idx == 0 and self.accelerator.is_local_main_process:
                    self.logger.tensor("pred_rates", forward_results['pred_rates'], level=2, context=f"维度{dimension}_batch0")
                    self.logger.tensor("pred_ins_probs", forward_results['pred_ins_probs'], level=2, context=f"维度{dimension}_batch0")
                    self.logger.tensor("pred_sub_probs", forward_results['pred_sub_probs'], level=2, context=f"维度{dimension}_batch0")

                # 分布式健康检查：记录前向传播中的NaN（仅用于监控）
                if self.accelerator.distributed_type != "NO":
                    pred_rates = forward_results['pred_rates']

                    # 检查是否有任何进程的模型输出包含NaN
                    local_has_nan = torch.isnan(pred_rates).any().float()
                    gathered_nan_results = self.accelerator.gather(local_has_nan)
                    global_has_nan = gathered_nan_results.sum()

                    if global_has_nan.item() > 0:
                        if self.accelerator.is_local_main_process:
                            self.logger.error("FORWARD_NAN", f"维度{dimension} 检测到前向传播NaN", f"batch_idx:{batch_idx}")

                # ✅ 不再手动除以 gradient_accumulation_steps，accelerator.accumulate 会自动处理
                loss = self.compute_loss(forward_results, criterion, dataset)
                # 记录损失值（每个batch都记录）
                if self.accelerator.is_local_main_process:
                    self.logger.log("LOSS_COMPUTED", f"loss={loss.item():.6f}", f"维度{dimension}_batch{batch_idx}", level=2)

                grad_norm = 0.0
                # 使用 Accelerate 的 backward 而不是直接调用 loss.backward()
                self.accelerator.backward(loss)

                # 记录梯度统计信息（每个batch都记录，用于调试NaN来源）
                # 不再需要判断是否是最后一步，因为 accumulate 会自动处理
                all_params = list(model.parameters()) + list(condition_encoder.parameters())

                # 只在每10个batch或第一个batch记录详细梯度统计，减少日志量
                if batch_idx % 10 == 0 and self.accelerator.is_local_main_process:
                    grad_max = 0.0
                    grad_min = float('inf')
                    grad_mean = 0.0
                    grad_std = 0.0
                    grad_has_nan = False
                    grad_has_inf = False
                    grad_num_zero = 0
                    grad_num_params = 0

                    for param in all_params:
                        if param.grad is not None:
                            grad_num_params += 1
                            grad_data = param.grad.data

                            # 检查NaN和Inf
                            if torch.isnan(grad_data).any():
                                grad_has_nan = True
                            if torch.isinf(grad_data).any():
                                grad_has_inf = True

                            # 统计梯度值
                            grad_abs = grad_data.abs()
                            grad_max = max(grad_max, float(grad_abs.max().item()))
                            grad_min = min(grad_min, float(grad_abs.min().item()))
                            grad_mean += float(grad_abs.mean().item())
                            grad_std += float(grad_abs.std().item())
                            grad_num_zero += int((grad_data == 0).sum().item())

                    # 计算平均值
                    if grad_num_params > 0:
                        grad_mean = float(grad_mean / grad_num_params)
                        grad_std = float(grad_std / grad_num_params)

                    # 记录详细的梯度统计
                    self.logger.log("GRAD_STATS",
                                    f"params={grad_num_params} | max={grad_max:.6f} min={grad_min:.6f} "
                                    f"mean={grad_mean:.6f} std={grad_std:.6f} | zeros={grad_num_zero} "
                                    f"has_nan={grad_has_nan} has_inf={grad_has_inf}",
                                    f"维度{dimension}_batch{batch_idx}", level=2)

                # 使用Accelerate的梯度裁剪（会自动处理混合精度）
                # ⚠️ 关键修复：只在梯度完全同步时才执行裁剪和优化器更新
                # 这确保在梯度累积期间不会在未同步的梯度上进行操作
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(all_params, self.GRADIENT_CLIP_NORM)

                    # 记录梯度范数和参数统计（只在每10个batch记录）
                    if batch_idx % 10 == 0 and self.accelerator.is_local_main_process:
                        # 统计参数范数
                        param_norm = 0.0
                        param_max = 0.0
                        param_mean = 0.0
                        param_count = 0
                        for param in all_params:
                            if param.data is not None:
                                param_count += 1
                                param_norm += float(param.data.norm().item() ** 2)
                                param_max = max(param_max, float(param.data.abs().max().item()))
                                param_mean += float(param.data.abs().mean().item())
                        param_norm = float(param_norm ** 0.5)
                        param_mean = float(param_mean / param_count if param_count > 0 else 0.0)

                        self.logger.log("GRAD_NORM",
                                        f"grad_norm={grad_norm:.4f} | param_norm={param_norm:.4f} "
                                        f"param_max={param_max:.4f} param_mean={param_mean:.4f}",
                                        f"维度{dimension}_batch{batch_idx}", level=2)

                    # ✅ 只在梯度同步时更新参数
                    optimizer.step()

                    # 记录优化器状态（学习率等）
                    if self.accelerator.is_local_main_process and batch_idx % 10 == 0:
                        current_lr = float(optimizer.param_groups[0]['lr'])
                        self.logger.log("OPTIMIZER_STATE", f"lr={current_lr:.6f}",
                                        f"维度{dimension}_batch{batch_idx}", level=2)

                    optimizer.zero_grad()
                else:
                    # 梯度累积期间：不执行优化器更新，保持 grad_norm 为 0
                    grad_norm = 0.0

                total_loss += loss.item()
                num_batches += 1

                # 更新进度条显示（每个batch都更新）
                if self.accelerator.is_local_main_process:
                    postfix_dict = {
                        'loss': f'{loss.item():.4f}',
                        'grad_norm': f'{grad_norm:.3f}' if isinstance(grad_norm, (int, float)) else f'{grad_norm:.3f}'
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

        with torch.no_grad():
            # === 修改：不再循环 dim，直接遍历 dataloader ===
            for batch in test_dataloader:
                x_values = batch['x_values'].to(self.device)
                y_target = batch['y_target'].to(self.device)  # 修改：使用y_target
                z0_token_ids = batch['z0_token_ids'].to(self.device)
                z1_token_ids = batch['z1_token_ids'].to(self.device)
                point_mask = batch['point_mask'].to(self.device)

                # 修改：使用y_target而非residuals
                condition_embeddings = condition_encoder(x_values, y_target, point_mask)
                forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, test_dataset)

                # 计算损失
                loss = self.compute_loss(forward_results, criterion, test_dataset)
                total_loss += loss.item()
                num_batches += 1

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


    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, is_final=False):
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

            # 从 model 中提取配置信息
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
            self.logger.log("TRAINING_START", f"开始训练 | num_epochs={self.args.num_epochs} | model_params={model_params:,} | encoder_params={encoder_params:,}", level=1)

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
                self.logger.log("EPOCH_COMPLETE", f"Epoch {epoch+1}/{self.args.num_epochs} | train_loss={avg_loss:.4f} | batches={num_batches}", level=1)

            # 修改 evaluate 调用，传入单个 dataloader
            if (epoch + 1) % eval_every == 0 or epoch == self.args.num_epochs - 1:
                test_loss = self.evaluate(model, condition_encoder, criterion, test_dataloader, test_dataset)
                if self.accelerator.is_local_main_process:
                    print(f"测试集损失: {test_loss:.4f}")
                    # 记录评估结果到 training.log
                    self.logger.log("EVALUATION", f"Epoch {epoch+1}/{self.args.num_epochs} | test_loss={test_loss:.4f}", level=1)

            # 保存检查点
            if (epoch + 1) % self.args.save_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch
                )
                if self.accelerator.is_local_main_process:
                    print(f"检查点已保存到: {checkpoint_path}")
                    # 记录检查点保存到 training.log
                    self.logger.log("CHECKPOINT_SAVED", f"Epoch {epoch+1}/{self.args.num_epochs} | path={checkpoint_path} | train_loss={avg_loss:.4f}", level=1)

        # 保存最终模型
        final_path = self.save_checkpoint(
            model, condition_encoder, optimizer, avg_loss, self.args.num_epochs - 1, is_final=True
        )
        if self.accelerator.is_local_main_process:
            print(f"最终模型已保存到: {final_path}")
            # 记录训练完成到 training.log
            self.logger.log("TRAINING_COMPLETE", f"训练完成 | final_path={final_path} | final_train_loss={avg_loss:.4f} | total_epochs={self.args.num_epochs}", level=1)

        return model, condition_encoder

    def symbolic_regression(self, model_path, x_data, y_data, n_steps=100, input_dim=None, max_expr_length=None):
        """符号回归 - 使用简单推理(贪婪搜索)接收数据点对，输出表达式

        Args:
            model_path: 模型检查点路径
            x_data: 输入x数据
            y_data: 目标y数据
            n_steps: 推理步数
            input_dim: 输入维度，如果为None则自动推断
            max_expr_length: 表达式最大token长度，如果为None则使用args中的值
        """
        # 记录开始
        self.logger.log("SYMBOLIC_REGRESSION_START",
                       f"输入数据: x形状={x_data.shape}, y形状={y_data.shape} | n_steps={n_steps}",
                       "inference", level=1)

        # 加载模型
        checkpoint_path = model_path if model_path and os.path.exists(model_path) else None
        if checkpoint_path:
            self.logger.log("MODEL_LOAD", f"使用检查点: {checkpoint_path}", "inference", level=1)
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
            self.logger.log("MODEL_LOAD", "⚠️ 未找到检查点，使用随机初始化权重（警告：推理质量会很差）", "inference", level=1)

        model, condition_encoder, _, _, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

        # 设置设备和模式
        device = self.device
        model.eval()
        condition_encoder.eval()

        # 准备输入数据
        x_values = torch.FloatTensor(x_data).unsqueeze(0).to(device)
        y_values = torch.FloatTensor(y_data).unsqueeze(0).to(device)  # 这是目标值

        # 推断输入维度并生成初始表达式
        if input_dim is None:
            input_dim = x_data.shape[1] if len(x_data.shape) > 1 else 1

        # 计算初始残差 (真实值 - 初始表达式的预测值)
        import sympy as sp
        from ..symbolic.symbolic_utils import evaluate_expression_safe, evaluate_expression_with_constants, tree_to_expr

        # initial_expr = sum(sp.Symbol(f'x{i}') for i in range(input_dim))
        initial_expr = sp.Symbol('x0')

        # 计算初始表达式在x_data上的预测值
        success, y_pred = evaluate_expression_safe(initial_expr, x_data)
        if not success:
            self.logger.error("INITIAL_EXPR_FAILED", f"无法计算初始表达式 '{initial_expr}' 的预测值", "inference")
            return ""

        # 计算残差：真实值 - 预测值（仅用于评估，不作为条件）
        residuals = y_values - torch.FloatTensor(y_pred).unsqueeze(0).to(device)

        # 关键修改：使用目标值y_values作为条件，而非残差
        # 这样条件在推理过程中保持恒定，作为"北极星"指引方向
        point_mask = torch.ones_like(y_values)
        condition = condition_encoder(x_values, y_values, point_mask)

        # 记录初始数据
        self.logger.log("INITIAL_DATA",
                       f"x_values: shape={x_values.shape} range=[{x_values.min():.4f},{x_values.max():.4f}] | "
                       f"y_target: shape={y_values.shape} range=[{y_values.min():.4f},{y_values.max():.4f}] | "
                       f"residuals: shape={residuals.shape} range=[{residuals.min():.4f},{residuals.max():.4f}]",
                       "inference", level=1)
        self.logger.log("ARCHITECTURE_INFO",
                       "使用目标值y_target作为条件（架构改进：北极星模式）",
                       "inference", level=1)
        self.logger.log("INITIAL_CONDITION",
                       f"condition: shape={condition.shape} range=[{condition.min():.4f},{condition.max():.4f}]",
                       "inference", level=1)

        # 打印条件嵌入的前10个维度的具体值
        condition_cpu = condition.cpu().squeeze(0)
        condition_values = condition_cpu.detach().numpy()
        # 处理序列格式 (num_seeds, dim_hidden) 或向量格式 (dim_hidden,)
        if condition_values.ndim == 2:
            condition_preview = condition_values.flatten()[:10]  # 展平后取前10个
        else:
            condition_preview = condition_values[:10]
        self.logger.log("INITIAL_CONDITION_VALUES",
                       f"condition前10维: [{', '.join([f'{float(v):.6f}' for v in condition_preview])}]",
                       "inference", level=1)

        self.logger.tensor("initial_condition", condition, level=1, context="inference")
        self.logger.tensor("initial_residuals", residuals, level=1, context="inference")

        # 构建初始前缀表达式（与训练格式一致）
        # 统一处理：对于n维，需要(n-1)个add + n个变量
        # 例如：dim=1 -> ['x0']；dim=2 -> ['add','x0','x1']；dim=3 -> ['add','add','x0','x1','x2']
        # current_tokens = ['add'] * (input_dim - 1) + [f'x{i}' for i in range(input_dim)]
        # 使用 x0 - x1 作为初始表达式
        current_tokens = ['x0']

        # 创建简单推理器
        self.logger.log("SIMPLE_SEARCH_INIT", f"初始化简单推理器 | n_steps={n_steps}", "inference", level=1)

        simple_searcher = SimpleSymbolicRegression(
            model=model,
            condition_encoder=condition_encoder,
            tokenizer=tokenizer,
            # special_tokens_manager=special_tokens_manager,  # 已移除：使用小词表后不再需要
            device=device,
            args=self.args,
            logger=self.logger,
            min_action_score=self.MIN_ACTION_SCORE,
            max_expression_length=self.MAX_EXPRESSION_LENGTH,
            numerical_clip_threshold=self.NUMERICAL_CLIP_THRESHOLD
        )

        # 执行贪婪搜索
        initial_residuals_np = residuals.cpu().squeeze(0).numpy()
        best_candidate = simple_searcher.greedy_search(
            initial_tokens=current_tokens,
            initial_condition=condition,
            initial_residuals=initial_residuals_np,
            x_data=x_data,
            y_data=y_data,
            x_values=x_values,
            n_steps=n_steps
        )

        # 返回最佳候选的表达式
        final_expression = ','.join(best_candidate.tokens) if best_candidate and best_candidate.tokens else ""

        if best_candidate and self.accelerator.is_local_main_process:
            # 记录MSE分数
            mse_score = getattr(best_candidate, 'mse_score', None)
            mse_str = f'{mse_score:.6f}' if mse_score is not None else 'N/A'
            self.logger.log("SIMPLE_SEARCH_RESULT",
                           f"MSE分数: {mse_str} | "
                           f"操作历史: {' -> '.join(best_candidate.history[-5:]) if best_candidate.history else 'N/A'}",
                           "inference", level=1)

        self.logger.log("INFERENCE_COMPLETE", f"最终表达式: {final_expression}", "inference", level=1)
        return final_expression


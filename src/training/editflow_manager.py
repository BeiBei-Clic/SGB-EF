"""
EditFlow迭代优化训练器 - 实现基于迭代式编辑操作的符号回归模型训练
使用 Hugging Face Accelerate 进行分布式训练加速
"""

import os
import time
import sympy as sp

import torch
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
    GRADIENT_CLIP_NORM = 10.0  # 提高到10.0，避免过度裁剪
    NUMERICAL_CLIP_THRESHOLD = 1e6
    MAX_EXPRESSION_LENGTH = 50
    MIN_ACTION_SCORE = 0.01  # 最小操作分数阈值

    def __init__(self, args):
        self.args = args

        # 初始化 Accelerate - 自动处理分布式训练设置
        # 注意：mixed_precision 由 accelerate launch 命令行参数控制
        # 不要在代码中硬编码，否则会覆盖命令行的 --mixed_precision=bf16 设置
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.log_with
        )

        # 设置随机种子
        set_seed(args.seed)

        # 保存debug模式标志
        self.debug_mode = args.debug

        # 初始化统一日志管理器，传入debug_mode参数
        self.logger = Logger(self.accelerator, enabled=True, debug_mode=self.debug_mode)

        # 设备信息
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

        # 记录训练开始日志
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

    def set_seed(self, seed: int):
        """设置随机种子 - 现在使用 Accelerate 的 set_seed"""
        set_seed(seed)

    def prepare_data(self, tokenizer):
        """准备训练数据，使用 Hugging Face datasets 加载"""

        # 1. 数据生成阶段：只使用主进程（单进程）
        cache_filename = f"data/flow_samples_{self.args.num_samples}_{self.args.max_dim}dim_{self.args.n_points}pts_{self.args.max_depth}depth_{self.args.max_expr_length}len.txt"

        # 只有主进程负责数据生成，避免NCCL通信问题
        if self.accelerator.is_local_main_process:
            print(f"准备连续流训练数据 (单进程生成模式)...")

            # 获取对齐方法配置
            print(f"使用对齐方法: {self.args.alignment_method}")

            # 调用数据生成函数
            generate_flow_samples(
                num_samples=self.args.num_samples,
                max_dim=self.args.max_dim,
                n_points=self.args.n_points,
                max_depth=self.args.max_depth,
                max_expr_length=self.args.max_expr_length,
                verbose=True,  # 显示详细日志
                alignment_method=self.args.alignment_method,
            )
        else:
            # 非主进程跳过数据生成，等待主进程完成
            print(f"[Rank {self.accelerator.process_index}] 跳过数据生成，等待主进程完成...")

        # 2. 同步屏障：等待主进程完成数据生成
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            print("[主进程] 数据生成完成，开始加载训练数据")

        # 3. 同步屏障：确保所有进程都能访问到完整的数据文件
        print(f"[Rank {self.accelerator.process_index}] 准备开始训练阶段...")
        self.accelerator.wait_for_everyone()

        # 4. 使用 Hugging Face datasets 加载数据
        # 使用命令行参数控制数据加载模式
        use_stream = self.args.dataset_stream
        num_proc = self.args.dataset_num_proc

        if self.accelerator.is_local_main_process:
            print(f"使用 Hugging Face datasets 加载数据 (stream={use_stream})...")

        # 加载完整数据集（train和test将通过train_test_split分割）
        full_dataset = FlowDataset(
            data_file=cache_filename,
            tokenizer=tokenizer,
            max_dim=self.args.max_dim,
            max_expr_length=self.args.max_expr_length,
            stream=use_stream,
            num_proc=num_proc
        )

        # 5. 分割训练集和测试集
        # 注意：流式模式下无法直接使用train_test_split，需要手动处理
        if use_stream:
            # 流式模式：使用迭代器分割（近似）
            # 先计算分割点
            split_ratio = 1 - self.args.test_split
            train_size = int(self.args.num_samples * split_ratio)
            test_size = self.args.num_samples - train_size

            if self.accelerator.is_local_main_process:
                print(f"流式模式: 训练集约 {train_size} 样本, 测试集约 {test_size} 样本")

            # 创建两个数据集实例（通过跳过不同的行数实现）
            # 注意：这种方式不够精确，但流式模式下无法预先知道确切数量
            train_dataset = full_dataset
            # 对于测试集，我们可以创建一个新的实例，但需要在迭代时跳过训练样本
            # 简化处理：这里暂时使用相同的数据集，实际训练时通过采样控制
            test_dataset = full_dataset  # 简化处理

            train_size_estimate = train_size
            test_size_estimate = test_size
        else:
            # 非流式模式：可以精确分割
            total_size = len(full_dataset)
            train_size = int(total_size * (1 - self.args.test_split))

            # 手动分割列表
            from torch.utils.data import Subset

            # 生成索引并打乱
            indices = list(range(total_size))
            np.random.shuffle(indices)

            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            # 如果只有1个样本，让它既用于训练又用于测试
            if total_size == 1:
                if self.accelerator.is_local_main_process:
                    print("警告: 只有1个样本，将同时用于训练和测试")
                train_dataset = full_dataset
                test_dataset = full_dataset
                train_size_estimate = 1
                test_size_estimate = 1
            else:
                train_dataset = Subset(full_dataset, train_indices)
                test_dataset = Subset(full_dataset, test_indices)
                train_size_estimate = len(train_indices)
                test_size_estimate = len(test_indices)

            if self.accelerator.is_local_main_process:
                print(f"非流式模式: 训练集 {train_size_estimate} 样本, 测试集 {test_size_estimate} 样本")

        # 6. 创建 DataLoader

        # 检查是否为stream模式
        is_stream_mode = getattr(train_dataset, 'stream', False)

        # 获取数据集大小，智能调整drop_last
        train_size = len(train_dataset)
        test_size = len(test_dataset)

        # 对于小数据集，禁用drop_last以免所有数据都被丢弃
        train_drop_last = train_size >= self.args.batch_size
        test_drop_last = test_size >= self.args.batch_size

        if self.accelerator.is_local_main_process:
            if not train_drop_last:
                print(f"警告: 训练集大小({train_size}) < batch_size({self.args.batch_size})，禁用drop_last")
            if not test_drop_last:
                print(f"警告: 测试集大小({test_size}) < batch_size({self.args.batch_size})，禁用drop_last")

        # 根据stream模式确定DataLoader参数
        train_shuffle = not is_stream_mode
        num_workers = 0 if is_stream_mode else self.accelerator.num_processes

        # 创建DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            drop_last=train_drop_last,
            pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            drop_last=test_drop_last
        )

        # 使用 Accelerate 准备
        train_dataloader, test_dataloader = self.accelerator.prepare(
            train_dataloader, test_dataloader
        )

        if self.accelerator.is_local_main_process:
            print(f"数据准备完成: 训练集约 {train_size_estimate} 样本, 测试集约 {test_size_estimate} 样本")

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
            max_input_dim=self.args.condition_max_input_dim,
            dim_hidden=self.args.condition_dim_hidden,
            num_heads=self.args.condition_num_heads,
            num_inds=self.args.condition_num_inds,
            num_layers=self.args.condition_num_layers,
            num_seeds=self.args.condition_num_seeds,
            dim_output=self.args.condition_dim_output,
            verbose=self.accelerator.is_local_main_process
        ).to(self.device)

        if self.accelerator.is_local_main_process:
            print("初始化LLaMA EditFlow模型（自定义架构，不加载预训练权重）...")

        # 获取条件编码器的隐藏层维度
        # 现在条件编码器输出 (batch_size, num_seeds, dim_hidden) 格式
        # 所以 condition_dim 应该等于 dim_hidden
        condition_hidden_dim = self.args.condition_dim_hidden

        # 直接实例化 LlamaEditFlowBackbone
        model = LlamaEditFlowBackbone(
            vocab_size=len(tokenizer.get_vocab()),  # 符号回归专用小词表
            hidden_dim=self.args.hidden_dim,  # LLaMA隐藏层维度
            n_layers=self.args.n_layers,  # Transformer层数
            n_heads=self.args.n_heads,  # 注意力头数
            condition_dim=condition_hidden_dim,
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
            eps=1e-8,  # 增加数值稳定性
            betas=(0.9, 0.999)  # 使用默认的beta值
        )

        # 添加学习率调度器（余弦退火）
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.args.num_epochs,
            eta_min=1e-6  # 最小学习率
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

        # 保存tokenizer引用供后续使用
        self.tokenizer = tokenizer

        return model, condition_encoder, criterion, optimizer, scheduler, tokenizer

  
    def forward_pass(self, model, condition_embeddings, z0_token_ids, z1_token_ids, debug_info=None):
        """
        修改后的前向传播：移除中间状态插值，直接预测从z0到z1的编辑操作
        这将模型从"连续流匹配"转变为"迭代优化"架构
        """
        batch_size = z0_token_ids.size(0)

        # 获取 vocab_size（使用self.tokenizer，避免Subset对象没有tokenizer属性的问题）
        vocab_size = self.tokenizer.vocab_size

        # 迭代优化模式：使用z0作为当前状态的输入（z0 -> z1的编辑操作）
        batch_size, seq_len = z0_token_ids.shape
        z0_probs = torch.zeros(batch_size, seq_len, vocab_size, device=z0_token_ids.device)
        z0_probs.scatter_(2, z0_token_ids.unsqueeze(-1), 1.0)

        # z1 token序列用于计算目标编辑操作
        z1_probs = torch.zeros(batch_size, seq_len, vocab_size, device=z1_token_ids.device)
        z1_probs.scatter_(2, z1_token_ids.unsqueeze(-1), 1.0)

        # 记录输入变量的完整值（仅在debug模式下记录）
        if debug_info and self.accelerator.is_local_main_process and self.debug_mode:
            context = debug_info.get('context', '')
            batch_idx = debug_info.get('batch_idx', 0)
            sample_idx = 0

            # 记录第一个样本的token序列（完整值）
            self.logger.tensor_values(f"z0_token_ids_batch{batch_idx}", z0_token_ids[sample_idx],
                                     context=context, level=2, max_elements=50)
            self.logger.tensor_values(f"z1_token_ids_batch{batch_idx}", z1_token_ids[sample_idx],
                                     context=context, level=2, max_elements=50)

            # 记录condition_embeddings（显示完整值）
            self.logger.tensor_values(f"condition_embeddings_batch{batch_idx}", condition_embeddings[sample_idx],
                                     context=context, level=2, max_elements=100)

        # 迭代优化模式：直接使用z0作为当前状态，不再进行时间插值
        # 移除gap token得到输入序列x_t（原始序列空间，无gap重复）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(
            z0_token_ids, self.tokenizer
        )

        # 记录x_t的完整值（仅在debug模式下记录）
        if debug_info and self.accelerator.is_local_main_process and self.debug_mode:
            context = debug_info.get('context', '')
            batch_idx = debug_info.get('batch_idx', 0)
            self.logger.tensor_values(f"x_t_batch{batch_idx}", x_t[0],
                                     context=context, level=2, max_elements=50)

        attention_mask = (~x_pad_mask).float()

        # 记录attention_mask的完整值（仅在debug模式下记录）
        if debug_info and self.accelerator.is_local_main_process and self.debug_mode:
            context = debug_info.get('context', '')
            batch_idx = debug_info.get('batch_idx', 0)
            self.logger.tensor_values(f"attention_mask_batch{batch_idx}", attention_mask[0],
                                     context=context, level=2, max_elements=50)

        # 调用 LlamaEditFlowBackbone，返回字典格式
        output = model(
            input_ids=x_t, condition=condition_embeddings, attention_mask=attention_mask
        )

        # 合并四个速率为一个tensor（添加KEEP操作）
        ins_rate, del_rate, sub_rate, keep_rate = output['rates']
        pred_rates = torch.cat([ins_rate, del_rate, sub_rate, keep_rate], dim=-1)

        # 记录模型输出的完整值（仅在debug模式下记录）
        if debug_info and self.accelerator.is_local_main_process and self.debug_mode:
            context = debug_info.get('context', '')
            batch_idx = debug_info.get('batch_idx', 0)
            sample_idx = 0

            # 记录第一个样本的pred_rates完整值
            self.logger.tensor_values(f"pred_rates_batch{batch_idx}", pred_rates[sample_idx],
                                     context=context, level=2, max_elements=100)

            # 记录第一个样本的insert_logits和substitute_logits完整值
            self.logger.tensor_values(f"insert_logits_batch{batch_idx}", output['insert_logits'][sample_idx],
                                     context=context, level=2, max_elements=100)
            self.logger.tensor_values(f"substitute_logits_batch{batch_idx}", output['substitute_logits'][sample_idx],
                                     context=context, level=2, max_elements=100)

        return {
            'pred_rates': pred_rates,
            'pred_ins_logits': output['insert_logits'],  # 返回logits而不是概率
            'pred_sub_logits': output['substitute_logits'],  # 返回logits而不是概率
            'x_t': x_t,
            'z0': z0_token_ids,
            'z1_token_ids': z1_token_ids,
            'z_gap_mask': z_gap_mask,
            'z_pad_mask': z_pad_mask,
            'attention_mask': attention_mask,  # 返回attention_mask用于屏蔽无效位置
            'vocab_size': vocab_size,
        }

    def compute_loss(self, forward_results, criterion, debug_info=None):
        pred_rates = forward_results['pred_rates']
        x_t = forward_results['x_t']
        z0 = forward_results['z0']  # 当前状态（起点）
        z1_token_ids = forward_results['z1_token_ids']  # 目标状态（终点）
        z_gap_mask = forward_results['z_gap_mask']
        z_pad_mask = forward_results['z_pad_mask']
        effective_vocab_size = forward_results['vocab_size']
        gap_token = self.tokenizer.convert_tokens_to_ids('<gap>')

        # 修复索引错位bug：模型输出顺序是 [ins_rate, del_rate, sub_rate, keep_rate]
        # 因此索引 0=插入, 1=删除, 2=替换, 3=保持
        lambda_ins = pred_rates[:, :, 0:1]  # 插入速率
        lambda_del = pred_rates[:, :, 1:2]  # 删除速率
        lambda_sub = pred_rates[:, :, 2:3]  # 替换速率
        lambda_keep = pred_rates[:, :, 3:4]  # 保持速率

        # 关键修复：直接使用logits而不是概率
        # 将速率作为log空间加到logits上
        ins_logits = forward_results['pred_ins_logits'] + torch.log(lambda_ins.clamp(min=1e-8))
        sub_logits = forward_results['pred_sub_logits'] + torch.log(lambda_sub.clamp(min=1e-8))
        del_logits = torch.log(lambda_del.clamp(min=1e-8))
        keep_logits = torch.log(lambda_keep.clamp(min=1e-8))  # KEEP操作的logits

        # 🔧 修复：不再在这里应用attention_mask
        # attention_mask会在loss计算时通过u_mask自然处理
        # 避免在这里设置-1e9导致log_softmax后出现-inf污染

        # u_cat_x 现在是logits而不是概率
        # 形状: [batch, x_seq_len, 2*vocab_size+2] (添加KEEP操作)
        # 🔧 修复：调整顺序以匹配u_mask的维度定义和rates_head的输出顺序
        # 统一顺序：[INS(vocab_size) | DEL(1) | SUB(vocab_size) | KEEP(1)]
        #           位置: 0~v-1          位置: v   位置: v+1~2v      位置: 2v+1(-1)
        # 其中 v = vocab_size
        # 拼接顺序：ins | del | sub | keep (与rates_head的[ins, del, sub, keep]一致)
        u_cat_x = torch.cat([ins_logits, del_logits, sub_logits, keep_logits], dim=-1)

        # u_z 是 Z 空间（扩展空间，含gap重复）的预测速率
        # 形状: [batch, z_seq_len, 2*vocab_size+2] (添加KEEP操作)
        u_z = fill_gap_tokens_with_repeats(u_cat_x, z_gap_mask, z_pad_mask)

        # 生成编辑操作掩码：使用双索引追踪逻辑
        # 在Z空间（z0）遍历，映射到X空间（x_t）的编辑操作
        # u_mask在X空间生成: [batch, x_seq_len, 2*vocab_size+1]
        u_mask_x = criterion.make_ut_mask_from_z(z0, z1_token_ids, effective_vocab_size, gap_token, self.tokenizer, x_t)

        # ⚠️ 关键修复：将u_mask扩展到Z空间以匹配log_u_z的维度
        # 使用fill_gap_tokens_with_repeats将X空间的mask扩展到Z空间
        u_mask = fill_gap_tokens_with_repeats(u_mask_x, z_gap_mask, z_pad_mask)

        # 记录损失计算中的关键变量值（仅在debug模式下记录）
        if self.accelerator.is_local_main_process and self.debug_mode:
            context = debug_info.get('context', '') if debug_info else ''
            batch_idx = debug_info.get('batch_idx', 0) if debug_info else 0
            sample_idx = 0

            # 记录标准答案：z0、z1和x_t的token序列
            self.logger.tensor_values(f"GT_z0_batch{batch_idx}", z0[sample_idx],
                                     context=context, level=2, max_elements=50)
            self.logger.tensor_values(f"GT_z1_batch{batch_idx}", z1_token_ids[sample_idx],
                                     context=context, level=2, max_elements=50)
            self.logger.tensor_values(f"GT_x_t_batch{batch_idx}", x_t[sample_idx],
                                     context=context, level=2, max_elements=50)

            # 记录分解后的速率（模型预测）
            self.logger.tensor_values(f"pred_lambda_ins_batch{batch_idx}", lambda_ins[sample_idx],
                                     context=context, level=2, max_elements=50)
            self.logger.tensor_values(f"pred_lambda_del_batch{batch_idx}", lambda_del[sample_idx],
                                     context=context, level=2, max_elements=50)
            self.logger.tensor_values(f"pred_lambda_sub_batch{batch_idx}", lambda_sub[sample_idx],
                                     context=context, level=2, max_elements=50)

            # 记录标准答案：u_mask（真实编辑操作标签，one-hot编码）
            # 使用专门的日志方法按语义拆分记录
            self.logger.log_u_mask_split(f"GT_u_mask", u_mask[sample_idx:sample_idx+1], x_t[sample_idx:sample_idx+1],
                                        effective_vocab_size, context=context, level=2)

            # 解码并记录Ground Truth编辑操作（可读格式，使用ID）
            self.logger.log_edit_operations(
                u_mask[sample_idx],
                x_t[sample_idx],
                effective_vocab_size,
                context=context,
                level=2,
                max_ops=20
            )

            # 记录模型预测的u_cat_x（用于对比）
            self.logger.tensor_values(f"pred_u_cat_x_batch{batch_idx}_first5pos", u_cat_x[sample_idx, :5, :],
                                     context=context, level=2, max_elements=100)

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
            batch_start_time = time.time()

            # 使用 Accelerate 的梯度累积上下文管理器
            # 自动处理梯度同步、累积步数判断、优化器更新
            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("BATCH_START", f"开始处理 Batch {batch_idx} | timestamp={time.time():.2f}",
                                f"维度{dimension}_batch{batch_idx}", level=2)

            with self.accelerator.accumulate([model, condition_encoder]):
                data_load_start = time.time()
                x_values = batch['x_values'].to(self.device)
                y_target = batch['y_target'].to(self.device)  # 修改：使用y_target而非residuals
                z0_token_ids = batch['z0_token_ids'].to(self.device)
                z1_token_ids = batch['z1_token_ids'].to(self.device)
                data_load_time = time.time() - data_load_start

                if self.accelerator.is_local_main_process and self.debug_mode:
                    self.logger.log("DATA_LOAD", f"数据加载完成 | 耗时={data_load_time:.3f}s",
                                    f"维度{dimension}_batch{batch_idx}", level=2)

                # 移除过度的token解码日志以提高性能

                point_mask = batch['point_mask'].to(self.device) if 'point_mask' in batch else None

                condition_start = time.time()
                # 关键修改：使用y_target作为条件而非residuals（架构改进）
                condition_embeddings = condition_encoder(x_values, y_target, point_mask)
                condition_time = time.time() - condition_start

                if self.accelerator.is_local_main_process and self.debug_mode:
                    self.logger.log("CONDITION_ENCODE", f"条件编码完成 | 耗时={condition_time:.3f}s",
                                    f"维度{dimension}_batch{batch_idx}", level=2)

                # 记录输入数据的完整值（仅在debug模式下记录）
                if self.accelerator.is_local_main_process and self.debug_mode:
                    context = f'维度{dimension}'
                    self.logger.tensor_values(f"x_values_batch{batch_idx}", x_values[0],
                                             context=context, level=2, max_elements=50)
                    self.logger.tensor_values(f"y_target_batch{batch_idx}", y_target[0],
                                             context=context, level=2, max_elements=50)

                # 准备调试信息（每个batch都传递）
                debug_info = {
                    'batch_idx': batch_idx,
                    'context': f'维度{dimension}'
                }

                forward_start = time.time()
                forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids, debug_info)
                forward_time = time.time() - forward_start

                if self.accelerator.is_local_main_process and self.debug_mode:
                    self.logger.log("FORWARD_PASS", f"前向传播完成 | 耗时={forward_time:.3f}s",
                                    f"维度{dimension}_batch{batch_idx}", level=2)

                # 分布式健康检查：记录前向传播中的NaN（仅用于监控）
                nan_check_start = time.time()
                if self.accelerator.distributed_type != "NO":
                    pred_rates = forward_results['pred_rates']

                    # 检查是否有任何进程的模型输出包含NaN
                    local_has_nan = torch.isnan(pred_rates).any().float()
                    gathered_nan_results = self.accelerator.gather(local_has_nan)
                    global_has_nan = gathered_nan_results.sum()

                    if global_has_nan.item() > 0:
                        if self.accelerator.is_local_main_process:
                            self.logger.error("FORWARD_NAN", f"维度{dimension} 检测到前向传播NaN", f"batch_idx:{batch_idx}")
                nan_check_time = time.time() - nan_check_start

                loss_compute_start = time.time()
                # ✅ 不再手动除以 gradient_accumulation_steps，accelerator.accumulate 会自动处理
                loss = self.compute_loss(forward_results, criterion, debug_info)
                loss_compute_time = time.time() - loss_compute_start

                # 记录损失值（仅在debug模式下记录详细信息）
                if self.accelerator.is_local_main_process and self.debug_mode:
                    self.logger.log("LOSS_COMPUTED", f"loss={loss.item():.6f} | 耗时={loss_compute_time:.3f}s | NaN检查耗时={nan_check_time:.3f}s",
                                    f"维度{dimension}_batch{batch_idx}", level=2)

                grad_norm = 0.0
                # 使用 Accelerate 的 backward 而不是直接调用 loss.backward()
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
                    # 记录反向传播崩溃信息
                    self.logger.log_crash(
                        step_name="BACKWARD",
                        batch_idx=batch_idx,
                        dimension=dimension,
                        error=e,
                        extra_info=f"loss={loss.item():.6f}"
                    )
                    raise  # 重新抛出异常以终止训练

                # 不再需要判断是否是最后一步，因为 accumulate 会自动处理
                all_params = list(model.parameters()) + list(condition_encoder.parameters())

                # 🔧 修复：遵循 Accelerate 官方文档，在 accumulate 内无条件调用 optimizer
                # 参考: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
                # Accelerate 会自动在正确的时机（基于 gradient_accumulation_steps）执行更新
                # 我们不需要手动检查 sync_gradients

                # 应用梯度裁剪（防止梯度爆炸和消失）
                # ⚠️ 关键修复：在optimizer.step()之前裁剪梯度
                self.accelerator.clip_grad_norm_(all_params, self.GRADIENT_CLIP_NORM)

                # 计算梯度范数（用于监控，不影响训练）
                grad_norm = 0.0
                for param in all_params:
                    if param.grad is not None:
                        grad_norm += float(param.grad.data.norm().item() ** 2)
                grad_norm = float(grad_norm ** 0.5)

                # 无条件执行优化器更新（Accumulate 会自动处理）
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

                batch_total_time = time.time() - batch_start_time

                # 更新进度条显示（每个batch都更新）
                if self.accelerator.is_local_main_process:
                    postfix_dict = {
                        'loss': f'{loss.item():.4f}',
                        'grad_norm': f'{grad_norm:.3f}' if isinstance(grad_norm, (int, float)) else f'{grad_norm:.3f}',
                        'time': f'{batch_total_time:.2f}s' if self.debug_mode else ''
                    }
                    progress_bar.set_postfix(postfix_dict)

                # accumulate 上下文管理器即将退出，记录batch完成
                if self.accelerator.is_local_main_process and self.debug_mode:
                    self.logger.log("BATCH_COMPLETE", f"Batch {batch_idx} 完成 | 总耗时={batch_total_time:.3f}s | timestamp={time.time():.2f}",
                                    f"维度{dimension}_batch{batch_idx}", level=2)

        # 等待所有进程完成
        # 跨进程收集并计算平均损失
        avg_loss = self._gather_average_loss(total_loss, num_batches, default_value=0.0)

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
                forward_results = self.forward_pass(model, condition_embeddings, z0_token_ids, z1_token_ids)

                # 计算损失
                loss = self.compute_loss(forward_results, criterion)
                total_loss += loss.item()
                num_batches += 1

        # 等待所有进程完成
        # 跨进程收集并计算平均损失
        avg_loss = self._gather_average_loss(total_loss, num_batches, default_value=float('inf'))

        return avg_loss


    def save_checkpoint(self, model, condition_encoder, loss, epoch, is_final=False):
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

        model, condition_encoder, criterion, optimizer, scheduler, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

        # 注意这里接收返回值的变化
        train_dataloader, train_dataset, test_dataloader, test_dataset = self.prepare_data(tokenizer)

        model_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad)
        if self.accelerator.is_local_main_process:
            print(f"模型参数数量: {model_params:,}, 条件编码器参数数量: {encoder_params:,}")
            print(f"开始连续流训练 ({self.args.num_epochs} epochs)...")
            # 记录训练开始到 training.log
            self.logger.log("TRAINING_START", f"开始训练 | num_epochs={self.args.num_epochs} | model_params={model_params:,} | encoder_params={encoder_params:,}", level=1)

        eval_every = self.args.eval_every

        for epoch in range(self.args.num_epochs):
            # === 修改开始：不再循环 dim，直接传整个 dataloader ===
            # 这里传入 "Mixed" 作为维度名称仅用于显示
            avg_loss, num_batches = self.train_epoch(
                model, condition_encoder, criterion, optimizer,
                train_dataloader, train_dataset, epoch, "Mixed"
            )

            if self.accelerator.is_local_main_process:
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.args.num_epochs} 完成, 训练损失: {avg_loss:.4f}, 学习率: {current_lr:.2e}")
                # 记录训练成果到 training.log
                self.logger.log("EPOCH_COMPLETE", f"Epoch {epoch+1}/{self.args.num_epochs} | train_loss={avg_loss:.4f} | lr={current_lr:.2e} | batches={num_batches}", level=1)

            # 更新学习率
            scheduler.step()

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
                    model, condition_encoder, avg_loss, epoch
                )
                if self.accelerator.is_local_main_process:
                    print(f"检查点已保存到: {checkpoint_path}")
                    # 记录检查点保存到 training.log
                    self.logger.log("CHECKPOINT_SAVED", f"Epoch {epoch+1}/{self.args.num_epochs} | path={checkpoint_path} | train_loss={avg_loss:.4f}", level=1)

        # 保存最终模型
        final_path = self.save_checkpoint(
            model, condition_encoder, avg_loss, self.args.num_epochs - 1, is_final=True
        )
        if self.accelerator.is_local_main_process:
            print(f"最终模型已保存到: {final_path}")
            # 记录训练完成到 training.log
            self.logger.log("TRAINING_COMPLETE", f"训练完成 | final_path={final_path} | final_train_loss={avg_loss:.4f} | total_epochs={self.args.num_epochs}", level=1)

        return model, condition_encoder

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
        # 记录开始
        self.logger.log("SYMBOLIC_REGRESSION_START",
                       f"输入数据: x形状={x_data.shape}, y形状={y_data.shape} | n_steps={n_steps}",
                       "inference", level=1)

        # 加载模型
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

        model, condition_encoder, _, _, _, tokenizer = self.setup_models(checkpoint_path=checkpoint_path)

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
        from ..symbolic.symbolic_utils import evaluate_expression_safe, expr_to_tree

        # 处理初始表达式
        if initial_expr is None:
            initial_expr = sp.Symbol('x0')
            # 将sympy表达式转换为前缀表达式tokens
            initial_expr_str = expr_to_tree(initial_expr)
            current_tokens = initial_expr_str.split(',') if initial_expr_str else ['x0']
        elif isinstance(initial_expr, str):
            # 将字符串表达式转换为sympy表达式
            initial_expr = sp.sympify(initial_expr)
            # 将sympy表达式转换为前缀表达式tokens
            initial_expr_str = expr_to_tree(initial_expr)
            current_tokens = initial_expr_str.split(',') if initial_expr_str else ['x0']
        elif isinstance(initial_expr, list):
            # 直接传入token列表（用于从训练数据中恢复）
            current_tokens = initial_expr
            # 尝试将tokens转换回sympy表达式（用于计算预测值）
            # 注意：这个转换可能失败，因为tokenizer不支持完全双向转换
            # 我们先尝试评估，如果失败则使用默认值
            initial_expr = None  # 标记为需要特殊处理

        if self.accelerator.is_local_main_process:
            print(f"初始表达式: {initial_expr}")
            print(f"初始tokens: {current_tokens}")

        # 计算初始表达式在x_data上的预测值
        if initial_expr is not None:
            success, y_pred = evaluate_expression_safe(initial_expr, x_data)
        else:
            # 如果无法从tokens恢复表达式，使用常量0作为初始预测
            # 这不是最优解，但可以避免程序崩溃
            success = False
            y_pred = [0.0] * len(y_data)

        if not success:
            self.logger.log("INITIAL_EXPR_WARN", f"无法计算初始表达式的预测值，使用零初始化", "inference", level=1)
            # 继续执行，不返回，因为条件编码器使用的是y_target而非残差

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
                       f"residuals: shape={residuals.shape} range=[{residuals.min():.4f},{residuals.max():.4f}] | "
                       f"initial_expr: {initial_expr} | initial_tokens: {current_tokens}",
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

        # 创建简单推理器
        self.logger.log("SIMPLE_SEARCH_INIT", f"初始化简单推理器 | n_steps={n_steps}", "inference", level=3)

        # 解析action_thresholds参数
        action_thresholds = getattr(self.args, 'action_thresholds', None)
        if action_thresholds:
            try:
                action_thresholds = [float(x.strip()) for x in action_thresholds.split(',')]
                self.logger.log("ACTION_THRESHOLDS_CONFIG",
                               f"使用多阈值推理模式 | thresholds={action_thresholds}",
                               "inference", level=1)
                if self.accelerator.is_local_main_process:
                    print(f"\n使用多阈值推理模式，阈值: {action_thresholds}")
            except ValueError as e:
                self.logger.log("ACTION_THRESHOLDS_PARSE_ERROR",
                               f"无法解析action_thresholds参数: {action_thresholds} | error={e}",
                               "inference", level=1)
                if self.accelerator.is_local_main_process:
                    print(f"\n⚠️ 警告: 无法解析action_thresholds参数 '{action_thresholds}'，回退到单最佳操作模式")
                action_thresholds = None

        simple_searcher = SimpleSymbolicRegression(
            model=model,
            condition_encoder=condition_encoder,
            tokenizer=tokenizer,
            device=device,
            args=self.args,
            logger=self.logger,
            min_action_score=self.MIN_ACTION_SCORE,
            max_expression_length=self.MAX_EXPRESSION_LENGTH,
            numerical_clip_threshold=self.NUMERICAL_CLIP_THRESHOLD,
            action_thresholds=action_thresholds
        )

        # 执行推理（单阈值或多阈值模式）
        initial_residuals_np = residuals.cpu().squeeze(0).numpy()

        # 根据是否启用多阈值模式选择推理方法
        if simple_searcher.use_multi_threshold:
            # 多阈值推理模式
            if self.accelerator.is_local_main_process:
                print(f"\n执行多阈值推理...")

            results_dict = simple_searcher.multi_threshold_search(
                initial_tokens=current_tokens,
                initial_condition=condition,
                initial_residuals=initial_residuals_np,
                x_data=x_data,
                y_data=y_data,
                x_values=x_values,
                n_steps=n_steps
            )

            # 返回所有结果的表达式字典
            if self.accelerator.is_local_main_process:
                print(f"\n多阈值推理完成，返回 {len(results_dict)} 个候选结果")

                # 记录所有结果到日志
                for threshold, candidate in results_dict.items():
                    expr_str = ','.join(candidate.tokens) if candidate and candidate.tokens else ""
                    mse_str = f'{candidate.mse_score:.6f}' if candidate.mse_score is not None else 'N/A'
                    self.logger.log("MULTI_THRESHOLD_RESULT",
                                   f"threshold={threshold} | expression={expr_str} | MSE={mse_str}",
                                   "inference", level=1)

            # 返回字典格式的结果：{threshold: expression}
            return {threshold: ','.join(candidate.tokens) if candidate and candidate.tokens else ""
                    for threshold, candidate in results_dict.items()}

        else:
            # 单最佳操作模式（原有逻辑）
            if self.accelerator.is_local_main_process:
                print(f"\n执行单最佳操作推理...")

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
                mse_score = best_candidate.mse_score
                mse_str = f'{mse_score:.6f}' if mse_score is not None else 'N/A'
                self.logger.log("SIMPLE_SEARCH_RESULT",
                               f"MSE分数: {mse_str} | "
                               f"操作历史: {' -> '.join(best_candidate.history[-5:]) if best_candidate.history else 'N/A'}",
                               "inference", level=3)

            self.logger.log("INFERENCE_COMPLETE", f"最终表达式: {final_expression}", "inference", level=3)
            return final_expression


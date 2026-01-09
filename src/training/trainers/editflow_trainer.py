"""
EditFlow训练器 - 专注于训练循环和评估逻辑
"""

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


class EditFlowTrainer:
    """EditFlow模型训练器 - 负责训练循环、前向传播、损失计算和评估"""

    # 类常量：训练配置参数
    GRADIENT_CLIP_NORM = 10.0
    NUMERICAL_CLIP_THRESHOLD = 1e6
    MAX_EXPRESSION_LENGTH = 50
    MIN_ACTION_SCORE = 0.01

    def __init__(self, model, condition_encoder, criterion, optimizer, scheduler,
                 tokenizer, args, logger, accelerator):
        """
        初始化训练器

        Args:
            model: EditFlow模型
            condition_encoder: 条件编码器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            tokenizer: 分词器
            args: 训练参数配置
            logger: 日志记录器
            accelerator: Accelerate加速器
        """
        self.model = model
        self.condition_encoder = condition_encoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self.accelerator = accelerator
        self.device = accelerator.device
        self.debug_mode = args.debug

    # ============= 分布式工具方法 =============
    def gather_and_format_metrics(self, num_batches, total_loss, total_grad_norm, default_value=0.0):
        """跨进程收集并格式化训练指标（合并方法）

        Args:
            num_batches: 当前进程的批次数
            total_loss: 当前进程的总损失
            total_grad_norm: 当前进程的总梯度范数
            default_value: 默认值（当没有批次时使用）

        Returns:
            dict: 包含所有收集到的指标
                - gpu_metrics: 各GPU的指标信息列表
                - gathered_batches: 各进程的批次数
                - gathered_losses: 各进程的总损失
                - gathered_grad_norms: 各进程的总梯度范数
                - global_total_batches: 全局总批次数
                - global_avg_loss: 全局平均损失
                - global_avg_grad_norm: 全局平均梯度范数
        """
        self.accelerator.wait_for_everyone()

        # 收集各进程的指标
        if self.accelerator.num_processes > 1:
            gathered_batches = self.accelerator.gather(torch.tensor(num_batches, device=self.device))
            gathered_losses = self.accelerator.gather(torch.tensor(total_loss, device=self.device))
            gathered_grad_norms = self.accelerator.gather(torch.tensor(total_grad_norm, device=self.device))
        else:
            gathered_batches = torch.tensor([num_batches], device=self.device)
            gathered_losses = torch.tensor([total_loss], device=self.device)
            gathered_grad_norms = torch.tensor([total_grad_norm], device=self.device)

        # 计算全局汇总
        global_total_batches = gathered_batches.sum().item()
        global_avg_loss = gathered_losses.sum().item() / global_total_batches if global_total_batches > 0 else default_value
        global_avg_grad_norm = gathered_grad_norms.mean().item()

        # 格式化各GPU的详细信息
        gpu_metrics = []
        for gpu_idx in range(self.accelerator.num_processes):
            gpu_batches = gathered_batches[gpu_idx].item()
            gpu_total_loss = gathered_losses[gpu_idx].item()
            gpu_total_grad_norm = gathered_grad_norms[gpu_idx].item()

            gpu_avg_loss = gpu_total_loss / gpu_batches if gpu_batches > 0 else 0.0
            gpu_avg_grad_norm = gpu_total_grad_norm / gpu_batches if gpu_batches > 0 else 0.0

            gpu_metrics.append(
                f"  [GPU {gpu_idx}] batches={gpu_batches} | "
                f"total_loss={gpu_total_loss:.2f} | avg_loss={gpu_avg_loss:.6f} | "
                f"avg_grad_norm={gpu_avg_grad_norm:.3f}"
            )

        return {
            'gpu_metrics': gpu_metrics,
            'gathered_batches': gathered_batches,
            'gathered_losses': gathered_losses,
            'gathered_grad_norms': gathered_grad_norms,
            'global_total_batches': global_total_batches,
            'global_avg_loss': global_avg_loss,
            'global_avg_grad_norm': global_avg_grad_norm
        }

    # ============= 前向传播和损失计算 =============
    def forward_and_compute_loss(self, condition_embeddings, z0_token_ids, z1_token_ids, debug_info=None):
        """前向传播并计算损失（合并方法，减少中间状态传递）"""
        from ..flow import remove_gap_tokens, fill_gap_tokens_with_repeats

        batch_size = z0_token_ids.size(0)
        vocab_size = self.tokenizer.vocab_size

        # 移除gap token得到输入序列x_t（原始序列空间，无gap重复）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = remove_gap_tokens(z0_token_ids, self.tokenizer)
        attention_mask = (~x_pad_mask).float()

        # 调用模型
        output = self.model(input_ids=x_t, condition=condition_embeddings, attention_mask=attention_mask)
        pred_rates = output['rates_logits']

        # 记录debug信息
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

        # 拆分操作logits：ins, del, sub, keep
        ins_logits_rate = pred_rates[:, :, 0:1]
        del_logits_rate = pred_rates[:, :, 1:2]
        sub_logits_rate = pred_rates[:, :, 2:3]
        keep_logits_rate = pred_rates[:, :, 3:4]

        # 在logit空间相加
        ins_logits = output['insert_logits'] + ins_logits_rate
        sub_logits = output['substitute_logits'] + sub_logits_rate
        del_logits = del_logits_rate
        keep_logits = keep_logits_rate

        # 拼接所有操作logits：ins | del | sub | keep
        u_cat_x = torch.cat([ins_logits, del_logits, sub_logits, keep_logits], dim=-1)

        # 将X空间的预测扩展到Z空间
        u_z = fill_gap_tokens_with_repeats(u_cat_x, z_gap_mask, z_pad_mask)

        # 生成编辑操作掩码
        gap_token = self.tokenizer.convert_tokens_to_ids('<gap>')
        u_mask_x = self.criterion.make_ut_mask_from_z(z0_token_ids, z1_token_ids, vocab_size, gap_token, self.tokenizer, x_t)
        u_mask = fill_gap_tokens_with_repeats(u_mask_x, z_gap_mask, z_pad_mask)

        # 记录debug信息
        if self.accelerator.is_local_main_process and self.debug_mode:
            self.logger.log_compute_loss_debug(debug_info, z0_token_ids, z1_token_ids, x_t, pred_rates, u_cat_x, u_mask_x, vocab_size, self.tokenizer)

        # 计算损失
        loss = self.criterion(u_cat_x, u_z, u_mask, vocab_size,
                        accelerator=self.accelerator, logger=self.logger)

        return loss, pred_rates

    # ============= 训练和评估 =============
    def train_epoch(self, dataloader, dataset, epoch, dimension):
        """训练一个epoch"""
        self.model.train()
        self.condition_encoder.train()

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
        try:
            dataset_size = len(dataset)
        except TypeError:
            dataset_size = getattr(dataset, "_size_estimate", self.args.num_samples)
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
                batch, batch_idx, epoch, dimension
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

        # 跨进程收集并格式化训练指标
        metrics = self.gather_and_format_metrics(num_batches, total_loss, local_total_grad_norm, default_value=0.0)
        avg_loss = metrics['global_avg_loss']

        # 数据消耗监控：记录实际处理的 batch 数（只在主进程）
        if self.accelerator.is_local_main_process:
            expected_batches_global = dataset_size // self.args.batch_size
            actual_batches = num_batches
            total_batches_all_processes = metrics['global_total_batches']

            # 计算样本覆盖率
            total_samples_processed = total_batches_all_processes * self.args.batch_size
            coverage_rate = (total_samples_processed / dataset_size * 100) if dataset_size > 0 else 0.0

            # 根据是否分布式训练，显示不同的日志格式
            num_processes = self.accelerator.num_processes
            if num_processes > 1:
                expected_batches_per_process = dataset_size // (self.args.batch_size * num_processes)
                # 构建完整的日志消息
                gpu_metrics_summary = "\n" + "\n".join(metrics['gpu_metrics'])
                data_allocation_summary = (
                    f"\n--- 数据分配 --- | 进程数={num_processes} | 数据集大小={dataset_size} | "
                    f"批次大小={self.args.batch_size} | 预期单进程批次数={expected_batches_per_process} | "
                    f"覆盖率={coverage_rate:.1f}%"
                )
                global_summary = (
                    f"\n--- 全局汇总 --- | 总批次数={total_batches_all_processes} | "
                    f"avg_loss={metrics['global_avg_loss']:.6f} | avg_grad_norm={metrics['global_avg_grad_norm']:.3f}"
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
                    f"Epoch {epoch+1} 完成 | 预期批次数={expected_batches_global} | "
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
            elif epoch > 0:
                expected_batches_warning = expected_batches_global
                if num_processes > 1:
                    expected_batches_warning = dataset_size // (self.args.batch_size * num_processes)
                if actual_batches < expected_batches_warning * 0.5:
                    self.logger.error(
                        "INSUFFICIENT_DATA",
                        f"警告：Epoch {epoch+1} 实际批次数({actual_batches}) "
                        f"远少于预期({expected_batches_warning})，可能存在数据加载问题。",
                        f"epoch{epoch+1}_warning"
                    )

        # 返回平均损失、批次数、总损失和总梯度范数（用于GPU级别信息汇总）
        return avg_loss, num_batches, total_loss, local_total_grad_norm

    def _process_batch(self, batch, batch_idx, epoch, dimension):
        """处理单个训练batch"""
        if self.accelerator.is_local_main_process and self.debug_mode:
            self.logger.log("BATCH_START", f"开始处理 Batch {batch_idx} | timestamp={time.time():.2f}",
                            f"维度{dimension}_batch{batch_idx}", level=2)

        with self.accelerator.accumulate([self.model, self.condition_encoder]):
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
            condition_embeddings = self.condition_encoder(x_values, y_target, point_mask)
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

            # 前向传播并计算损失（合并方法）
            forward_start = time.time()
            loss, pred_rates = self.forward_and_compute_loss(condition_embeddings, z0_token_ids, z1_token_ids, debug_info)
            forward_time = time.time() - forward_start

            if self.accelerator.is_local_main_process and self.debug_mode:
                self.logger.log("FORWARD_PASS", f"前向传播完成 | 耗时={forward_time:.3f}s",
                                f"维度{dimension}_batch{batch_idx}", level=2)

            # NaN检查
            nan_check_start = time.time()
            if self.accelerator.distributed_type != "NO":
                local_has_nan = torch.isnan(pred_rates).any().float()
                gathered_nan_results = self.accelerator.gather(local_has_nan)
                global_has_nan = gathered_nan_results.sum()

                if global_has_nan.item() > 0:
                    self.logger.error("FORWARD_NAN", f"维度{dimension} 检测到前向传播NaN", f"batch_idx:{batch_idx}")

            nan_check_time = time.time() - nan_check_start

            loss_compute_time = 0.0  # 已合并到forward_time中

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
            all_params = list(self.model.parameters()) + list(self.condition_encoder.parameters())
            self.accelerator.clip_grad_norm_(all_params, self.GRADIENT_CLIP_NORM)

            grad_norm = 0.0
            for param in all_params:
                if param.grad is not None:
                    grad_norm += float(param.grad.data.norm().item() ** 2)
            grad_norm = float(grad_norm ** 0.5)

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item(), grad_norm

    def evaluate(self, test_dataloader, test_dataset):
        """测试集评估"""
        self.model.eval()
        self.condition_encoder.eval()

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
        try:
            test_size = len(test_dataset)
        except TypeError:
            test_size = getattr(test_dataset, "_size_estimate", self.args.num_samples)
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

                condition_embeddings = self.condition_encoder(x_values, y_target, point_mask)
                loss, _ = self.forward_and_compute_loss(condition_embeddings, z0_token_ids, z1_token_ids)

                total_loss += loss.item()
                num_batches += 1

        metrics = self.gather_and_format_metrics(num_batches, total_loss, 0.0, default_value=float('inf'))
        avg_loss = metrics['global_avg_loss']

        return avg_loss

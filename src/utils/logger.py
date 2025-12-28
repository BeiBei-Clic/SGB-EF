"""统一的日志管理类 - 支持训练、推理和样本生成日志"""
import os
import datetime
import torch


class Logger:
    """统一的日志管理类

    支持多种日志场景：
    - 训练日志 (training.log)
    - 训练调试日志 (training_debug.log)
    - 推理日志 (inference.log)
    - 样本生成日志 (sample_generation.log)

    日志级别说明：
    - level <= 1: 训练主日志
    - level == 2: 训练调试日志
    - level >= 3: 推理详细日志
    """

    # 日志文件路径
    TRAIN_LOG = "logs/training.log"
    TRAIN_DEBUG_LOG = "logs/training_debug.log"
    INFERENCE_LOG = "logs/inference.log"
    SAMPLE_LOG = "logs/sample_generation.log"

    MAX_LOG_LINES = 100000
    _log_line_count = {}

    def __init__(self, accelerator=None, enabled=True):
        """初始化日志管理器

        Args:
            accelerator: Accelerate 对象（可选）
            enabled (bool): 是否启用日志
        """
        self.accelerator = accelerator
        self.enabled = enabled
        if accelerator:
            self.enabled = enabled and accelerator.is_local_main_process

        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)

    def _write(self, msg, filename):
        """内部方法：写入日志到文件，支持自动轮转

        Args:
            msg (str): 日志消息
            filename (str): 目标文件路径
        """
        if filename not in self._log_line_count:
            self._log_line_count[filename] = sum(1 for _ in open(filename, 'r', errors='ignore')) if os.path.exists(filename) else 0

        if self._log_line_count[filename] >= self.MAX_LOG_LINES:
            # 轮转：保留最近一半
            with open(filename, 'r', errors='ignore') as f:
                lines = f.readlines()[-(self.MAX_LOG_LINES // 2):]
            with open(filename, 'w') as f:
                f.writelines(lines)
            self._log_line_count[filename] = len(lines)

        with open(filename, "a") as f:
            f.write(msg + "\n")
        self._log_line_count[filename] += 1

    # ==================== 通用日志方法 ====================

    def log(self, step_type, details="", context="", level=1):
        """通用日志记录方法

        Args:
            step_type (str): 步骤类型/标签
            details (str): 详细信息
            context (str): 上下文
            level (int): 日志级别 (1=主日志, 2=调试, 3=推理)
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} {step_type}"
        if context:
            msg += f" [{context}]"
        if details:
            msg += f" | {details}"

        # 根据级别选择日志文件
        if level <= 1:
            self._write(msg, self.TRAIN_LOG)
        elif level == 2:
            self._write(msg, self.TRAIN_DEBUG_LOG)
        else:
            self._write(msg, self.INFERENCE_LOG)

        # 推理步骤时同时打印到控制台
        if step_type.startswith("INFERENCE_STEP") and self.enabled:
            print(details)

    def error(self, error_type, error_msg, context=""):
        """记录错误信息

        Args:
            error_type (str): 错误类型
            error_msg (str): 错误消息
            context (str): 上下文
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} ERROR {error_type}"
        if context:
            msg += f" [{context}]"
        msg += f" | {error_msg}"

        self._write(msg, self.TRAIN_LOG)

    # ==================== 训练相关日志 ====================

    def training_start(self, args):
        """记录训练开始"""
        if not self.enabled:
            return

        gpu_info = ""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = f" | GPUs: {gpu_count}"
            for i in range(min(gpu_count, 2)):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_info += f" | GPU{i}: {gpu_name} ({gpu_memory:.1f}GB)"

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"{timestamp} TRAINING START | Model: {getattr(args, 'model_name', 'Unknown')}{gpu_info}"
        self._write(msg, self.TRAIN_LOG)

    def training_complete(self, total_epochs, final_loss=None):
        """记录训练完成"""
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"{timestamp} TRAINING COMPLETE | Total epochs: {total_epochs}"
        if final_loss is not None:
            msg += f" | Final loss: {final_loss:.6f}"
        self._write(msg, self.TRAIN_LOG)

    def model_info(self, model):
        """记录模型信息"""
        if not self.enabled:
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} MODEL_INFO | Total params: {total_params:,} | Trainable: {trainable_params:,}"
        self._write(msg, self.TRAIN_LOG)

    # ==================== 张量和GPU日志 ====================

    def tensor(self, tensor_name, tensor, context="", level=2):
        """记录张量信息

        Args:
            tensor_name (str): 张量名称
            tensor: 张量对象
            context (str): 上下文
            level (int): 日志级别
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} TENSOR {tensor_name}"
        if context:
            msg += f" [{context}]"

        if isinstance(tensor, torch.Tensor):
            msg += f" | shape={list(tensor.shape)} | dtype={tensor.dtype} | device={tensor.device}"
            if tensor.numel() > 0:
                msg += f" | min={tensor.min().item():.6f} | max={tensor.max().item():.6f} | mean={tensor.float().mean().item():.6f}"
                if torch.isnan(tensor).any():
                    msg += " | HAS_NAN"
                if torch.isinf(tensor).any():
                    msg += " | HAS_INF"
        else:
            msg += f" | type={type(tensor)} | value={tensor}"

        # 根据级别选择日志文件
        if level == 1:
            self._write(msg, self.TRAIN_LOG)
        elif level == 2:
            self._write(msg, self.TRAIN_DEBUG_LOG)
        else:
            self._write(msg, self.INFERENCE_LOG)

    def gpu_memory(self, context=""):
        """记录GPU内存使用情况

        Args:
            context (str): 上下文
        """
        if not self.enabled or not torch.cuda.is_available():
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} GPU_MEMORY"
        if context:
            msg += f" [{context}]"

        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            msg += f" | GPU{i}: {allocated:.1f}GB/{total:.1f}GB ({allocated/total*100:.1f}%)"

        self._write(msg, self.TRAIN_LOG)

    # ==================== 样本生成日志 ====================

    def sample_step(self, sample_id, step, info=""):
        """记录样本步骤（跳过常规步骤）

        Args:
            sample_id: 样本ID
            step: 步骤描述
            info: 额外信息
        """
        if not self.enabled:
            return

        # 跳过一些常规步骤，减少日志量
        if not info and any(step.startswith(s) for s in ["生成数据点", "计算", "处理删减", "生成当前", "生成删减"]):
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] {step}" + (f" - {info}" if info else "")
        self._write(msg, self.SAMPLE_LOG)

    def expression_eval(self, sample_id, expr_str, eval_time_ms, success=True, error_msg=""):
        """记录表达式评估结果

        Args:
            sample_id: 样本ID
            expr_str: 表达式字符串
            eval_time_ms: 评估耗时（毫秒）
            success: 是否成功
            error_msg: 错误信息
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        status = "OK" if success else f"FAIL: {error_msg}"
        msg = f"{timestamp} [{sample_id}] EVAL {status} {eval_time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def write(self, msg, filename=None):
        """直接写入日志（用于样本生成等特殊场景）

        Args:
            msg: 日志消息
            filename: 目标文件，默认为 SAMPLE_LOG
        """
        if filename is None:
            filename = self.SAMPLE_LOG
        self._write(msg, filename)

    # ==================== 数据生成相关日志 ====================

    def batch_start(self, batch_idx, total_batches, process_id=""):
        """记录批次开始

        Args:
            batch_idx: 批次索引
            total_batches: 总批次数
            process_id: 进程标识
        """
        if not self.enabled:
            return
        prefix = f"[B{batch_idx+1}]" if not process_id else f"[{process_id}]"
        self.log("BATCH_START", f"批次 {batch_idx+1}/{total_batches}", prefix, level=1)

    def batch_complete(self, batch_idx, sample_count, dimension_count=None):
        """记录批次完成

        Args:
            batch_idx: 批次索引
            sample_count: 样本数量
            dimension_count: 维度统计字典
        """
        if not self.enabled:
            return

        prefix = f"[B{batch_idx+1}]"
        msg = f"完成 (生成{sample_count}个样本)"
        if dimension_count:
            dim_info = ", ".join([f"{dim}维:{count}" for dim, count in sorted(dimension_count.items())])
            msg += f" | 维度分布: {dim_info}"

        self.log("BATCH_COMPLETE", msg, prefix, level=1)

    def sample_step(self, sample_id, step, details="", info_only=False):
        """记录样本生成步骤

        Args:
            sample_id: 样本ID
            step: 步骤描述
            details: 详细信息
            info_only: 是否仅记录重要信息（跳过常规步骤）
        """
        if not self.enabled:
            return

        # 跳过一些常规步骤，减少日志量
        if info_only and not details and any(step.startswith(s) for s in ["生成数据点", "计算", "处理删减", "生成当前", "生成删减"]):
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] {step}" + (f" | {details}" if details else "")
        self._write(msg, self.SAMPLE_LOG)

    def expression_generate(self, sample_id, expr_str, gen_time_ms):
        """记录表达式生成

        Args:
            sample_id: 样本ID
            expr_str: 表达式字符串
            gen_time_ms: 生成耗时（毫秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] GENERATE_RANDOM_EXPR '{expr_str}' | time={gen_time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def expression_validation(self, sample_id, expr_str, expr_len, token_count):
        """记录表达式验证通过

        Args:
            sample_id: 样本ID
            expr_str: 表达式字符串
            expr_len: 表达式长度
            token_count: token数量
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] EXPR_VALIDATION_PASSED '{expr_str}' len={expr_len} tokens={token_count}"
        self._write(msg, self.SAMPLE_LOG)

    def expression_convert(self, sample_id, token_count, convert_time_ms):
        """记录表达式转换

        Args:
            sample_id: 样本ID
            token_count: token数量
            convert_time_ms: 转换耗时（毫秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] EXPR_TO_TREE tokens={token_count} | time={convert_time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def reduction_sequence(self, sample_id, step_count, time_ms):
        """记录删减序列生成

        Args:
            sample_id: 样本ID
            step_count: 步骤数量
            time_ms: 耗时（毫秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] REDUCE_SEQUENCE {step_count} steps | time={time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def corrupt_expression(self, sample_id, step, orig_expr, corrupt_expr, time_ms):
        """记录表达式破坏

        Args:
            sample_id: 样本ID
            step: 步骤编号
            orig_expr: 原始表达式
            corrupt_expr: 破坏后表达式
            time_ms: 耗时（毫秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg1 = f"{timestamp} [{sample_id}] CORRUPT_EXPRESSION step{step} | time={time_ms:.1f}ms"
        msg2 = f"{timestamp} [{sample_id}] CORRUPT_RESULT step{step} '{orig_expr}' → '{corrupt_expr}'"
        self._write(msg1, self.SAMPLE_LOG)
        self._write(msg2, self.SAMPLE_LOG)

    def skip_duplicate(self, sample_id, step):
        """记录跳过重复表达式

        Args:
            sample_id: 样本ID
            step: 步骤编号
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] SKIP_DUPLICATE step{step}"
        self._write(msg, self.SAMPLE_LOG)

    def skip_complex(self, sample_id, step, expr_str):
        """记录跳过复数表达式

        Args:
            sample_id: 样本ID
            step: 步骤编号
            expr_str: 表达式字符串
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] SKIP_COMPLEX step{step} expr='{expr_str}'"
        self._write(msg, self.SAMPLE_LOG)

    def eval_curr_expression(self, sample_id, step, success, time_ms, expr_str=""):
        """记录当前表达式评估

        Args:
            sample_id: 样本ID
            step: 步骤编号
            success: 是否成功
            time_ms: 耗时（毫秒）
            expr_str: 表达式字符串
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        expr_info = f" expr='{expr_str}'" if expr_str else ""
        msg = f"{timestamp} [{sample_id}] EVAL_CURR_EXPRESSION step{step} success={success} | time={time_ms:.1f}ms{expr_info}"
        self._write(msg, self.SAMPLE_LOG)

    def convert_to_trees(self, sample_id, step, target_tokens, curr_tokens, time_ms):
        """记录转换为树

        Args:
            sample_id: 样本ID
            step: 步骤编号
            target_tokens: 目标token数
            curr_tokens: 当前token数
            time_ms: 耗时（毫秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] CONVERT_TO_TREES step{step} | time={time_ms:.1f}ms target_tokens={target_tokens} curr_tokens={curr_tokens}"
        self._write(msg, self.SAMPLE_LOG)

    def levenshtein_alignment(self, sample_id, step, z0_len, z1_len, time_ms):
        """记录对齐操作

        Args:
            sample_id: 样本ID
            step: 步骤编号
            z0_len: z0长度
            z1_len: z1长度
            time_ms: 耗时（毫秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] LEVENSHTEIN_ALIGNMENT step{step} | time={time_ms:.1f}ms z0_len={z0_len} z1_len={z1_len}"
        self._write(msg, self.SAMPLE_LOG)

    def residuals_before_clip(self, sample_id, step, min_val, max_val, mean_val):
        """记录裁剪前的residuals统计

        Args:
            sample_id: 样本ID
            step: 步骤编号
            min_val: 最小值
            max_val: 最大值
            mean_val: 平均值
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] RESIDUALS_BEFORE_CLIP step{step} | min={min_val:.6f} max={max_val:.6f} mean={mean_val:.6f}"
        self._write(msg, self.SAMPLE_LOG)

    def skip_clipped(self, sample_id, step, clip_count, total_count, threshold):
        """记录跳过裁剪的样本

        Args:
            sample_id: 样本ID
            step: 步骤编号
            clip_count: 裁剪数量
            total_count: 总数量
            threshold: 阈值
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] SKIP_CLIPPED step{step} | clipped={clip_count}/{total_count} threshold={threshold}"
        self._write(msg, self.SAMPLE_LOG)

    def sample_success(self, sample_id):
        """记录样本生成成功

        Args:
            sample_id: 样本ID
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] SUCCESS"
        self._write(msg, self.SAMPLE_LOG)

    def sample_error(self, sample_id, error_type, error_msg, duration=None):
        """记录样本生成错误

        Args:
            sample_id: 样本ID
            error_type: 错误类型
            error_msg: 错误消息
            duration: 持续时间（秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] ERROR {error_type}: {error_msg}"
        self._write(msg, self.SAMPLE_LOG)

        if duration is not None:
            stuck_msg = f"{timestamp} [{sample_id}] STUCK {duration:.1f}s"
            self._write(stuck_msg, self.SAMPLE_LOG)

    def sample_failed(self, sample_id, reason):
        """记录样本生成失败

        Args:
            sample_id: 样本ID
            reason: 失败原因
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] FAILED: {reason}"
        self._write(msg, self.SAMPLE_LOG)

    def sample_timeout(self, sample_id, timeout_seconds):
        """记录样本生成超时

        Args:
            sample_id: 样本ID
            timeout_seconds: 超时时间（秒）
        """
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} [{sample_id}] TIMEOUT: Sample generation exceeded {timeout_seconds}s"
        self._write(msg, self.SAMPLE_LOG)

    # ==================== 束搜索推理日志 ====================

    def log_beam_search_separator(self, title="", level=2):
        """记录分隔线

        Args:
            title: 标题
            level: 日志级别
        """
        if not self.enabled:
            return

        separator = "=" * 80
        if title:
            msg = separator
            self._write(msg, self.TRAIN_DEBUG_LOG if level == 2 else self.INFERENCE_LOG)
            self.log("BEAM_SEARCH", title, "beam_search", level=level)
        else:
            msg = separator
            self._write(msg, self.TRAIN_DEBUG_LOG if level == 2 else self.INFERENCE_LOG)

    def log_beam_search_insert_probs(self, position, lambda_rate, tokens, probs, level=2):
        """记录插入操作的预测概率

        Args:
            position: 位置索引
            lambda_rate: 插入速率
            tokens: token列表
            probs: 概率列表
            level: 日志级别
        """
        if not self.enabled:
            return

        tokens_str = str(tokens)
        probs_str = [f"{p:.6f}" for p in probs]
        self.log("INSERT_PROBS_DEBUG",
                f"位置{position}: lambda={lambda_rate:.4f} | top_tokens={tokens_str} | probs={probs_str}",
                "beam_search", level=level)

    def log_beam_search_substitute_probs(self, position, current_token, lambda_rate, tokens, probs, level=2):
        """记录替换操作的预测概率

        Args:
            position: 位置索引
            current_token: 当前token
            lambda_rate: 替换速率
            tokens: token列表
            probs: 概率列表
            level: 日志级别
        """
        if not self.enabled:
            return

        tokens_str = str(tokens)
        probs_str = [f"{p:.6f}" for p in probs]
        self.log("SUBSTITUTE_PROBS_DEBUG",
                f"位置{position}(当前={current_token}): lambda={lambda_rate:.4f} | top_tokens={tokens_str} | probs={probs_str}",
                "beam_search", level=level)

    def log_beam_search_delete_probs(self, position, current_token, lambda_rate, above_threshold, level=2):
        """记录删除操作的预测概率

        Args:
            position: 位置索引
            current_token: 当前token
            lambda_rate: 删除速率
            above_threshold: 是否超过阈值
            level: 日志级别
        """
        if not self.enabled:
            return

        self.log("DELETE_PROBS_DEBUG",
                f"位置{position}(当前={current_token}): lambda_del={lambda_rate:.4f} | above_threshold={above_threshold}",
                "beam_search", level=level)

    def log_beam_search_token_type_stats(self, token_categories, level=2):
        """记录词汇类型的预测统计

        Args:
            token_categories: 字典，键为类型名，值为[(token, prob), ...]列表
            level: 日志级别
        """
        if not self.enabled:
            return

        for category_name, token_probs in token_categories.items():
            if token_probs:
                # 只显示前5个
                top_tokens = [(t, f"{p:.6f}") for t, p in token_probs[:5]]
                self.log("TOKEN_TYPE_STATS", f"{category_name}: {top_tokens}", "beam_search", level=level)


    # ==================== 推理详细日志 ====================

    def log_residuals_stats(self, step, residuals, context="greedy_search", level=3):
        """记录残差的详细统计信息

        Args:
            step: 推理步数
            residuals: 残差数组 (numpy array)
            context: 上下文标识
            level: 日志级别
        """
        if not self.enabled or residuals is None:
            return

        import numpy as np

        # 基本统计信息
        self.log("RESIDUAL_STATS",
                f"step={step} | "
                f"shape={residuals.shape} | "
                f"min={residuals.min():.6f} | max={residuals.max():.6f} | "
                f"mean={residuals.mean():.6f} | std={residuals.std():.6f} | "
                f"median={np.median(residuals):.6f} | "
                f"l2_norm={np.linalg.norm(residuals):.6f}",
                context, level=level)

        # 分位数统计
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(residuals, p) for p in percentiles]
        percentile_str = " | ".join([f"p{p}={v:.6f}" for p, v in zip(percentiles, percentile_values)])
        self.log("RESIDUAL_DISTRIBUTION",
                f"step={step} | {percentile_str}",
                context, level=level)

        # 异常值检测
        threshold = 3 * residuals.std()
        outliers_mask = np.abs(residuals) > threshold
        n_outliers = int(outliers_mask.sum())
        if n_outliers > 0:
            outlier_max = np.abs(residuals[outliers_mask]).max()
            outlier_mean = np.abs(residuals[outliers_mask]).mean()
            self.log("RESIDUAL_OUTLIERS",
                    f"step={step} | n_outliers={n_outliers}/{len(residuals)} | "
                    f"threshold={threshold:.6f} | max_outlier={outlier_max:.6f} | "
                    f"mean_outlier={outlier_mean:.6f}",
                    context, level=level)

    def log_condition_stats(self, step, condition, context="greedy_search", level=3):
        """记录条件嵌入的详细统计信息

        Args:
            step: 推理步数
            condition: 条件嵌入张量 (torch.Tensor)
            context: 上下文标识
            level: 日志级别
        """
        if not self.enabled or condition is None:
            return

        import numpy as np

        # 处理不同维度的条件嵌入
        if isinstance(condition, torch.Tensor):
            # 处理序列格式 (batch, num_seeds, dim_hidden) 或 (batch, dim_hidden)
            condition_cpu = condition.detach().cpu()

            if condition_cpu.dim() == 3:
                # (batch, num_seeds, dim_hidden)
                condition_flat = condition_cpu.squeeze(0).flatten().numpy()
                shape_info = f"{list(condition.shape)} (序列格式)"
            elif condition_cpu.dim() == 2:
                # (batch, dim_hidden)
                condition_flat = condition_cpu.squeeze(0).numpy()
                shape_info = f"{list(condition.shape)} (向量格式)"
            else:
                condition_flat = condition_cpu.flatten().numpy()
                shape_info = f"{list(condition.shape)}"

            # 基本统计信息
            self.log("CONDITION_STATS",
                    f"step={step} | "
                    f"shape={shape_info} | "
                    f"min={condition_flat.min():.6f} | max={condition_flat.max():.6f} | "
                    f"mean={condition_flat.mean():.6f} | std={condition_flat.std():.6f} | "
                    f"l2_norm={np.linalg.norm(condition_flat):.6f}",
                    context, level=level)

            # 分位数统计
            percentiles = [10, 25, 50, 75, 90]
            percentile_values = [np.percentile(condition_flat, p) for p in percentiles]
            percentile_str = " | ".join([f"p{p}={v:.6f}" for p, v in zip(percentiles, percentile_values)])
            self.log("CONDITION_DISTRIBUTION",
                    f"step={step} | {percentile_str}",
                    context, level=level)

    def log_inference_step(self, step, total_steps, current_tokens, t,
                          residuals=None, condition=None,
                          model_pred_time=None, total_time=None,
                          context="greedy_search"):
        """记录推理步骤的综合信息

        Args:
            step: 当前步骤
            total_steps: 总步骤数
            current_tokens: 当前token列表
            t: 时间步
            residuals: 残差数组（可选）
            condition: 条件嵌入（可选）
            model_pred_time: 模型预测时间（毫秒）
            total_time: 总时间（毫秒）
            context: 上下文标识
        """
        if not self.enabled:
            return

        expr_str = ','.join(current_tokens) if current_tokens else '<blank>'
        expr_short = expr_str if len(expr_str) <= 30 else expr_str[:30] + '...'

        timing_info = ""
        if model_pred_time is not None:
            timing_info += f" model_time={model_pred_time:.1f}ms"
        if total_time is not None:
            timing_info += f" total_time={total_time:.1f}ms"

        self.log("INFERENCE_STEP",
                f"step={step}/{total_steps} | t={t:.4f} | "
                f"expr='{expr_short}' | "
                f"len={len(current_tokens)}{timing_info}",
                context, level=2)

    def log_residuals_evolution(self, steps_list, residuals_list, context="greedy_search"):
        """记录残差演化趋势

        Args:
            steps_list: 步骤列表
            residuals_list: 对应的残差列表
            context: 上下文标识
        """
        if not self.enabled or not steps_list or not residuals_list:
            return

        import numpy as np

        evolution_lines = []
        evolution_lines.append(f"残差演化趋势 (共{len(steps_list)}步):")
        evolution_lines.append(f"{'步骤':<6} {'L2范数':<12} {'均值':<12} {'标准差':<12} {'最大值':<12}")
        evolution_lines.append("-" * 60)

        for step, residuals in zip(steps_list, residuals_list):
            if residuals is not None:
                l2_norm = np.linalg.norm(residuals)
                mean_val = residuals.mean()
                std_val = residuals.std()
                max_val = np.abs(residuals).max()
                evolution_lines.append(f"{step:<6} {l2_norm:<12.6f} {mean_val:<12.6f} {std_val:<12.6f} {max_val:<12.6f}")

        evolution_str = "\n".join(evolution_lines)
        self.log("RESIDUALS_EVOLUTION", evolution_str, context, level=2)

    def log_model_prediction_stats(self, step, pred_rates, context="greedy_search", level=3):
        """记录模型预测的统计信息

        Args:
            step: 推理步数
            pred_rates: 预测的速率张量 [batch, seq_len, 3]
            context: 上下文标识
            level: 日志级别
        """
        if not self.enabled or pred_rates is None:
            return

        import torch

        # pred_rates: [batch, seq_len, 3] -> [ins_rate, del_rate, sub_rate]
        rates_cpu = pred_rates[0].cpu().numpy()  # 去掉batch维度

        lambda_ins = rates_cpu[:, 0]  # 插入速率
        lambda_del = rates_cpu[:, 1]  # 删除速率
        lambda_sub = rates_cpu[:, 2]  # 替换速率

        self.log("MODEL_PREDICTION_STATS",
                f"step={step} | "
                f"ins_rate: mean={lambda_ins.mean():.4f} max={lambda_ins.max():.4f} | "
                f"del_rate: mean={lambda_del.mean():.4f} max={lambda_del.max():.4f} | "
                f"sub_rate: mean={lambda_sub.mean():.4f} max={lambda_sub.max():.4f}",
                context, level=level)


__all__ = ['Logger']

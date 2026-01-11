"""统一的日志管理类 - 支持训练、推理和样本生成日志"""
import os
import datetime
import torch
import numpy as np


class Logger:
    """统一的日志管理类

    支持多种日志场景：
    - 训练日志 (training.log) - 不受debug控制
    - 训练调试日志 (training_debug.log) - 受debug控制
    - 推理日志 (inference.log) - 受debug控制
    - 样本生成日志 (sample_generation.log) - 不受debug控制

    日志级别说明：
    - level = 1: 训练主日志（training.log，不受debug控制）
    - level = 2: 训练调试日志（training_debug.log，受debug控制）
    - level = 3: 推理详细日志（inference.log，受debug控制）
    """

    # 日志文件路径
    TRAIN_LOG = "logs/training.log"
    TRAIN_DEBUG_LOG = "logs/training_debug.log"
    INFERENCE_LOG = "logs/inference.log"
    SAMPLE_LOG = "logs/sample_generation.log"
    CRASH_LOG = "logs/training_crash.log"

    MAX_LOG_LINES = 100000000
    _log_line_count = {}

    def __init__(self, accelerator=None, enabled=True, debug_mode=False):
        """初始化日志管理器

        Args:
            accelerator: Accelerate 对象（可选）
            enabled (bool): 是否启用日志
            debug_mode (bool): 是否启用调试模式
        """
        self.accelerator = accelerator
        self.enabled = enabled
        self.debug_mode = debug_mode
        if accelerator:
            self.enabled = enabled and accelerator.is_local_main_process

        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)

    def _get_timestamp(self):
        """获取格式化的时间戳"""
        return datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]

    def _should_log(self, level):
        """检查是否应该记录该级别的日志"""
        if not self.enabled:
            return False
        # level=2需要debug_mode启用
        if level == 2 and not self.debug_mode:
            return False
        # level=3推理日志始终记录，不受debug控制
        return True

    def _get_log_file(self, level):
        """根据日志级别获取日志文件路径"""
        if level == 1:
            return self.TRAIN_LOG
        elif level == 2:
            return self.TRAIN_DEBUG_LOG
        elif level == 3:
            return self.INFERENCE_LOG
        return self.TRAIN_LOG

    def _format_array_info(self, values, max_elements, sample_first_n, sample_last_n):
        """格式化数组信息（通用方法，用于torch.Tensor和np.ndarray）

        Args:
            values: 展平后的数组值
            max_elements: 最大元素数量
            sample_first_n: 采样前N个
            sample_last_n: 采样后N个

        Returns:
            格式化后的字符串
        """
        total = len(values)
        if total == 0:
            return " | EMPTY"
        elif total <= max_elements:
            # 元素较少，显示所有内容
            values_str = ", ".join([f"{v:.6f}" for v in values])
            return f" | values=[{values_str}]"
        else:
            # 元素过多，显示采样
            first_vals = values[:sample_first_n]
            last_vals = values[-sample_last_n:]

            first_str = ", ".join([f"{v:.6f}" for v in first_vals])
            last_str = ", ".join([f"{v:.6f}" for v in last_vals])

            result = f" | total_elements={total} | sampled: first_{sample_first_n}=[{first_str}] ... last_{sample_last_n}=[{last_str}]"
            # 额外统计信息
            result += f" | min={values.min():.6f} | max={values.max():.6f} | mean={values.mean():.6f} | std={values.std():.6f}"
            return result

    def _write(self, msg, filename):
        """内部方法：写入日志到文件，支持自动轮转"""
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
            level (int): 日志级别 (1=训练主日志, 2=训练调试日志, 3=推理日志)
        """
        if not self._should_log(level):
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} {step_type}"
        if context:
            msg += f" [{context}]"
        if details:
            msg += f" | {details}"

        self._write(msg, self._get_log_file(level))

        # 推理步骤时同时打印到控制台
        if step_type.startswith("INFERENCE_STEP") and self.enabled:
            print(details)

    def error(self, error_type, error_msg, context=""):
        """记录错误信息"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
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

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"{timestamp} TRAINING START | Model: {getattr(args, 'model_name', 'Unknown')}"
        self._write(msg, self.TRAIN_LOG)

    # ==================== 张量和GPU日志 ====================

    def tensor_values(self, tensor_name, tensor, context="", level=2,
                      max_elements=100, sample_first_n=5, sample_last_n=5):
        """记录张量的完整内容（直接输出变量值）

        Args:
            tensor_name (str): 张量名称
            tensor: 张量对象
            context (str): 上下文
            level (int): 日志级别 (1=训练主日志, 2=训练调试日志, 3=推理日志)
            max_elements (int): 最大元素数量，超过则采样显示
            sample_first_n (int): 当元素过多时，显示前N个
            sample_last_n (int): 当元素过多时，显示后N个
        """
        if not self._should_log(level):
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} TENSOR_VALUES {tensor_name}"
        if context:
            msg += f" [{context}]"

        if isinstance(tensor, torch.Tensor):
            msg += f" | shape={list(tensor.shape)} | dtype={tensor.dtype} | device={tensor.device}"
            tensor_cpu = tensor.detach().cpu()

            # 转换为numpy数组便于处理
            if tensor_cpu.numel() == 0:
                msg += " | EMPTY"
            else:
                values = tensor_cpu.flatten().numpy()
                msg += self._format_array_info(values, max_elements, sample_first_n, sample_last_n)

        elif isinstance(tensor, np.ndarray):
            msg += f" | shape={list(tensor.shape)} | dtype={tensor.dtype}"

            if tensor.size == 0:
                msg += " | EMPTY"
            else:
                values = tensor.flatten()
                msg += self._format_array_info(values, max_elements, sample_first_n, sample_last_n)
        else:
            # 标量或其他类型
            msg += f" | type={type(tensor).__name__} | value={tensor}"

        self._write(msg, self._get_log_file(level))

    # ==================== 样本生成日志 ====================

    def sample_step(self, sample_id, step, details="", info_only=False):
        """记录样本生成步骤"""
        if not self.enabled:
            return

        # 跳过一些常规步骤，减少日志量
        if info_only and not details and any(step.startswith(s) for s in ["生成数据点", "计算", "处理删减", "生成当前", "生成删减"]):
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] {step}" + (f" | {details}" if details else "")
        self._write(msg, self.SAMPLE_LOG)

    def expression_generate(self, sample_id, expr_str, gen_time_ms=None):
        """记录表达式生成"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] GENERATE_RANDOM_EXPR '{expr_str}'"
        self._write(msg, self.SAMPLE_LOG)

    def expression_validation(self, sample_id, expr_str, expr_len, token_count):
        """记录表达式验证通过"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] EXPR_VALIDATION_PASSED '{expr_str}' len={expr_len} tokens={token_count}"
        self._write(msg, self.SAMPLE_LOG)

    def expression_convert(self, sample_id, token_count, convert_time_ms=None):
        """记录表达式转换"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] EXPR_TO_TREE tokens={token_count}"
        self._write(msg, self.SAMPLE_LOG)

    def reduction_sequence(self, sample_id, step_count, time_ms=None):
        """记录删减序列生成"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] REDUCE_SEQUENCE {step_count} steps"
        self._write(msg, self.SAMPLE_LOG)

    def corrupt_expression(self, sample_id, step, orig_expr, corrupt_expr, time_ms=None):
        """记录表达式破坏"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] CORRUPT_RESULT step{step} '{orig_expr}' → '{corrupt_expr}'"
        self._write(msg, self.SAMPLE_LOG)

    def skip_duplicate(self, sample_id, step):
        """记录跳过重复表达式"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] SKIP_DUPLICATE step{step}"
        self._write(msg, self.SAMPLE_LOG)

    def skip_complex(self, sample_id, step, expr_str):
        """记录跳过复数表达式"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] SKIP_COMPLEX step{step} expr='{expr_str}'"
        self._write(msg, self.SAMPLE_LOG)

    def eval_curr_expression(self, sample_id, step, success, time_ms=None, expr_str=""):
        """记录当前表达式评估"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        expr_info = f" expr='{expr_str}'" if expr_str else ""
        msg = f"{timestamp} [{sample_id}] EVAL_CURR_EXPRESSION step{step} success={success}{expr_info}"
        self._write(msg, self.SAMPLE_LOG)

    def convert_to_trees(self, sample_id, step, target_tokens, curr_tokens, time_ms=None):
        """记录转换为树"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] CONVERT_TO_TREES step{step} target_tokens={target_tokens} curr_tokens={curr_tokens}"
        self._write(msg, self.SAMPLE_LOG)

    def levenshtein_alignment(self, sample_id, step, z0_len, z1_len, time_ms=None):
        """记录对齐操作"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] LEVENSHTEIN_ALIGNMENT step{step} z0_len={z0_len} z1_len={z1_len}"
        self._write(msg, self.SAMPLE_LOG)

    def residuals_before_clip(self, sample_id, step, min_val, max_val, mean_val):
        """记录裁剪前的residuals统计（已废弃，保留方法以避免调用错误）"""
        pass

    def skip_clipped(self, sample_id, step, clip_count, total_count, threshold):
        """记录跳过裁剪的样本"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] SKIP_CLIPPED step{step} | clipped={clip_count}/{total_count} threshold={threshold}"
        self._write(msg, self.SAMPLE_LOG)

    def sample_success(self, sample_id):
        """记录样本生成成功"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] SUCCESS"
        self._write(msg, self.SAMPLE_LOG)

    def sample_error(self, sample_id, error_type, error_msg, duration=None):
        """记录样本生成错误"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] ERROR {error_type}: {error_msg}"
        self._write(msg, self.SAMPLE_LOG)

        if duration is not None:
            stuck_msg = f"{timestamp} [{sample_id}] STUCK {duration:.1f}s"
            self._write(stuck_msg, self.SAMPLE_LOG)

    def sample_failed(self, sample_id, reason):
        """记录样本生成失败"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] FAILED: {reason}"
        self._write(msg, self.SAMPLE_LOG)

    def sample_timeout(self, sample_id, timeout_seconds):
        """记录样本生成超时"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] TIMEOUT: Sample generation exceeded {timeout_seconds}s"
        self._write(msg, self.SAMPLE_LOG)

    def expression_eval(self, sample_id, expr_str, eval_time_ms=None, success=True, error_msg=""):
        """记录表达式评估结果"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        status = "OK" if success else f"FAIL: {error_msg}"
        msg = f"{timestamp} [{sample_id}] EVAL {status}"
        self._write(msg, self.SAMPLE_LOG)

    # ==================== 编辑操作日志 ====================

    def log_u_mask_split(self, tensor_name, u_mask, x_t, vocab_size, context="", level=3):
        """按语义拆分记录u_mask（INSERT/DELETE/SUBSTITUTE/KEEP四个独立张量），按位置分行输出"""
        if not self._should_log(level):
            return

        timestamp = self._get_timestamp()
        batch_size, x_seq_len, mask_dim = u_mask.shape

        # 按样本分别记录
        for batch_idx in range(batch_size):
            msg = f"{timestamp} TENSOR_VALUES {tensor_name}_batch{batch_idx}"
            if context:
                msg += f" [{context}]"
            msg += f" | shape=[{x_seq_len}, {mask_dim}] | dtype={u_mask.dtype} | device={u_mask.device}"

            # 对每个位置分别输出
            for pos in range(x_seq_len):
                # 边界检查：确保pos在x_t的范围内
                if pos >= x_t.shape[1]:
                    break
                if x_t[batch_idx, pos].item() == 0:  # pad token
                    break

                # INSERT部分：前vocab_size维（索引 0 ~ vocab_size-1）
                ins_vals = u_mask[batch_idx, pos, :vocab_size].cpu().numpy()
                ins_str = ", ".join([f"{v:.0f}" for v in ins_vals])

                # DELETE部分：第vocab_size位（索引 vocab_size）
                del_val = u_mask[batch_idx, pos, vocab_size].cpu().item()

                # SUBSTITUTE部分：vocab_size+1 ~ 2*vocab_size（偏移+1）
                sub_vals = u_mask[batch_idx, pos, vocab_size+1:2*vocab_size+1].cpu().numpy()
                sub_str = ", ".join([f"{v:.0f}" for v in sub_vals])

                # KEEP部分：最后1位（索引 2*vocab_size+1，即-1）
                keep_val = u_mask[batch_idx, pos, -1].cpu().item()

                msg += f"\n  pos{pos}(x_t={x_t[batch_idx, pos].item()}):"
                msg += f"\n    INSERT:    [{ins_str}]"
                msg += f"\n    DELETE:    {del_val:.0f}"
                msg += f"\n    SUBSTITUTE: [{sub_str}]"
                msg += f"\n    KEEP:      {keep_val:.0f}"

            self._write(msg, self._get_log_file(level))

    def log_edit_operations(self, u_mask_sample, x_t_sample, vocab_size,
                           context="", level=3, max_ops=20, pad_token_id=None):
        """解码并记录Ground Truth编辑操作（使用ID，不翻译成token）

        Args:
            pad_token_id: pad token的ID (必须从tokenizer传入，不可为None)
        """
        if not self._should_log(level):
            return

        # 要求必须传入pad_token_id
        if pad_token_id is None:
            raise ValueError("pad_token_id is required. Please pass tokenizer.pad_token_id")

        ops = []
        x_seq_len = u_mask_sample.shape[0]

        for pos in range(min(x_seq_len, len(x_t_sample))):
            # 跳过pad位置
            if pos >= len(x_t_sample):
                break
            if x_t_sample[pos].item() == pad_token_id:
                break

            # 检查是否需要操作
            if u_mask_sample[pos].any():
                # 检查插入操作 (前vocab_size维，索引 0 ~ vocab_size-1)
                ins_idx = u_mask_sample[pos, :vocab_size].nonzero(as_tuple=False)
                if ins_idx.numel() > 0:
                    token_id = ins_idx[0].item()
                    current_id = x_t_sample[pos].item()
                    # INSERT语义：在位置pos的元素之后插入token_id
                    ops.append(f"pos{pos}之后: INSERT id({token_id})  (在id({current_id})之后)")

                # 检查删除操作 (第vocab_size位，索引 vocab_size)
                if u_mask_sample[pos, vocab_size].item():
                    current_id = x_t_sample[pos].item()
                    ops.append(f"pos{pos}: DELETE id({current_id})")

                # 检查替换操作 (vocab_size+1 ~ 2*vocab_size，偏移+1)
                sub_idx = u_mask_sample[pos, vocab_size+1:2*vocab_size+1].nonzero(as_tuple=False)
                if sub_idx.numel() > 0:
                    token_id = sub_idx[0].item()
                    current_id = x_t_sample[pos].item()
                    ops.append(f"pos{pos}: SUBSTITUTE id({current_id})→id({token_id})")

                # 检查KEEP操作 (最后一位，索引 2*vocab_size+1，即-1)
                if u_mask_sample[pos, -1].item():
                    current_id = x_t_sample[pos].item()
                    ops.append(f"pos{pos}: KEEP id({current_id})")

            # 限制显示数量
            if len(ops) >= max_ops:
                remaining = sum([u_mask_sample[i].any().item() for i in range(pos, x_seq_len)]) - (len(ops) if pos < x_seq_len - 1 else 0)
                if remaining > 0:
                    ops.append(f"... (还有{remaining}个操作)")
                break

        # 记录解码后的操作
        if ops:
            self.log("GT_EDIT_OPS_DECODED",
                    f"Ground Truth编辑操作 ({len(ops)}个): {' | '.join(ops)}",
                    context, level=level)
        else:
            self.log("GT_EDIT_OPS_DECODED",
                    "Ground Truth编辑操作: 无操作（序列完全匹配）",
                    context, level=level)

    # ==================== 束搜索推理日志 ====================

    def log_position_best_actions(self, position_best_map, selected_position, level=3):
        """记录每个位置的最佳操作

        Args:
            position_best_map (dict): {position: ActionProposal}
            selected_position (int): 最终选中的位置
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        positions_info = []
        for pos, prop in sorted(position_best_map.items()):
            marker = " ← 最终选择" if pos == selected_position else ""
            positions_info.append(f"位置{pos}: {prop.action_type}(score={prop.score:.4f}){marker}")

        self.log("POSITION_BEST_ACTIONS",
                f"每个位置的最佳操作 | {' | '.join(positions_info)}",
                "greedy_search", level=level)

    def log_action_probabilities(self, step, position, expr_len,
                                 lambda_ins, lambda_del, lambda_sub, lambda_keep,
                                 context="greedy_search", level=3):
        """记录操作概率分布

        Args:
            step (int): 当前推理步数
            position (int): token 位置
            expr_len (int): 表达式长度
            lambda_ins (float): 插入概率
            lambda_del (float): 删除概率
            lambda_sub (float): 替换概率
            lambda_keep (float): 保持概率
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        self.log(
            "ACTION_PROBABILITIES",
            f"step={step} | 位置{position} | expr_len={expr_len} | "
            f"lambda_ins={lambda_ins:.6f} | "
            f"lambda_del={lambda_del:.6f} | "
            f"lambda_sub={lambda_sub:.6f} | "
            f"lambda_keep={lambda_keep:.6f}",
            context, level=level
        )

    def log_token_position_mapping(self, step, position, token_id, token_name,
                                    lambda_ins, lambda_del, lambda_sub, lambda_keep,
                                    context="greedy_search", level=3):
        """记录token位置映射和模型预测

        位置说明：位置0表示BOS token，位置1、2、3...表示序列中的后续token

        Args:
            step (int): 当前推理步数
            position (int): token在序列中的位置（位置0=BOS，位置1,2,3...=后续token）
            token_id (int): token的ID
            token_name (str): token的名称
            lambda_ins (float): INSERT概率
            lambda_del (float): DELETE概率
            lambda_sub (float): SUBSTITUTE概率
            lambda_keep (float): KEEP概率
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        self.log(
            "TOKEN_POSITION_MAPPING",
            f"step={step} | position={position} | "
            f"token_id={token_id} | token='{token_name}' | "
            f"lambda_ins={lambda_ins:.6f} | lambda_del={lambda_del:.6f} | "
            f"lambda_sub={lambda_sub:.6f} | lambda_keep={lambda_keep:.6f}",
            context, level=level
        )

    def log_action_execution(self, step, position, action_type, token_name, score,
                              context="greedy_search", level=3):
        """记录实际执行的操作

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            step (int): 当前推理步数
            position (int): 操作位置（位置0=BOS，位置1,2,3...=后续token）
            action_type (str): 操作类型（insert/substitute/delete/keep）
            token_name (str): 操作涉及的token名称
            score (float): 操作分数
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        token_str = f"token='{token_name}'" if token_name else ""
        self.log(
            "ACTION_EXECUTION",
            f"step={step} | position={position} | "
            f"action={action_type.upper()} | {token_str} | score={score:.4f}",
            context, level=level
        )

    def log_position_prediction(self, step, pos, token_name,
                                 lambda_ins, lambda_del, lambda_sub, lambda_keep,
                                 best_action, context="greedy_search", level=3):
        """记录每个位置的预测结果（带token信息）

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            step (int): 当前推理步数
            pos (int): 位置索引（位置0=BOS，位置1,2,3...=后续token）
            token_name (str): 该位置的token名称
            lambda_ins (float): INSERT概率
            lambda_del (float): DELETE概率
            lambda_sub (float): SUBSTITUTE概率
            lambda_keep (float): KEEP概率
            best_action (str): 预测的最佳操作
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        # 确定最佳操作的lambda值
        if best_action == "INSERT":
            best_lambda = lambda_ins
        elif best_action == "DELETE":
            best_lambda = lambda_del
        elif best_action == "SUBSTITUTE":
            best_lambda = lambda_sub
        else:  # KEEP
            best_lambda = lambda_keep

        self.log(
            "POSITION_PREDICTION",
            f"step={step} | pos={pos} | token='{token_name}' | "
            f"pred={best_action}(λ_{best_action.lower()}={best_lambda:.6f}) | "
            f"[ins={lambda_ins:.6f}, del={lambda_del:.6f}, sub={lambda_sub:.6f}]",
            context, level=level
        )

    def log_position_after_deletion(self, step, deleted_pos, old_tokens, new_tokens,
                                     context="greedy_search", level=3):
        """记录删除操作后的位置变化

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            step (int): 当前推理步数
            deleted_pos (int): 被删除的位置（位置0=BOS，位置1,2,3...=后续token）
            old_tokens (list): 删除前的token列表
            new_tokens (list): 删除后的token列表
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        # 格式化token列表，显示位置索引（添加BOS token，位置0=BOS）
        old_tokens_with_bos = ['<BOS>'] + list(old_tokens)
        new_tokens_with_bos = ['<BOS>'] + list(new_tokens)
        old_str = ", ".join([f"{i}:{t}" for i, t in enumerate(old_tokens_with_bos)])
        new_str = ", ".join([f"{i}:{t}" for i, t in enumerate(new_tokens_with_bos)])

        self.log(
            "POSITION_AFTER_DELETE",
            f"step={step} | deleted_pos={deleted_pos} | "
            f"old: [{old_str}] | new: [{new_str}]",
            context, level=level
        )

    def log_insert_candidates(self, step, position, position_desc, lambda_ins,
                             top_token_ids, top_probs, tokenizer, top_k=5,
                             context="greedy_search", level=3):
        """记录 INSERT 操作的候选 token

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            step (int): 当前推理步数
            position (int): 位置索引（位置0=BOS，位置1,2,3...=后续token）
            position_desc (str): 位置描述（如 "位置0(开头插入)"）
            lambda_ins (float): 插入概率
            top_token_ids (torch.Tensor): top-k token IDs
            top_probs (torch.Tensor): top-k token 概率
            tokenizer: tokenizer 对象，用于转换 token ID 到 token 名称
            top_k (int): 记录前 k 个候选
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        candidates_info = []
        for idx, (token_idx, prob) in enumerate(zip(top_token_ids[:top_k], top_probs[:top_k])):
            token_id = token_idx.item() if isinstance(token_idx, torch.Tensor) else token_idx
            prob_val = prob.item() if isinstance(prob, torch.Tensor) else prob
            score = lambda_ins * prob_val
            token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            candidates_info.append(f"{idx+1}.{token_name}(prob={prob_val:.6f},score={score:.6f})")

        self.log(
            "INSERT_CANDIDATES",
            f"step={step} | {position_desc} | lambda_ins={lambda_ins:.6f} | "
            f"top_{len(candidates_info)}: {' | '.join(candidates_info)}",
            context, level=level
        )

    def log_substitute_candidates(self, step, position, current_token, lambda_sub,
                                  top_token_ids, top_probs, tokenizer, top_k=5,
                                  context="greedy_search", level=3):
        """记录 SUBSTITUTE 操作的候选 token

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            step (int): 当前推理步数
            position (int): 位置索引（位置0=BOS，位置1,2,3...=后续token）
            current_token (str): 当前 token
            lambda_sub (float): 替换概率
            top_token_ids (torch.Tensor): top-k token IDs
            top_probs (torch.Tensor): top-k token 概率
            tokenizer: tokenizer 对象，用于转换 token ID 到 token 名称
            top_k (int): 记录前 k 个候选
            context (str): 上下文标识
            level (int): 日志级别
        """
        if not self._should_log(level):
            return

        candidates_info = []
        for idx, (token_idx, prob) in enumerate(zip(top_token_ids[:top_k], top_probs[:top_k])):
            token_id = token_idx.item() if isinstance(token_idx, torch.Tensor) else token_idx
            prob_val = prob.item() if isinstance(prob, torch.Tensor) else prob
            score = lambda_sub * prob_val
            token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            candidates_info.append(f"{idx+1}.{token_name}(prob={prob_val:.6f},score={score:.6f})")

        self.log(
            "SUBSTITUTE_CANDIDATES",
            f"step={step} | 位置{position}(当前={current_token}) | lambda_sub={lambda_sub:.6f} | "
            f"top_{len(candidates_info)}: {' | '.join(candidates_info)}",
            context, level=level
        )

    # ==================== 训练监控日志 ====================

    def log_training_action_probabilities(self, batch_idx, sample_idx, position, x_t_value,
                                         lambda_ins, lambda_del, lambda_sub, lambda_keep,
                                         context="", level=2):
        """记录训练时每个位置的操作概率分布

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            batch_idx (int): batch 索引
            sample_idx (int): sample 索引
            position (int): token 位置（位置0=BOS，位置1,2,3...=后续token）
            x_t_value (int): 当前位置的 token ID
            lambda_ins (float): 插入概率
            lambda_del (float): 删除概率
            lambda_sub (float): 替换概率
            lambda_keep (float): 保持概率
            context (str): 上下文标识
            level (int): 日志级别 (默认2=训练调试)
        """
        if not self._should_log(level):
            return

        self.log(
            "TRAIN_ACTION_PROBS",
            f"batch{batch_idx}_sample{sample_idx} | pos{position}(x_t={x_t_value}) | "
            f"lambda_ins={lambda_ins:.6f} | lambda_del={lambda_del:.6f} | "
            f"lambda_sub={lambda_sub:.6f} | lambda_keep={lambda_keep:.6f}",
            context, level=level
        )

    def log_training_substitute_candidates(self, batch_idx, sample_idx, position, x_t_value,
                                          lambda_sub, substitute_logits, tokenizer, top_k=5,
                                          context="", level=2):
        """记录训练时 SUBSTITUTE 操作的候选 token

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            batch_idx (int): batch 索引
            sample_idx (int): sample 索引
            position (int): token 位置（位置0=BOS，位置1,2,3...=后续token）
            x_t_value (int): 当前位置的 token ID
            lambda_sub (float): 替换概率
            substitute_logits (torch.Tensor): [vocab_size] 替换操作的 logits
            tokenizer: tokenizer 对象
            top_k (int): 记录前 k 个候选
            context (str): 上下文标识
            level (int): 日志级别 (默认2=训练调试)
        """
        if not self._should_log(level):
            return

        import torch.nn.functional as F
        substitute_probs = F.softmax(substitute_logits, dim=-1)
        top_probs, top_indices = torch.topk(substitute_probs, k=min(top_k, substitute_logits.shape[0]))

        candidates_info = []
        for idx, (token_idx, prob) in enumerate(zip(top_indices, top_probs)):
            token_id = token_idx.item() if isinstance(token_idx, torch.Tensor) else token_idx
            prob_val = prob.item() if isinstance(prob, torch.Tensor) else prob
            token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            candidates_info.append(f"{token_name}(prob={prob_val:.6f})")

        self.log(
            "TRAIN_SUBSTITUTE_CANDIDATES",
            f"batch{batch_idx}_sample{sample_idx} | pos{position}(x_t={x_t_value}) | "
            f"lambda_sub={lambda_sub:.6f} | top_{len(candidates_info)}: [{', '.join(candidates_info)}]",
            context, level=level
        )

    def log_training_insert_candidates(self, batch_idx, sample_idx, position,
                                      lambda_ins, insert_logits, tokenizer, top_k=5,
                                      context="", level=2):
        """记录训练时 INSERT 操作的候选 token

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            batch_idx (int): batch 索引
            sample_idx (int): sample 索引
            position (int): token 位置（位置0=BOS，位置1,2,3...=后续token）
            lambda_ins (float): 插入概率
            insert_logits (torch.Tensor): [vocab_size] 插入操作的 logits
            tokenizer: tokenizer 对象
            top_k (int): 记录前 k 个候选
            context (str): 上下文标识
            level (int): 日志级别 (默认2=训练调试)
        """
        if not self._should_log(level):
            return

        import torch.nn.functional as F
        insert_probs = F.softmax(insert_logits, dim=-1)
        top_probs, top_indices = torch.topk(insert_probs, k=min(top_k, insert_logits.shape[0]))

        candidates_info = []
        for idx, (token_idx, prob) in enumerate(zip(top_indices, top_probs)):
            token_id = token_idx.item() if isinstance(token_idx, torch.Tensor) else token_idx
            prob_val = prob.item() if isinstance(prob, torch.Tensor) else prob
            token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            candidates_info.append(f"{token_name}(prob={prob_val:.6f})")

        self.log(
            "TRAIN_INSERT_CANDIDATES",
            f"batch{batch_idx}_sample{sample_idx} | pos{position} | "
            f"lambda_ins={lambda_ins:.6f} | top_{len(candidates_info)}: [{', '.join(candidates_info)}]",
            context, level=level
        )

    def log_training_pred_vs_gt(self, batch_idx, sample_idx, position, x_t_value,
                               gt_operation, pred_operation, is_match,
                               context="", level=2):
        """记录训练时预测 vs Ground Truth 的对比

        位置说明：位置0=BOS token，位置1,2,3...=序列中的后续token

        Args:
            batch_idx (int): batch 索引
            sample_idx (int): sample 索引
            position (int): token 位置（位置0=BOS，位置1,2,3...=后续token）
            x_t_value (int): 当前位置的 token ID
            gt_operation (str): Ground Truth 操作描述
            pred_operation (str): 预测操作描述
            is_match (bool): 是否匹配
            context (str): 上下文标识
            level (int): 日志级别 (默认2=训练调试)
        """
        if not self._should_log(level):
            return

        match_symbol = "✓" if is_match else "✗"

        self.log(
            "TRAIN_PRED_VS_GT",
            f"batch{batch_idx}_sample{sample_idx} | pos{position}(x_t={x_t_value}) | "
            f"GT:{gt_operation} | PRED:{pred_operation} | {match_symbol}",
            context, level=level
        )

    # ==================== 前向传播日志 ====================

    def log_forward_debug(self, debug_info, tokenizer, **tensors):
        """通用的前向传播debug信息记录

        Args:
            debug_info (dict): 调试信息字典，包含'context', 'batch_idx', 'sample_idx'等
            tokenizer: tokenizer对象，用于token转换
            **tensors: 要记录的tensor字典，可包含：
                - z0_token_ids: 起始token序列
                - z1_token_ids: 目标token序列
                - condition_embeddings: 条件嵌入
                - x_t: 移除gap后的序列
                - attention_mask: 注意力掩码
                - pred_rates: 预测的操作概率
                - insert_logits: 插入操作的logits
                - substitute_logits: 替换操作的logits
        """
        if not self._should_log(2):
            return

        context = debug_info.get('context', '') if debug_info else ''
        batch_idx = debug_info.get('batch_idx', 0) if debug_info else 0
        sample_idx = debug_info.get('sample_idx', 0) if debug_info else 0

        # 记录所有传入的tensor
        for name, tensor in tensors.items():
            if name in ['pred_rates', 'insert_logits', 'substitute_logits']:
                continue  # 特殊处理
            self.tensor_values(f"{name}_batch{batch_idx}", tensor[sample_idx],
                             context=context, level=2, max_elements=100)

        # 特殊处理：记录SUBSTITUTE和INSERT操作的候选token
        if 'pred_rates' in tensors and 'insert_logits' in tensors and 'substitute_logits' in tensors and 'x_t' in tensors:
            import torch.nn.functional as F
            pred_rates = tensors['pred_rates']
            insert_logits = tensors['insert_logits']
            substitute_logits = tensors['substitute_logits']
            x_t_sample = tensors['x_t'][sample_idx]
            rates_probs = F.softmax(pred_rates[sample_idx], dim=-1)
            insert_logits_sample = insert_logits[sample_idx]
            substitute_logits_sample = substitute_logits[sample_idx]

            # 记录SUBSTITUTE操作的候选token（前5个位置）
            for pos in range(min(5, x_t_sample.shape[0])):
                if x_t_sample[pos].item() == 0:  # pad token
                    break
                lambda_sub = rates_probs[pos, 2].item()
                if lambda_sub > 0.01:
                    self.log_training_substitute_candidates(
                        batch_idx=batch_idx, sample_idx=sample_idx, position=pos,
                        x_t_value=x_t_sample[pos].item(), lambda_sub=lambda_sub,
                        substitute_logits=substitute_logits_sample[pos], tokenizer=tokenizer,
                        top_k=5, context=context, level=2
                    )

            # 记录INSERT操作的候选token（前3个位置）
            for pos in range(min(3, insert_logits_sample.shape[0])):
                lambda_ins = rates_probs[pos, 0].item()
                if lambda_ins > 0.01:
                    self.log_training_insert_candidates(
                        batch_idx=batch_idx, sample_idx=sample_idx, position=pos,
                        lambda_ins=lambda_ins, insert_logits=insert_logits_sample[pos],
                        tokenizer=tokenizer, top_k=5, context=context, level=2
                    )

    def log_compute_loss_debug(self, debug_info, z0, z1_token_ids, x_t, pred_rates, u_cat_x, u_mask_x, vocab_size, tokenizer, attention_mask=None):
        """记录compute_loss的debug信息

        Args:
            debug_info (dict): 调试信息字典，包含'context', 'batch_idx'等
            z0: 起始token序列
            z1_token_ids: 目标token序列
            x_t: 移除gap后的序列
            pred_rates: 预测的操作概率
            u_cat_x: 拼接的操作logits
            u_mask_x: 编辑操作掩码
            vocab_size (int): 词汇表大小
            tokenizer: tokenizer对象，用于token转换
            attention_mask: 注意力掩码（仅记录有效位置）
        """
        if not self._should_log(2):
            return

        import torch.nn.functional as F

        context = debug_info.get('context', '') if debug_info else ''
        batch_idx = debug_info.get('batch_idx', 0) if debug_info else 0
        sample_idx = 0

        # 记录输入tensor
        self.tensor_values(f"GT_z0_batch{batch_idx}", z0[sample_idx], context=context, level=2, max_elements=50)
        self.tensor_values(f"GT_z1_batch{batch_idx}", z1_token_ids[sample_idx], context=context, level=2, max_elements=50)
        self.tensor_values(f"GT_x_t_batch{batch_idx}", x_t[sample_idx], context=context, level=2, max_elements=50)

        # 记录操作概率
        rates_probs = F.softmax(pred_rates, dim=-1)
        lambda_ins = rates_probs[:, :, 0:1]
        lambda_del = rates_probs[:, :, 1:2]
        lambda_sub = rates_probs[:, :, 2:3]

        self.tensor_values(f"pred_lambda_ins_batch{batch_idx}", lambda_ins[sample_idx], context=context, level=2, max_elements=50)
        self.tensor_values(f"pred_lambda_del_batch{batch_idx}", lambda_del[sample_idx], context=context, level=2, max_elements=50)
        self.tensor_values(f"pred_lambda_sub_batch{batch_idx}", lambda_sub[sample_idx], context=context, level=2, max_elements=50)

        # 记录操作概率分布（前10个位置）
        for pos in range(min(10, x_t.shape[1])):
            if x_t[sample_idx, pos].item() == 0:  # pad token
                break

            self.log_training_action_probabilities(
                batch_idx=batch_idx, sample_idx=sample_idx, position=pos,
                x_t_value=x_t[sample_idx, pos].item(),
                lambda_ins=lambda_ins[sample_idx, pos, 0].item(),
                lambda_del=lambda_del[sample_idx, pos, 0].item(),
                lambda_sub=lambda_sub[sample_idx, pos, 0].item(),
                lambda_keep=rates_probs[sample_idx, pos, 3].item(),
                context=context, level=2
            )

        self.log_u_mask_split(f"GT_u_mask", u_mask_x[sample_idx:sample_idx+1], x_t[sample_idx:sample_idx+1],
                            vocab_size, context=context, level=2)

        self.log_edit_operations(
            u_mask_x[sample_idx], x_t[sample_idx], vocab_size,
            context=context, level=2, max_ops=20, pad_token_id=tokenizer.pad_token_id
        )

        # 对比预测 vs Ground Truth
        self.log_pred_vs_gt(sample_idx, batch_idx, context, x_t, u_cat_x, u_mask_x, vocab_size, tokenizer)

        if attention_mask is None:
            self.tensor_values(
                f"pred_u_cat_x_batch{batch_idx}_first5pos",
                u_cat_x[sample_idx, :5, :],
                context=context,
                level=2,
                max_elements=100
            )
        else:
            mask = attention_mask[sample_idx].bool()
            valid_u_cat_x = u_cat_x[sample_idx][mask]
            self.tensor_values(
                f"pred_u_cat_x_batch{batch_idx}_validpos",
                valid_u_cat_x,
                context=context,
                level=2,
                max_elements=5000
            )

        if debug_info and "op_stats" in debug_info:
            op_stats = debug_info["op_stats"]
            self.log(
                "OP_STATS",
                f"batch{batch_idx} | ins={op_stats['ins']} | del={op_stats['del']} | "
                f"sub={op_stats['sub']} | keep={op_stats['keep']} | valid={op_stats['valid']}",
                context,
                level=2
            )

    def log_pred_vs_gt(self, sample_idx, batch_idx, context, x_t, u_cat_x, u_mask_x, vocab_size, tokenizer):
        """记录预测与Ground Truth的对比

        Args:
            sample_idx (int): 样本索引
            batch_idx (int): batch索引
            context (str): 上下文标识
            x_t: 移除gap后的序列
            u_cat_x: 拼接的操作logits
            u_mask_x: 编辑操作掩码
            vocab_size (int): 词汇表大小
            tokenizer: tokenizer对象，用于token转换
        """
        if not self._should_log(2):
            return

        pred_ops_ids = u_cat_x[sample_idx].argmax(dim=-1)

        for pos in range(min(10, x_t.shape[1])):
            if x_t[sample_idx, pos].item() == 0:  # pad token
                break

            gt_op_id = u_mask_x[sample_idx, pos].argmax().item()
            pred_op_id = pred_ops_ids[pos].item()

            gt_op = self.decode_operation(gt_op_id, vocab_size, tokenizer)
            pred_op = self.decode_operation(pred_op_id, vocab_size, tokenizer)

            self.log_training_pred_vs_gt(
                batch_idx=batch_idx, sample_idx=sample_idx, position=pos,
                x_t_value=x_t[sample_idx, pos].item(),
                gt_operation=gt_op, pred_operation=pred_op,
                is_match=(gt_op_id == pred_op_id), context=context, level=2
            )

    @staticmethod
    def decode_operation(op_id, vocab_size, tokenizer):
        """解码操作ID为可读字符串（静态工具方法）

        Args:
            op_id (int): 操作ID
            vocab_size (int): 词汇表大小
            tokenizer: tokenizer对象，用于token转换

        Returns:
            str: 操作描述字符串，如 "INSERT(token_id=5→x1)"
        """
        if op_id < vocab_size:
            token = tokenizer.convert_ids_to_tokens([op_id])[0]
            return f"INSERT(token_id={op_id}→{token})"
        elif op_id == vocab_size:
            return "DELETE"
        elif op_id < 2 * vocab_size + 1:
            sub_token_id = op_id - vocab_size - 1
            sub_token = tokenizer.convert_ids_to_tokens([sub_token_id])[0]
            return f"SUBSTITUTE(token_id={sub_token_id}→{sub_token})"
        else:
            return "KEEP"

    # ==================== 崩溃日志 ====================

    def log_crash(self, step_name, batch_idx, dimension, error, extra_info=None):
        """记录训练崩溃信息到专门的crash日志文件"""
        import traceback
        import sys

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 收集GPU内存信息
        gpu_info = ""
        try:
            if torch.cuda.is_available():
                gpu_info = "\nGPU内存状态:\n"
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    gpu_info += f"  GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB\n"
        except:
            gpu_info = "\nGPU内存信息获取失败\n"

        # 构建崩溃日志
        crash_info = f"""
{'='*60}
训练崩溃 - {step_name}
{'='*60}
时间: {timestamp}
批次: {batch_idx}
维度: {dimension}
步骤: {step_name}
错误类型: {type(error).__name__}
错误信息: {str(error)}
{gpu_info}
异常栈:
{traceback.format_exc()}
"""

        if extra_info:
            crash_info += f"\n额外信息:\n{extra_info}\n"

        crash_info += f"{'='*60}\n"

        # 写入crash日志文件（始终写入，不受enabled限制）
        try:
            with open(self.CRASH_LOG, 'a', encoding='utf-8') as f:
                f.write(crash_info)
        except Exception as e:
            # 如果无法写入crash日志，至少打印到标准错误
            print(f"无法写入崩溃日志: {e}", file=sys.stderr)
            print(crash_info, file=sys.stderr)

        # 同时输出到标准错误
        print(crash_info, file=sys.stderr)


__all__ = ['Logger']

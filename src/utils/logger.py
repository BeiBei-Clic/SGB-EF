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

    MAX_LOG_LINES = 100000
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

    def expression_generate(self, sample_id, expr_str, gen_time_ms):
        """记录表达式生成"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] GENERATE_RANDOM_EXPR '{expr_str}' | time={gen_time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def expression_validation(self, sample_id, expr_str, expr_len, token_count):
        """记录表达式验证通过"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] EXPR_VALIDATION_PASSED '{expr_str}' len={expr_len} tokens={token_count}"
        self._write(msg, self.SAMPLE_LOG)

    def expression_convert(self, sample_id, token_count, convert_time_ms):
        """记录表达式转换"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] EXPR_TO_TREE tokens={token_count} | time={convert_time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def reduction_sequence(self, sample_id, step_count, time_ms):
        """记录删减序列生成"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] REDUCE_SEQUENCE {step_count} steps | time={time_ms:.1f}ms"
        self._write(msg, self.SAMPLE_LOG)

    def corrupt_expression(self, sample_id, step, orig_expr, corrupt_expr, time_ms):
        """记录表达式破坏"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg1 = f"{timestamp} [{sample_id}] CORRUPT_EXPRESSION step{step} | time={time_ms:.1f}ms"
        msg2 = f"{timestamp} [{sample_id}] CORRUPT_RESULT step{step} '{orig_expr}' → '{corrupt_expr}'"
        self._write(msg1, self.SAMPLE_LOG)
        self._write(msg2, self.SAMPLE_LOG)

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

    def eval_curr_expression(self, sample_id, step, success, time_ms, expr_str=""):
        """记录当前表达式评估"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        expr_info = f" expr='{expr_str}'" if expr_str else ""
        msg = f"{timestamp} [{sample_id}] EVAL_CURR_EXPRESSION step{step} success={success} | time={time_ms:.1f}ms{expr_info}"
        self._write(msg, self.SAMPLE_LOG)

    def convert_to_trees(self, sample_id, step, target_tokens, curr_tokens, time_ms):
        """记录转换为树"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] CONVERT_TO_TREES step{step} | time={time_ms:.1f}ms target_tokens={target_tokens} curr_tokens={curr_tokens}"
        self._write(msg, self.SAMPLE_LOG)

    def levenshtein_alignment(self, sample_id, step, z0_len, z1_len, time_ms):
        """记录对齐操作"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] LEVENSHTEIN_ALIGNMENT step{step} | time={time_ms:.1f}ms z0_len={z0_len} z1_len={z1_len}"
        self._write(msg, self.SAMPLE_LOG)

    def residuals_before_clip(self, sample_id, step, min_val, max_val, mean_val):
        """记录裁剪前的residuals统计"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        msg = f"{timestamp} [{sample_id}] RESIDUALS_BEFORE_CLIP step{step} | min={min_val:.6f} max={max_val:.6f} mean={mean_val:.6f}"
        self._write(msg, self.SAMPLE_LOG)

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

    def expression_eval(self, sample_id, expr_str, eval_time_ms, success=True, error_msg=""):
        """记录表达式评估结果"""
        if not self.enabled:
            return

        timestamp = self._get_timestamp()
        status = "OK" if success else f"FAIL: {error_msg}"
        msg = f"{timestamp} [{sample_id}] EVAL {status} {eval_time_ms:.1f}ms"
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

"""简化的日志工具，支持自动轮转"""
import os
import datetime

try:
    import psutil
except ImportError:
    psutil = None

LOG_FILE = "logs/sample_generation.log"
PERF_LOG_FILE = "logs/performance.log"
TRAIN_LOG_FILE = "logs/training.log"
MAX_LOG_LINES = 100000
_log_line_count = {}


def _write_log(msg, filename=LOG_FILE):
    """写入日志，超过限制时自动轮转"""
    global _log_line_count
    os.makedirs("logs", exist_ok=True)

    if filename not in _log_line_count:
        _log_line_count[filename] = sum(1 for _ in open(filename, 'r', errors='ignore')) if os.path.exists(filename) else 0

    if _log_line_count[filename] >= MAX_LOG_LINES:
        # 轮转：保留最近一半
        with open(filename, 'r', errors='ignore') as f:
            lines = f.readlines()[-(MAX_LOG_LINES // 2):]
        with open(filename, 'w') as f:
            f.writelines(lines)
        _log_line_count[filename] = len(lines)

    with open(filename, "a") as f:
        f.write(msg + "\n")
    _log_line_count[filename] += 1


def log_sample_step(sample_id, step, info=""):
    """记录样本步骤（跳过常规步骤）"""
    if not info and any(step.startswith(s) for s in ["生成数据点", "计算", "处理删减", "生成当前", "生成删减"]):
        return
    _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] {step}" + (f" - {info}" if info else ""))

def log_expression_eval(sample_id, expr_str, eval_time_ms, success=True, error_msg=""):
    status = "OK" if success else f"FAIL: {error_msg}"
    _write_log(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} [{sample_id}] EVAL {status} {eval_time_ms:.1f}ms")

def log_batch_progress(batch_idx, total_batches, samples_completed, total_samples, avg_time_per_sample=0, success_rate=0):
    mem = f"{psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024}MB" if psutil else "N/A"
    pct = samples_completed / total_samples * 100 if total_samples else 0
    msg = f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} BATCH [{batch_idx+1}/{total_batches}] {samples_completed}/{total_samples} ({pct:.1f}%) mem={mem}"
    _write_log(msg, PERF_LOG_FILE)


# ==================== 训练日志记录功能 ====================

def log_training_start(args):
    """记录训练开始"""
    import torch
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = f" | GPUs: {gpu_count}"
        for i in range(min(gpu_count, 2)):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_info += f" | GPU{i}: {gpu_name} ({gpu_memory:.1f}GB)"

    _write_log(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} TRAINING START | Model: {getattr(args, 'model_name', 'Unknown')}{gpu_info}", TRAIN_LOG_FILE)


def log_training_epoch(epoch, total_epochs, loss=None, lr=None, additional_info=""):
    """记录训练epoch信息"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    msg = f"{timestamp} EPOCH [{epoch+1}/{total_epochs}]"
    if loss is not None:
        msg += f" loss={loss:.6f}"
    if lr is not None:
        msg += f" lr={lr:.2e}"
    if additional_info:
        msg += f" | {additional_info}"
    _write_log(msg, TRAIN_LOG_FILE)


def log_training_batch(batch_idx, total_batches, loss, dimension=None, step_time=None, gpu_info="", additional_info=""):
    """记录训练batch信息"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    msg = f"{timestamp} BATCH [{batch_idx}/{total_batches}] loss={loss:.6f}"
    if dimension is not None:
        msg += f" dim={dimension}"
    if step_time is not None:
        msg += f" time={step_time:.3f}s"
    if gpu_info:
        msg += f" gpu={gpu_info}"
    if additional_info:
        msg += f" | {additional_info}"
    _write_log(msg, TRAIN_LOG_FILE)


def log_training_step(step_type, details="", context="", debug_level=0):
    """记录训练步骤详细信息"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    msg = f"{timestamp} {step_type}"
    if context:
        msg += f" [{context}]"
    if details:
        msg += f" | {details}"

    if debug_level <= 1:
        _write_log(msg, TRAIN_LOG_FILE)
    elif debug_level == 2:
        _write_log(msg, TRAIN_LOG_FILE.replace('.log', '_debug.log'))
    else:
        _write_log(msg, TRAIN_LOG_FILE.replace('.log', '_verbose.log'))


def log_model_info(model, additional_info=""):
    """记录模型信息"""
    import torch
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    msg = f"{timestamp} MODEL_INFO | Total params: {total_params:,} | Trainable: {trainable_params:,}"
    if additional_info:
        msg += f" | {additional_info}"
    _write_log(msg, TRAIN_LOG_FILE)

    if hasattr(model, '__class__'):
        model_str = str(model)
        _write_log(f"{timestamp} MODEL_STRUCTURE | {model.__class__.__name__}\n{model_str}",
                  TRAIN_LOG_FILE.replace('.log', '_model.log'))


def log_tensor_info(tensor_name, tensor, level=1, context=""):
    """记录张量信息用于调试"""
    import torch
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

    if level == 1:
        _write_log(msg, TRAIN_LOG_FILE)
    elif level == 2:
        _write_log(msg, TRAIN_LOG_FILE.replace('.log', '_debug.log'))
    else:
        _write_log(msg, TRAIN_LOG_FILE.replace('.log', '_verbose.log'))


def log_gpu_memory_usage(context=""):
    """记录GPU内存使用情况"""
    import torch
    if torch.cuda.is_available():
        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = f"{timestamp} GPU_MEMORY"
        if context:
            msg += f" [{context}]"

        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            msg += f" | GPU{i}: {allocated:.1f}GB/{total:.1f}GB ({allocated/total*100:.1f}%)"

        _write_log(msg, TRAIN_LOG_FILE)


def log_training_error(error_type, error_msg, context=""):
    """记录训练过程中的错误"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    msg = f"{timestamp} ERROR {error_type}"
    if context:
        msg += f" [{context}]"
    msg += f" | {error_msg}"
    _write_log(msg, TRAIN_LOG_FILE)


def log_training_complete(total_epochs, total_time=None, final_loss=None, additional_info=""):
    """记录训练完成"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f"{timestamp} TRAINING COMPLETE | Total epochs: {total_epochs}"
    if total_time is not None:
        msg += f" | Total time: {total_time:.2f}s"
    if final_loss is not None:
        msg += f" | Final loss: {final_loss:.6f}"
    if additional_info:
        msg += f" | {additional_info}"
    _write_log(msg, TRAIN_LOG_FILE)
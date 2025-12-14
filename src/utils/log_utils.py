"""简化的日志工具，支持自动轮转"""
import os
import datetime

try:
    import psutil
except ImportError:
    psutil = None

LOG_FILE = "logs/sample_generation.log"
PERF_LOG_FILE = "logs/performance.log"
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


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_sample_step(sample_id, step, info=""):
    """记录样本步骤（跳过常规步骤）"""
    if not info and any(step.startswith(s) for s in ["生成数据点", "计算", "处理删减", "生成当前", "生成删减"]):
        return
    _write_log(f"{_ts()} [{sample_id}] {step}" + (f" - {info}" if info else ""))


def log_sample_success(sample_id):
    _write_log(f"{_ts()} [{sample_id}] SUCCESS")


def log_sample_stuck(sample_id, duration, steps):
    _write_log(f"{_ts()} [{sample_id}] STUCK {duration:.1f}s steps={len(steps)}")


def cleanup_successful_logs():
    pass  # 已由自动轮转处理


def log_expression_generation(sample_id, expr_str, max_depth, complexity_score=0, extra_info=""):
    _write_log(f"{_ts()} [{sample_id}] EXPR_GEN '{expr_str}' len={len(expr_str)}" + (f" | {extra_info}" if extra_info else ""))


def log_expression_eval(sample_id, expr_str, eval_time_ms, success=True, error_msg=""):
    status = "OK" if success else f"FAIL: {error_msg}"
    _write_log(f"{_ts()} [{sample_id}] EVAL {status} {eval_time_ms:.1f}ms")


def log_retry_attempt(sample_id, retry_num, max_retries, reason, extra_info=""):
    _write_log(f"{_ts()} [{sample_id}] RETRY {retry_num}/{max_retries} {reason}")


def log_batch_progress(batch_idx, total_batches, samples_completed, total_samples, avg_time_per_sample=0, success_rate=0):
    mem = f"{psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024}MB" if psutil else "N/A"
    pct = samples_completed / total_samples * 100 if total_samples else 0
    msg = f"{_ts()} BATCH [{batch_idx+1}/{total_batches}] {samples_completed}/{total_samples} ({pct:.1f}%) mem={mem}"
    _write_log(msg, PERF_LOG_FILE)


def log_reduction_sequence(sample_id, original_expr, reduction_seq, final_expr):
    _write_log(f"{_ts()} [{sample_id}] REDUCE {len(reduction_seq)} steps")


def log_data_generation_stats(sample_id, n_points, input_dim, data_range=(-5.0, 5.0)):
    pass  # 不需要记录


def log_timeout_occurred(sample_id, operation, timeout_seconds, extra_info=""):
    _write_log(f"{_ts()} [{sample_id}] TIMEOUT {operation} >{timeout_seconds}s" + (f" | {extra_info}" if extra_info else ""))


def log_detailed_error(sample_id, operation, error, extra_context=""):
    _write_log(f"{_ts()} [{sample_id}] ERROR {operation}: {type(error).__name__}: {error}")

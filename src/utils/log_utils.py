import os
import datetime

try:
    import psutil
except ImportError:
    psutil = None

LOG_FILE = "logs/sample_generation.log"
STUCK_LOG_FILE = "logs/sample_stuck.log"
PERF_LOG_FILE = "logs/performance.log"


def _get_timestamp():
    """获取格式化的时间戳"""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _write_log(msg, filename=LOG_FILE):
    """统一的日志写入函数"""
    os.makedirs("logs", exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_sample_step(sample_id, step, info=""):
    """记录样本生成步骤"""
    msg = f"{_get_timestamp()} [{sample_id}] {step}"
    if info:
        msg += f" - {info}"
    _write_log(msg)


def log_sample_success(sample_id):
    """记录样本成功完成"""
    _write_log(f"{_get_timestamp()} [{sample_id}] SUCCESS")


def log_sample_stuck(sample_id, duration, steps):
    """记录卡住的样本"""
    _write_log(
        f"卡住样本记录:\n  时间: {_get_timestamp()}\n  样本ID: {sample_id}\n"
        f"  持续时间: {duration:.2f}秒\n  步骤: {steps}\n" + "=" * 50,
        STUCK_LOG_FILE
    )


def cleanup_successful_logs():
    """清理已完成样本的详细日志"""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) < 50 * 1024 * 1024:
        return

    print("日志文件过大，直接清空")
    _write_log(f"# 日志已清理 - {datetime.datetime.now()}")


def log_expression_generation(sample_id, expr_str, max_depth, complexity_score=0):
    """记录表达式生成"""
    ops_count = expr_str.count('(') + expr_str.count('sin') + expr_str.count('cos') + \
                expr_str.count('exp') + expr_str.count('log') + expr_str.count('sqrt')
    msg = f"{_get_timestamp()} [{sample_id}] EXPR_GEN - '{expr_str}' | len={len(expr_str)} | ops={ops_count} | depth≤{max_depth}"
    if complexity_score:
        msg += f" | score={complexity_score}"
    _write_log(msg)


def log_expression_eval(sample_id, expr_str, eval_time_ms, success=True, error_msg=""):
    """记录表达式计算"""
    if success:
        msg = f"{_get_timestamp()} [{sample_id}] EXPR_EVAL_OK - {expr_str} | {eval_time_ms:.1f}ms"
    else:
        msg = f"{_get_timestamp()} [{sample_id}] EXPR_EVAL_FAIL - {expr_str} | {eval_time_ms:.1f}ms | {error_msg}"
    _write_log(msg)


def log_retry_attempt(sample_id, retry_num, max_retries, reason):
    """记录重试"""
    _write_log(f"{_get_timestamp()} [{sample_id}] RETRY {retry_num}/{max_retries} - {reason}")


def log_batch_progress(batch_idx, total_batches, samples_completed, total_samples,
                      avg_time_per_sample=0, success_rate=0):
    """记录批次进度"""
    progress_pct = (samples_completed / total_samples) * 100 if total_samples > 0 else 0
    mem_info = f"mem={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.0f}MB" if psutil else "mem=N/A"

    msg = f"{_get_timestamp()} BATCH_PROGRESS [{batch_idx+1}/{total_batches}] " \
          f"samples={samples_completed}/{total_samples} ({progress_pct:.1f}%) " \
          f"avg_time={avg_time_per_sample:.2f}s succ_rate={success_rate:.1%} {mem_info}"

    _write_log(msg, PERF_LOG_FILE)
    _write_log(msg, LOG_FILE)


def log_reduction_sequence(sample_id, original_expr, reduction_seq, final_expr):
    """记录删减序列"""
    _write_log(f"{_get_timestamp()} [{sample_id}] REDUCTION_SEQ - {len(reduction_seq)} steps: "
               f"'{original_expr}' → ... → '{final_expr}'")


def log_data_generation_stats(sample_id, n_points, input_dim, data_range=(-5.0, 5.0)):
    """记录数据生成统计"""
    _write_log(f"{_get_timestamp()} [{sample_id}] DATA_GEN - {n_points} points, {input_dim}dim, range={data_range}")


def log_timeout_occurred(sample_id, operation, timeout_seconds):
    """记录超时"""
    _write_log(f"{_get_timestamp()} [{sample_id}] TIMEOUT - {operation} exceeded {timeout_seconds}s")

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
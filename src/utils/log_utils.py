import os
import time
import datetime


LOG_FILE = "logs/sample_generation.log"
STUCK_LOG_FILE = "logs/sample_stuck.log"


def log_sample_step(sample_id, step, info=""):
    """记录样本生成步骤"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"{timestamp} [{sample_id}] {step}"
    if info:
        log_msg += f" - {info}"

    # 确保logs目录存在
    os.makedirs("logs", exist_ok=True)

    # 写入详细日志
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def log_sample_success(sample_id):
    """记录样本成功完成，并清理详细日志"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    success_msg = f"{timestamp} [{sample_id}] SUCCESS"

    # 写入成功标记
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(success_msg + "\n")


def log_sample_stuck(sample_id, duration, steps):
    """记录卡住的样本到专门的stuck日志"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    stuck_info = {
        "timestamp": timestamp,
        "sample_id": sample_id,
        "duration_seconds": round(duration, 2),
        "steps": steps
    }

    # 写入卡住日志
    with open(STUCK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"卡住样本记录:\n")
        f.write(f"  时间: {timestamp}\n")
        f.write(f"  样本ID: {sample_id}\n")
        f.write(f"  持续时间: {duration:.2f}秒\n")
        f.write(f"  步骤: {steps}\n")
        f.write("=" * 50 + "\n")


def cleanup_successful_logs():
    """清理已完成样本的详细日志，保留卡住的"""
    if not os.path.exists(LOG_FILE):
        return

    start_time = time.time()
    max_cleanup_time = 30  # 最多30秒清理时间

    try:
        stuck_samples = set()
        if os.path.exists(STUCK_LOG_FILE):
            # 从卡住日志中提取卡住的样本ID，限制读取时间
            with open(STUCK_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if time.time() - start_time > max_cleanup_time:
                        print("日志清理超时，跳过此次清理")
                        return
                    if "样本ID:" in line:
                        sample_id = line.split("样本ID: ")[1].strip()
                        stuck_samples.add(sample_id)

        # 检查日志文件大小，如果太大则直接清空
        if os.path.getsize(LOG_FILE) > 50 * 1024 * 1024:  # 50MB
            print("日志文件过大，直接清空成功样本日志")
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"# 日志已清理 - {datetime.datetime.now()}\n")
            return

        # 更高效的过滤：直接写新文件而不是重读
        temp_file = LOG_FILE + ".tmp"
        with open(LOG_FILE, "r", encoding="utf-8") as infile, \
             open(temp_file, "w", encoding="utf-8") as outfile:

            outfile.write(f"# 清理后的日志 - {datetime.datetime.now()}\n")

            for line in infile:
                if time.time() - start_time > max_cleanup_time:
                    print("日志清理超时，终止清理")
                    break

                # 简化过滤逻辑：只过滤包含SUCCESS的行
                if "SUCCESS" not in line:
                    outfile.write(line)

        # 替换原文件
        os.replace(temp_file, LOG_FILE)

    except Exception as e:
        print(f"清理日志时出错: {e}")
        # 如果清理失败，直接清空日志文件避免卡住
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"# 日志清理失败后重置 - {datetime.datetime.now()}\n")
        except:
            pass

import signal


class TimeoutError(Exception):
    """自定义超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("操作超时")


def with_timeout(func, timeout_seconds, *args, **kwargs):
    """为函数调用添加超时保护"""
    if hasattr(signal, 'SIGALRM'):  # Unix系统支持
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # 取消超时
            return result
        except TimeoutError:
            signal.alarm(0)  # 取消超时
            raise TimeoutError(f"函数 {func.__name__} 在 {timeout_seconds} 秒后超时")
        finally:
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
    else:
        # Windows系统不支持signal.SIGALRM，直接调用函数
        return func(*args, **kwargs)

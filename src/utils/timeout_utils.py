import signal


class TimeoutError(Exception):
    """自定义超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("操作超时")


def with_timeout(func, timeout_seconds, *args, **kwargs):
    """为函数调用添加超时保护"""
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            return func(*args, **kwargs)
        except TimeoutError:
            raise TimeoutError(f"函数 {func.__name__} 在 {timeout_seconds} 秒后超时")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        return func(*args, **kwargs)

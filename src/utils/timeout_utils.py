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

        cleanup_done = False

        try:
            result = func(*args, **kwargs)
            # 函数正常完成，立即清理（关键改进）
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            cleanup_done = True
            return result
        except TimeoutError:
            raise TimeoutError(f"函数 {func.__name__} 在 {timeout_seconds} 秒后超时")
        finally:
            # 只处理未清理的情况（异常时）
            if not cleanup_done:
                try:
                    signal.alarm(0)
                except:
                    pass
                try:
                    signal.signal(signal.SIGALRM, old_handler)
                except:
                    pass
    else:
        return func(*args, **kwargs)

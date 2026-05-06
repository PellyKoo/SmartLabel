"""
PyInstaller runtime hook: 确保 sys.stdout / sys.stderr 始终可写。

在 windowed 模式 (console=False) 下，sys.stdout 和 sys.stderr 可能为 None。
某些库（如 numpy.f2py.cfuncs、nltk 等）会在 import 阶段调用
sys.stdout.write(...)，导致 AttributeError。

本 hook 在所有用户代码之前执行，注入一个空写入器兜底。
"""
import sys


class _NullWriter:
    """无操作的写入器，吞掉所有输出"""
    def write(self, *args, **kwargs):
        return 0

    def flush(self, *args, **kwargs):
        pass

    def isatty(self):
        return False

    def fileno(self):
        # 部分库会调 fileno()，给个无效 fd 但不抛异常
        raise OSError("stdout/stderr is null in windowed mode")

    def close(self):
        pass

    def writable(self):
        return True

    def readable(self):
        return False


if sys.stdout is None:
    sys.stdout = _NullWriter()
if sys.stderr is None:
    sys.stderr = _NullWriter()

import logging
import sys


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def _init_root_logger(level: str = "INFO"):
    """初始化根 logger，仅执行一次。"""
    global _initialized
    if _initialized:
        return
    root = logging.getLogger("smartlabel")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    _initialized = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    获取命名 logger。

    Args:
        name: logger 名称，通常传 __name__
        level: 日志级别

    Returns:
        logging.Logger 实例
    """
    _init_root_logger(level)
    if not name.startswith("smartlabel"):
        name = f"smartlabel.{name}"
    return logging.getLogger(name)

"""统一日志配置。

- 控制台输出 INFO 级别（简洁进度）；
- 文件 logs/opentsbench.log 记录 DEBUG 级别（完整细节，按天滚动、保留 14 天）。

使用方式::

    from tsbench.log import setup_logging, get_logger

    logger = get_logger(__name__)   # 各模块顶部取 logger
    setup_logging()                 # 入口处（cli.py / server）配置一次

未调用 setup_logging 时 get_logger 也可用（无 handler，交给 root logger）。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

ROOT_LOGGER_NAME = "tsbench"
_DEFAULT_LOG_DIR = os.environ.get("OPENTSB_LOG_DIR", "logs")
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """模块级 logger：统一挂在 'tsbench' 命名空间下。"""
    if not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    console_level: Optional[str] = None,
) -> str:
    """初始化日志（幂等，重复调用只更新级别）。

    返回日志文件路径。
    """
    global _configured
    log_dir = log_dir or _DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "opentsbench.log")

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level or os.environ.get("OPENTSB_LOG_LEVEL", "DEBUG"))

    file_fmt = logging.Formatter(_DEFAULT_FORMAT)

    if not _configured:
        # 文件 handler：DEBUG 全量，按天滚动
        fh = logging.handlers.TimedRotatingFileHandler(
            logfile, when="midnight", backupCount=14, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        root.addHandler(fh)

        # 控制台 handler：INFO，格式更短
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level or logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(ch)

        # 不向 root logger 重复传播（uvicorn 等已有自己的日志）
        root.propagate = False
        _configured = True
    else:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.handlers.TimedRotatingFileHandler
            ):
                h.setLevel(console_level or h.level)

    return logfile

"""一键加载全部内置插件（触发各模块的 @register 装饰器）。"""
from __future__ import annotations

_loaded = False


def load_builtin_plugins() -> None:
    global _loaded
    if _loaded:
        return
    import tsbench.datasets  # noqa: F401
    import tsbench.metrics  # noqa: F401
    import tsbench.models  # noqa: F401

    _loaded = True

"""指标抽象。

指标是满足以下签名的普通函数::

    fn(y_true, y_pred, *, y_train=None, seasonal_period=1) -> float
    # y_true/y_pred: (N, H, D)；y_train: (T_train, D)，供 MASE 等指标使用

通过 @register_metric("name", higher_is_better=False) 注册为插件，
元信息会暴露给 API/前端（排行榜据此决定排序与高亮方向）。
"""
from __future__ import annotations

from typing import Callable

from .registry import METRICS
from .types import MetricSpec


def register_metric(name: str, higher_is_better: bool = False) -> Callable:
    """把指标函数包装成 MetricSpec 并注册。"""

    def decorator(fn: Callable) -> Callable:
        spec = MetricSpec(name=name, fn=fn, higher_is_better=higher_is_better)
        METRICS.register(name, higher_is_better=higher_is_better)(spec)
        return fn

    return decorator

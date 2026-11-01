"""时序预测常用指标插件。

所有指标函数签名统一: fn(y_true, y_pred, *, y_train=None, seasonal_period=1) -> float
  y_true/y_pred: (N, H, D)；y_train: (T, D) 训练段（MASE 等需要）。
数值在传入前已按 evaluation 层的口径切好通道；指标本身不做 NaN 处理以外的假设。
"""
from __future__ import annotations

import numpy as np

from ..core.base_metric import register_metric

_EPS = 1e-8


@register_metric("mse", higher_is_better=False)
def mse(y_true, y_pred, **_) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


@register_metric("rmse", higher_is_better=False)
def rmse(y_true, y_pred, **_) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@register_metric("mae", higher_is_better=False)
def mae(y_true, y_pred, **_) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@register_metric("mape", higher_is_better=False)
def mape(y_true, y_pred, **_) -> float:
    """MAPE(%)，剔除 |y_true| 过小的点避免除零爆炸。"""
    mask = np.abs(y_true) > _EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


@register_metric("smape", higher_is_better=False)
def smape(y_true, y_pred, **_) -> float:
    """对称 MAPE(%)，分母为 (|y|+|ŷ|)/2。"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > _EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


@register_metric("mase", higher_is_better=False)
def mase(y_true, y_pred, *, y_train=None, seasonal_period=1, **_) -> float:
    """MASE：MAE 与训练集上季节朴素预测的 MAE 之比（<1 表示优于季节朴素）。"""
    if y_train is None or len(y_train) <= seasonal_period:
        return float("nan")
    naive_err = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    if naive_err < _EPS:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_err)


@register_metric("r2", higher_is_better=True)
def r2(y_true, y_pred, **_) -> float:
    """决定系数 R²（越高越好，1 为完美预测）。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < _EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

"""时序预测常用指标插件。

所有指标函数签名统一: fn(y_true, y_pred, *, y_train=None, seasonal_period=1) -> float
  y_true/y_pred: (N, H, D)；y_train: (T, D) 训练段（MASE 等需要）。
数值在传入前已按 evaluation 层的口径切好通道；指标本身不做 NaN 处理以外的假设。
"""
from __future__ import annotations

import numpy as np

from ..core.base_metric import register_metric

_EPS = 1e-8


@register_metric(
    "mse",
    higher_is_better=False,
    title="均方误差",
    description="衡量预测值与真实值之间的平均平方偏差，对大误差更敏感。",
    formula=r"\mathrm{MSE}=\frac{1}{NHD}\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}(y_{ijk}-\hat{y}_{ijk})^2",
    calculation="计算每个窗口、预测步长和通道的误差平方，再对全部 N×H×D 个误差项取平均。",
    note="单位为数据量纲的平方；当前实验使用标准化后的数值计算。",
)
def mse(y_true, y_pred, **_) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


@register_metric(
    "rmse",
    higher_is_better=False,
    title="均方根误差",
    description="MSE 的平方根，同样重视大误差，但数值尺度与原始数据一致。",
    formula=r"\mathrm{RMSE}=\sqrt{\frac{1}{NHD}\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}(y_{ijk}-\hat{y}_{ijk})^2}=\sqrt{\mathrm{MSE}}",
    calculation="先按 MSE 的方式计算平均平方误差，再开平方。",
    note="便于与 MAE 比较；RMSE 明显大于 MAE 时，通常说明存在较大的个别误差。",
)
def rmse(y_true, y_pred, **_) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@register_metric(
    "mae",
    higher_is_better=False,
    title="平均绝对误差",
    description="直接衡量预测值与真实值之间的平均绝对距离，直观且对异常值相对稳健。",
    formula=r"\mathrm{MAE}=\frac{1}{NHD}\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}\left|y_{ijk}-\hat{y}_{ijk}\right|",
    calculation="计算所有误差项的绝对值，再对 N×H×D 个结果取平均。",
    note="与标准化后的数据处于相同量纲；所有误差采用相同权重。",
)
def mae(y_true, y_pred, **_) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@register_metric(
    "mape",
    higher_is_better=False,
    title="平均绝对百分比误差",
    description="用百分比表示绝对误差相对于真实值的大小，适合真实值远离零的数据。",
    formula=r"\mathrm{MAPE}=\frac{100}{|\mathcal{M}|}\sum_{(i,j,k)\in\mathcal{M}}\left|\frac{y_{ijk}-\hat{y}_{ijk}}{y_{ijk}}\right|,\quad \mathcal{M}=\{(i,j,k):|y_{ijk}|>10^{-8}\}",
    calculation="逐点计算绝对误差除以真实值绝对值，剔除 |y|≤10⁻⁸ 的点，对剩余结果取平均并乘以 100。",
    note="真实值接近零时百分比会非常大。当前指标基于标准化数据，因此 MAPE 可能达到数百甚至数千，建议结合 MSE、MAE 判断。",
)
def mape(y_true, y_pred, **_) -> float:
    """MAPE(%)，剔除 |y_true| 过小的点避免除零爆炸。"""
    mask = np.abs(y_true) > _EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


@register_metric(
    "smape",
    higher_is_better=False,
    title="对称平均绝对百分比误差",
    description="同时用真实值和预测值的绝对值作为分母，降低只使用真实值造成的不对称性。",
    formula=r"\mathrm{sMAPE}=\frac{100}{|\mathcal{M}|}\sum_{(i,j,k)\in\mathcal{M}}\frac{2\left|y_{ijk}-\hat{y}_{ijk}\right|}{|y_{ijk}|+|\hat{y}_{ijk}|}",
    calculation="逐点计算绝对误差，再除以真实值与预测值绝对值的平均数；剔除分母 ≤10⁻⁸ 的点，取平均并乘以 100。",
    note="通常位于 0～200% 之间，但仍不适合直接比较量纲或数据分布差异很大的数据集。",
)
def smape(y_true, y_pred, **_) -> float:
    """对称 MAPE(%)，分母为 (|y|+|ŷ|)/2。"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > _EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


@register_metric(
    "mase",
    higher_is_better=False,
    title="平均绝对标度误差",
    description="用训练集上的季节朴素预测误差对 MAE 进行标度，判断模型是否优于简单季节基线。",
    formula=r"\mathrm{MASE}=\frac{\frac{1}{NHD}\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}|y_{ijk}-\hat{y}_{ijk}|}{\frac{1}{D(T-s)}\sum_{k=1}^{D}\sum_{t=s+1}^{T}|x_{t,k}-x_{t-s,k}|}",
    calculation="分子是测试窗口的 MAE；分母仅在训练集上计算，即每个点与 s 步之前数值之差的平均绝对值。",
    note="MASE＜1 表示优于训练集季节朴素基线，=1 表示相当，＞1 表示更差；s 对应任务中的 seasonal_period。",
)
def mase(y_true, y_pred, *, y_train=None, seasonal_period=1, **_) -> float:
    """MASE：MAE 与训练集上季节朴素预测的 MAE 之比（<1 表示优于季节朴素）。"""
    if y_train is None or len(y_train) <= seasonal_period:
        return float("nan")
    naive_err = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    if naive_err < _EPS:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_err)


@register_metric(
    "r2",
    higher_is_better=True,
    title="决定系数",
    description="衡量模型相对于直接预测真实值均值所减少的平方误差比例。",
    formula=r"R^2=1-\frac{\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}(y_{ijk}-\hat{y}_{ijk})^2}{\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}(y_{ijk}-\bar{y})^2},\quad \bar{y}=\frac{1}{NHD}\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}y_{ijk}",
    calculation="分子汇总所有测试点的残差平方，分母汇总真实值相对其全局均值的平方偏差，再计算 1 减去两者之比。",
    note="R²=1 为完美预测，=0 相当于预测全局均值，＜0 表示比均值基线更差；这里对所有窗口和通道统一计算。",
)
def r2(y_true, y_pred, **_) -> float:
    """决定系数 R²（越高越好，1 为完美预测）。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < _EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@register_metric(
    "pearson",
    higher_is_better=True,
    title="皮尔逊相关系数",
    description="衡量预测值与真实值之间的线性相关程度，取值在 -1 到 1 之间。",
    formula=r"r=\frac{\sum_{i=1}^{N}\sum_{j=1}^{H}\sum_{k=1}^{D}(y_{ijk}-\bar{y})(\hat{y}_{ijk}-\bar{\hat{y}})}{\sqrt{\sum_{i,j,k}(y_{ijk}-\bar{y})^2}\sqrt{\sum_{i,j,k}(\hat{y}_{ijk}-\bar{\hat{y}})^2}}",
    calculation="将全部窗口、预测步长和通道展平，分别减去真实值和预测值的全局均值，再计算协方差与两个标准差乘积之比。",
    note="r=1 为完全正线性相关，0 为无线性相关，-1 为完全负线性相关；对整体缩放和平移不敏感，因此应结合 MSE、MAE 判断预测误差大小。任一序列为常数时无定义，返回 NaN。",
)
def pearson(y_true, y_pred, **_) -> float:
    """全部测试点的 Pearson 相关系数（越高越好）。"""
    x = np.asarray(y_true, dtype=np.float64).ravel()
    y = np.asarray(y_pred, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.dot(x_centered, x_centered)) * np.sqrt(
        np.dot(y_centered, y_centered)
    )
    if denom < _EPS:
        return float("nan")
    return float(np.clip(np.dot(x_centered, y_centered) / denom, -1.0, 1.0))

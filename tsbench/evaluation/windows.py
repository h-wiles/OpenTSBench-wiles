"""滑动窗口生成。

评估口径与 Time-Series-Library 等主流基准一致：
- 训练窗口：只在 train 段内取 (context, horizon) 对，stride=1。
- 评估窗口：预测起点 t 落在目标段（val/test）内，context 允许跨段回溯
  （即 test 第一个窗口的 context 取自 val 末尾），保证与公开结果可比。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def train_windows(series: np.ndarray, ctx: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """在单段序列上构建训练窗口: (N, ctx, D), (N, horizon, D)。"""
    n = len(series) - ctx - horizon + 1
    if n <= 0:
        return (
            np.empty((0, ctx, series.shape[1]), dtype=np.float32),
            np.empty((0, horizon, series.shape[1]), dtype=np.float32),
        )
    xs = np.stack([series[i : i + ctx] for i in range(n)])
    ys = np.stack([series[i + ctx : i + ctx + horizon] for i in range(n)])
    return xs.astype(np.float32), ys.astype(np.float32)


def sliding_eval_windows(
    full_series: np.ndarray,
    origin_start: int,
    origin_end: int,
    ctx: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """在完整序列上构建评估窗口。

    full_series: train+val+test 拼接；预测起点 t ∈ [origin_start, origin_end)，
    context = full[t-ctx : t]（可回溯到目标段之前），truth = full[t : t+horizon]。
    """
    contexts, targets = [], []
    for t in range(origin_start, origin_end):
        if t - ctx < 0 or t + horizon > len(full_series):
            continue
        contexts.append(full_series[t - ctx : t])
        targets.append(full_series[t : t + horizon])
    if not contexts:
        d = full_series.shape[1]
        return (
            np.empty((0, ctx, d), dtype=np.float32),
            np.empty((0, horizon, d), dtype=np.float32),
        )
    return (
        np.stack(contexts).astype(np.float32),
        np.stack(targets).astype(np.float32),
    )

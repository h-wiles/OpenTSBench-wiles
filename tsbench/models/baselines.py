"""统计基线模型：评测中必不可少的参照系。

- naive:           直接重复最后一个观测值
- seasonal_naive:  重复上一个季节周期的值（周期可配，如小时级数据 period=24）
- moving_average:  最近 window 个点的均值外推

这类模型无需训练，fit() 直接返回空信息。
"""
from __future__ import annotations

import numpy as np

from ..core import MODELS, BaseModel, DatasetBundle


@MODELS.register("naive", family="baseline", description="重复最后一个观测值")
class NaiveModel(BaseModel):
    def fit(self, bundle: DatasetBundle):
        return {"trainable_params": 0}

    def predict(self, context: np.ndarray) -> np.ndarray:
        last = context[:, -1:, :]  # (B, 1, D)
        return np.repeat(last, self.task.horizon, axis=1)


@MODELS.register("seasonal_naive", family="baseline", description="重复上一季节周期")
class SeasonalNaiveModel(BaseModel):
    def fit(self, bundle: DatasetBundle):
        return {"trainable_params": 0}

    def predict(self, context: np.ndarray) -> np.ndarray:
        period = int(self.config.get("period") or self.task.seasonal_period or 1)
        h = self.task.horizon
        if period >= context.shape[1]:
            period = context.shape[1]
        season = context[:, -period:, :]  # (B, P, D)
        reps = int(np.ceil(h / period))
        tiled = np.tile(season, (1, reps, 1))
        return tiled[:, :h, :]


@MODELS.register("moving_average", family="baseline", description="滑动窗口均值外推")
class MovingAverageModel(BaseModel):
    def fit(self, bundle: DatasetBundle):
        return {"trainable_params": 0}

    def predict(self, context: np.ndarray) -> np.ndarray:
        window = int(self.config.get("window", 24))
        window = max(1, min(window, context.shape[1]))
        mean = context[:, -window:, :].mean(axis=1, keepdims=True)  # (B, 1, D)
        return np.repeat(mean, self.task.horizon, axis=1)

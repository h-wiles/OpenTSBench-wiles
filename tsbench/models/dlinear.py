"""原生 DLinear 实现（Zeng et al., 2023）。

趋势/季节分解 + 两个线性层，是"任何 DL 模型都能接入本框架"的最小完整示例。
结构对齐 Time-Series-Library 版本，结果可直接与 TSLib Adapter 互证。
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..core import MODELS
from .torch_base import TorchForecastModel


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearNet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 moving_avg: int = 25, individual: bool = False):
        super().__init__()
        self.decomp_kernel = _MovingAvg(moving_avg)
        self.individual = individual
        if individual:
            self.linear_seasonal = nn.ModuleList(nn.Linear(seq_len, pred_len) for _ in range(n_channels))
            self.linear_trend = nn.ModuleList(nn.Linear(seq_len, pred_len) for _ in range(n_channels))
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D) -> (B, H, D)
        trend = self.decomp_kernel(x)
        seasonal = x - trend
        seasonal = seasonal.permute(0, 2, 1)  # (B, D, L)
        trend = trend.permute(0, 2, 1)
        if self.individual:
            seasonal_out = torch.stack([lin(seasonal[:, i, :]) for i, lin in enumerate(self.linear_seasonal)], dim=1)
            trend_out = torch.stack([lin(trend[:, i, :]) for i, lin in enumerate(self.linear_trend)], dim=1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)
        return (seasonal_out + trend_out).permute(0, 2, 1)


@MODELS.register("dlinear", family="linear", framework="pytorch",
                 description="原生 DLinear (分解+线性, Zeng et al. 2023)")
class DLinearModel(TorchForecastModel):
    DEFAULTS = {**TorchForecastModel.DEFAULTS, "lr": 5e-3, "epochs": 10,
                "moving_avg": 25, "individual": False}

    def build_network(self, n_channels_in: int, n_channels_out: int) -> nn.Module:
        return DLinearNet(
            seq_len=self.task.context_length,
            pred_len=self.task.horizon,
            n_channels=n_channels_in,
            moving_avg=int(self.config["moving_avg"]),
            individual=bool(self.config["individual"]),
        )

"""Time-Series-Library (thuml/Time-Series-Library) 第三方模型适配器。

这是"任意开源 DL 时序模型接入本框架"的标准做法：
  1. 把第三方仓库 clone 到 third_party/（不修改其任何源码）
  2. sys.path 注入仓库根目录，import 其模型
  3. 把我们统一的 (B, L, D) -> (B, H, D) 约定映射到 TSLib 的
     forward(x_enc, x_mark_enc, x_dec, x_mark_dec) 签名
  4. 用 SimpleNamespace 自动补全 TSLib 的 configs 对象

接入新 TSLib 模型 = 在文件底部 _register() 加一行。

注意：依赖时间戳特征 (x_mark) 的模型（TimesNet/Autoformer 等）暂不在
默认注册列表；需要时可在 wrapper 中注入时间特征后注册（见 README）。
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..core import MODELS
from ..log import get_logger
from .torch_base import TorchForecastModel

logger = get_logger("models.tslib_adapter")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "Time-Series-Library")
)

# TSLib configs 的公共默认值（等价于它 run.py 的 argparse 默认值）
_TSLIB_DEFAULTS: Dict[str, Any] = {
    "task_name": "long_term_forecast",
    "features": "M",
    "label_len": 48,
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "factor": 1,
    "distil": True,
    "dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
    "activation": "gelu",
    "output_attention": False,
    "moving_avg": 25,
    "individual": False,
    "top_k": 5,
    "num_kernels": 6,
    "num_class": 7,
    "use_norm": True,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "down_sampling_layers": 0,
    "down_sampling_window": 1,
    "down_sampling_method": None,
    "seg_len": 48,
    "patch_len": 16,
    "stride": 8,
}

_path_injected = False


def _ensure_repo_importable(repo_path: str) -> None:
    global _path_injected
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"TSLib repo not found at {repo_path}. "
            "Run: git clone https://github.com/thuml/Time-Series-Library third_party/Time-Series-Library"
        )
    if not _path_injected:
        sys.path.insert(0, repo_path)
        _path_injected = True
        logger.info("TSLib 仓库已加入 sys.path: %s", repo_path)


def _build_configs(n_channels: int, task, user_cfg: Dict[str, Any]) -> SimpleNamespace:
    merged = {**_TSLIB_DEFAULTS, **{k: v for k, v in user_cfg.items() if k not in TorchForecastModel.DEFAULTS}}
    merged.update(
        seq_len=task.context_length,
        pred_len=task.horizon,
        enc_in=n_channels,
        dec_in=n_channels,
        c_out=n_channels,
    )
    return SimpleNamespace(**merged)


class _TSLibNetWrapper(nn.Module):
    """把 TSLib 模型的 forward 签名统一成 (B, L, D) -> (B, H, D)。"""

    def __init__(self, net: nn.Module, label_len: int, pred_len: int, n_channels: int):
        super().__init__()
        self.net = net
        self.label_len = label_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        n_params = len(inspect.signature(net.forward).parameters)
        self._simple = n_params == 1  # 例如 TSLib 的 DLinear/PatchTST: forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._simple:
            return self.net(x)
        b = x.shape[0]
        x_mark_enc = None
        x_dec = torch.zeros(b, self.label_len + self.pred_len, self.n_channels, device=x.device)
        x_mark_dec = None
        return self.net(x, x_mark_enc, x_dec, x_mark_dec)


class TSLibForecastModel(TorchForecastModel):
    """TSLib 模型通用适配基类，子类只需指定 TSLIB_MODEL。"""

    TSLIB_MODEL: str = ""

    def build_network(self, n_channels_in: int, n_channels_out: int) -> nn.Module:
        repo = self.config.get("repo_path") or _REPO_ROOT
        _ensure_repo_importable(repo)
        module = importlib.import_module(f"models.{self.TSLIB_MODEL}")
        configs = _build_configs(n_channels_in, self.task, self.config)
        net = module.Model(configs)
        logger.info(
            "TSLib 模型 %s 实例化: seq_len=%d pred_len=%d enc_in=%d 参数量=%d",
            self.TSLIB_MODEL, configs.seq_len, configs.pred_len, configs.enc_in,
            sum(p.numel() for p in net.parameters()),
        )
        return _TSLibNetWrapper(net, configs.label_len, configs.pred_len, n_channels_in)


def _register(tslib_name: str, description: str, extra_defaults: Dict[str, Any] | None = None) -> None:
    """一行接入一个 TSLib 模型。"""

    @MODELS.register(
        f"tslib_{tslib_name.lower()}",
        family="tslib",
        framework="pytorch",
        description=f"TSLib {tslib_name}: {description}",
    )
    class _M(TSLibForecastModel):
        TSLIB_MODEL = tslib_name
        DEFAULTS = {**TorchForecastModel.DEFAULTS, **(extra_defaults or {})}

    _M.__name__ = f"TSLib{tslib_name}Model"


_register("DLinear", "分解线性基线（与原生源 dlinear 可互证）", {"lr": 5e-3})
_register("PatchTST", "Patch Transformer (ICLR 2023)", {"d_model": 128, "n_heads": 16, "e_layers": 3, "dropout": 0.2})
_register("iTransformer", "倒置 Transformer (ICLR 2024)", {"d_model": 128, "d_ff": 128})

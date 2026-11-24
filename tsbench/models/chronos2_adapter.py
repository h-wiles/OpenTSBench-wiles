"""Amazon Chronos-2 第三方模型适配器（零样本预测）。

源码通过 sys.path 从 third_party/chronos-forecasting 注入，不修改第三方仓库。
Chronos-2 原生支持多变量预测：每个窗口的 D 个通道会作为一个多变量任务联合推理。
默认权重 amazon/chronos-2 为 120M 参数（权重文件约 478 MB），首次 fit() 时下载。
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from ..core import MODELS, BaseModel, DatasetBundle
from ..log import get_logger

logger = get_logger("models.chronos2_adapter")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "chronos-forecasting")
)
_DEFAULT_CHECKPOINT = "amazon/chronos-2"


def _ensure_repo_importable(repo_path: str) -> None:
    src_path = os.path.join(os.path.abspath(repo_path), "src")
    if not os.path.isdir(src_path):
        raise FileNotFoundError(
            f"Chronos source not found at {src_path}. "
            "Run: git clone https://github.com/amazon-science/chronos-forecasting.git "
            "third_party/chronos-forecasting"
        )
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.info("Chronos 源码目录已加入 sys.path: %s", src_path)


@MODELS.register(
    "chronos2",
    family="foundation",
    framework="pytorch",
    source_url="https://github.com/amazon-science/chronos-forecasting",
    description="Amazon Chronos-2 120M 零样本多变量预测（Apache-2.0）",
)
class Chronos2Model(BaseModel):
    """加载 Chronos-2 预训练权重，不在评测数据集上训练。"""

    def __init__(self, config: dict[str, Any], task):
        super().__init__(config, task)
        self._pipeline = None

    def fit(self, bundle: DatasetBundle) -> dict[str, Any]:
        repo = self.config.get("repo_path") or _REPO_ROOT
        _ensure_repo_importable(repo)
        try:
            import torch
            from chronos import Chronos2Pipeline
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "Chronos-2 依赖缺失，请执行: pip install "
                "'transformers>=4.41,<6' 'accelerate>=1.1,<2' 'einops>=0.7,<1'"
            ) from e

        checkpoint = self.config.get("checkpoint") or _DEFAULT_CHECKPOINT
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        load_kwargs: dict[str, Any] = {"device_map": device}
        for key in ("revision", "cache_dir", "local_files_only"):
            if key in self.config:
                load_kwargs[key] = self.config[key]

        logger.info("加载 Chronos-2 权重: %s（首次运行需要下载）", checkpoint)
        self._pipeline = Chronos2Pipeline.from_pretrained(checkpoint, **load_kwargs)
        n_params = sum(p.numel() for p in self._pipeline.model.parameters())
        logger.info(
            "Chronos-2 加载完成: 参数量=%d, device=%s, context_limit=%d",
            n_params,
            device,
            self._pipeline.model_context_length,
        )
        return {
            "mode": "zero_shot",
            "checkpoint": checkpoint,
            "backend": "pytorch",
            "device": device,
            "trainable_params": n_params,
            "epochs_run": 0,
        }

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Chronos-2 必须先调用 fit() 加载权重")
        if context.ndim != 3:
            raise ValueError(f"context 应为 (B, L, D)，实际形状为 {context.shape}")

        batch, length, channels = context.shape
        # Chronos-2 的多变量输入约定为 (B, D, L)；整个窗口的所有通道联合预测。
        multivariate = np.ascontiguousarray(
            context.transpose(0, 2, 1), dtype=np.float32
        )
        batch_size = int(self.config.get("batch_size", 64))
        if batch_size < 1:
            raise ValueError("Chronos-2 batch_size 必须 >= 1")

        logger.info(
            "Chronos-2 开始推理: %d 个多变量窗口 × %d 通道，horizon=%d",
            batch,
            channels,
            self.task.horizon,
        )
        _, median = self._pipeline.predict_quantiles(
            inputs=multivariate,
            prediction_length=self.task.horizon,
            quantile_levels=[0.5],
            batch_size=batch_size,
            context_length=length,
            cross_learning=bool(self.config.get("cross_learning", False)),
        )
        points = np.stack([p.detach().cpu().numpy() for p in median], axis=0)
        expected = (batch, channels, self.task.horizon)
        if points.shape != expected:
            raise RuntimeError(f"Chronos-2 输出形状异常: 期望 {expected}，实际 {points.shape}")
        result = points.transpose(0, 2, 1).astype(np.float32)
        return np.ascontiguousarray(result)

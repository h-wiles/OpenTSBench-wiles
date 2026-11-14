"""Google TimesFM 2.5 第三方模型适配器（零样本预测）。

源码通过 sys.path 从 third_party/timesfm 注入，不修改第三方仓库。
TimesFM 2.5 是单变量模型，多通道输入会逐通道预测后恢复为 (B, H, D)。
首次 fit() 会从 Hugging Face 下载约 200M 参数的预训练权重。
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from ..core import MODELS, BaseModel, DatasetBundle
from ..log import get_logger

logger = get_logger("models.timesfm_adapter")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "timesfm")
)
_DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
_CONTEXT_LIMIT = 16384


def _ensure_repo_importable(repo_path: str) -> None:
    src_path = os.path.join(os.path.abspath(repo_path), "src")
    if not os.path.isdir(src_path):
        raise FileNotFoundError(
            f"TimesFM source not found at {src_path}. "
            "Run: git clone https://github.com/google-research/timesfm.git third_party/timesfm"
        )
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.info("TimesFM 源码目录已加入 sys.path: %s", src_path)


@MODELS.register(
    "timesfm",
    family="foundation",
    framework="pytorch",
    source_url="https://github.com/google-research/timesfm",
    description="Google TimesFM 2.5 200M 零样本预测（Apache-2.0，逐通道推理）",
)
class TimesFMModel(BaseModel):
    """加载 TimesFM 预训练权重，不在评测数据集上训练。"""

    def __init__(self, config: dict[str, Any], task):
        super().__init__(config, task)
        self._model = None

    def fit(self, bundle: DatasetBundle) -> dict[str, Any]:
        if self.task.context_length + self.task.horizon > _CONTEXT_LIMIT:
            raise ValueError(
                "TimesFM 2.5 要求 context_length + horizon <= "
                f"{_CONTEXT_LIMIT}，当前为 {self.task.context_length + self.task.horizon}"
            )
        batch_size = int(self.config.get("batch_size", 32))
        if batch_size < 1:
            raise ValueError("TimesFM batch_size 必须 >= 1")

        repo = self.config.get("repo_path") or _REPO_ROOT
        _ensure_repo_importable(repo)
        try:
            from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "TimesFM 依赖缺失，请执行: pip install 'huggingface-hub>=0.30.0' "
                "'safetensors>=0.5.3'"
            ) from e

        checkpoint = self.config.get("checkpoint") or _DEFAULT_CHECKPOINT
        logger.info("加载 TimesFM 2.5 权重: %s（首次运行需要下载）", checkpoint)
        load_kwargs: dict[str, Any] = {
            "torch_compile": bool(self.config.get("torch_compile", False))
        }
        for key in ("revision", "cache_dir", "local_files_only"):
            if key in self.config:
                load_kwargs[key] = self.config[key]
        self._model = TimesFM_2p5_200M_torch.from_pretrained(
            checkpoint, **load_kwargs
        )
        self._model.compile(
            ForecastConfig(
                max_context=self.task.context_length,
                max_horizon=self.task.horizon,
                per_core_batch_size=batch_size,
                normalize_inputs=bool(self.config.get("normalize_inputs", False)),
                infer_is_positive=bool(self.config.get("infer_is_positive", True)),
            )
        )
        n_params = sum(p.numel() for p in self._model.model.parameters())
        logger.info("TimesFM 加载完成: 参数量=%d, batch_size=%d", n_params, batch_size)
        return {
            "mode": "zero_shot",
            "checkpoint": checkpoint,
            "backend": "pytorch",
            "trainable_params": n_params,
            "epochs_run": 0,
        }

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TimesFM 必须先调用 fit() 加载权重")
        if context.ndim != 3:
            raise ValueError(f"context 应为 (B, L, D)，实际形状为 {context.shape}")

        batch, length, channels = context.shape
        # (B, L, D) -> (B, D, L) -> (B*D, L)，保持窗口与通道的对应顺序。
        flattened = np.ascontiguousarray(
            context.transpose(0, 2, 1), dtype=np.float32
        ).reshape(batch * channels, length)
        inputs = [series for series in flattened]
        logger.info("TimesFM 开始推理: %d 窗口 × %d 通道 = %d 条单变量序列", batch, channels, len(inputs))
        points, _ = self._model.forecast(horizon=self.task.horizon, inputs=inputs)
        points = np.asarray(points, dtype=np.float32)
        expected = (batch * channels, self.task.horizon)
        if points.shape != expected:
            raise RuntimeError(f"TimesFM 输出形状异常: 期望 {expected}，实际 {points.shape}")
        return points.reshape(batch, channels, self.task.horizon).transpose(0, 2, 1)

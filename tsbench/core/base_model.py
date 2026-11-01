"""模型抽象基类。

任何深度学习时序模型只需实现两个方法即可接入评测：
  fit(bundle)      —— 训练（无训练的 baseline 可直接返回 {}）
  predict(context) —— (B, L, D) -> (B, H, D_out)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from .types import DatasetBundle, Task


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any], task: Task):
        self.config = dict(config or {})
        self.task = task

    @abstractmethod
    def fit(self, bundle: DatasetBundle) -> Dict[str, Any]:
        """训练模型，返回训练信息（如 epochs、best_val_loss），会存入 meta。"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        """context: (B, context_length, D) -> (B, horizon, D_out) 的 float 数组。"""
        raise NotImplementedError

"""框架内部的标准数据结构。

各层之间只通过这些结构通信，保证解耦：
数据集层产出 DatasetBundle，模型层消费它并产出预测，评测层汇总为指标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DatasetBundle:
    """一个数据集加载后的标准产物。

    - train/val/test: (T, D) 的 float32 数组（val 可为 None）。
      若启用标准化，scaler 只在 train 上 fit。
    - timestamps: 全序列 (T_train + T_val + T_test,) 的时间戳字符串。
    """

    name: str
    train: np.ndarray
    val: Optional[np.ndarray]
    test: np.ndarray
    feature_names: List[str]
    timestamps: Optional[np.ndarray] = None
    freq: str = "h"
    scaler_mean: Optional[np.ndarray] = None  # (D,) 反标准化用
    scaler_std: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_features(self) -> int:
        return self.train.shape[1]

    def full_series(self) -> np.ndarray:
        """train/val/test 拼接后的完整序列（评估窗口的 context 可跨段回溯）。"""
        parts = [self.train] + ([self.val] if self.val is not None else []) + [self.test]
        return np.concatenate(parts, axis=0)

    def test_start_index(self) -> int:
        n = len(self.train)
        if self.val is not None:
            n += len(self.val)
        return n


@dataclass
class Task:
    """一次预测任务的定义。"""

    context_length: int = 96
    horizon: int = 96
    target: Optional[str] = None  # None => 预测全部通道; 否则只评估该通道
    seasonal_period: int = 1      # MASE 等指标使用的季节周期


@dataclass
class MetricSpec:
    """指标的元信息 + 计算函数。"""

    name: str
    fn: Any  # fn(y_true, y_pred, *, y_train=None, seasonal_period=1) -> float
    higher_is_better: bool = False

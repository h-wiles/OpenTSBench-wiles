"""通用 CSV 数据集适配器（ETT 风格：一个 date 列 + 若干数值列）。

配置示例::

    dataset:
      name: csv
      config:
        path: data/ETT/ETTh1.csv
        date_column: date
        split: {type: ett, months: [12, 4, 4]}   # ETT 标准切分
        # split: {type: ratio, train: 0.7, val: 0.1, test: 0.2}
        # split: {type: borders, borders: [0, 8640, 11520, 14400]}
        scale: true

任何"时间戳列 + 数值列"的 CSV（ETT/ECL/Traffic/Weather 导出后）都能直接用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..core import DATASETS, BaseDataset, DatasetBundle
from ..log import get_logger

logger = get_logger("datasets.csv")

# 每个"月"包含的行数（按 ETT 惯例 30 天/月），由频率推断
_FREQ_ROWS_PER_DAY = {"h": 24, "15min": 96, "min": 1440, "d": 1}


def _detect_freq(ts: pd.Series) -> str:
    if len(ts) < 3:
        return "h"
    delta = (ts.iloc[1] - ts.iloc[0]).total_seconds()
    for name, seconds in [("min", 60), ("15min", 900), ("h", 3600), ("d", 86400)]:
        if abs(delta - seconds) < 1:
            return name
    return "h"


def _ett_borders(n_rows: int, freq: str, months: List[int]) -> List[int]:
    rows_per_day = _FREQ_ROWS_PER_DAY.get(freq, 24)
    per_month = 30 * rows_per_day
    b0, b1, b2 = 0, months[0] * per_month, (months[0] + months[1]) * per_month
    b3 = (months[0] + months[1] + months[2]) * per_month
    return [b0, min(b1, n_rows), min(b2, n_rows), min(b3, n_rows)]


@DATASETS.register("csv", description="通用 CSV 数据集（ETT 风格）")
class CSVDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if "path" not in config:
            raise ValueError("csv dataset requires config.path")

    def _split_borders(self, n_rows: int, freq: str) -> List[int]:
        split = dict(self.config.get("split") or {"type": "ratio"})
        stype = split.get("type", "ratio")
        if stype == "ett":
            return _ett_borders(n_rows, freq, split.get("months", [12, 4, 4]))
        if stype == "borders":
            borders = list(split["borders"])
            assert len(borders) == 4, "borders must be [0, train_end, val_end, test_end]"
            return borders
        # ratio
        r_train = float(split.get("train", 0.7))
        r_val = float(split.get("val", 0.1))
        b1 = int(n_rows * r_train)
        b2 = int(n_rows * (r_train + r_val))
        return [0, b1, b2, n_rows]

    def load(self) -> DatasetBundle:
        cfg = self.config
        path = cfg["path"]
        logger.debug("读取 CSV: %s (config=%s)", path, cfg)
        df = pd.read_csv(path)

        date_col = cfg.get("date_column", "date")
        timestamps = None
        freq = cfg.get("freq", "h")
        if date_col in df.columns:
            ts = pd.to_datetime(df[date_col])
            timestamps = ts.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
            freq = cfg.get("freq") or _detect_freq(ts)
            df = df.drop(columns=[date_col])

        # 只保留数值列，丢弃可能的非数值杂列
        df = df.select_dtypes(include=[np.number])
        feature_names = list(df.columns)
        values = df.to_numpy(dtype=np.float64)
        n = len(values)

        b0, b1, b2, b3 = self._split_borders(n, freq)
        train, val, test = values[b0:b1], values[b1:b2], values[b2:b3]
        logger.info(
            "CSV 数据集 %s: 共 %d 行, 切分边界=%s, train=%d val=%d test=%d, freq=%s",
            path, n, [b0, b1, b2, b3], len(train), len(val), len(test), freq,
        )

        mean = std = None
        if cfg.get("scale", True):
            mean = train.mean(axis=0)
            std = train.std(axis=0)
            std[std < 1e-8] = 1.0
            train = (train - mean) / std
            val = (val - mean) / std
            test = (test - mean) / std

        name = cfg.get("name") or str(path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return DatasetBundle(
            name=name,
            train=train.astype(np.float32),
            val=val.astype(np.float32),
            test=test.astype(np.float32),
            feature_names=feature_names,
            timestamps=timestamps,
            freq=freq,
            scaler_mean=None if mean is None else mean.astype(np.float32),
            scaler_std=None if std is None else std.astype(np.float32),
            metadata={"path": path, "borders": [b0, b1, b2, b3]},
        )

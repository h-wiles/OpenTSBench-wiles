"""数据集抽象基类。

任何新数据集（M4、GluonTS、自定义 CSV…）只需实现 load()，
返回标准 DatasetBundle 即可接入整个评测流程。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import DatasetBundle


class BaseDataset(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load(self) -> DatasetBundle:
        """加载并返回标准化的数据集。"""
        raise NotImplementedError

"""插件注册中心。

数据集 / 模型 / 指标三类插件统一通过 Registry 注册与发现，
新增插件无需改动框架核心代码。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

_MISSING = object()


class Registry:
    """一个类型安全的命名注册表。

    用法::

        MODELS = Registry("model")

        @MODELS.register("dlinear", family="linear")
        class DLinearModel(BaseModel): ...

        cls = MODELS.get("dlinear")
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str | None = None, **meta) -> Callable:
        """装饰器：把对象注册到本表，并附带元信息（如指标方向）。"""

        def decorator(obj):
            key = name or obj.__name__.lower()
            if key in self._items:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            obj.registry_meta = {"name": key, **meta}
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str):
        obj = self._items.get(name, _MISSING)
        if obj is _MISSING:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            )
        return obj

    def list(self) -> List[str]:
        return sorted(self._items)

    def describe(self) -> List[Dict[str, Any]]:
        """返回所有插件的元信息（供 API/前端展示）。"""
        return [
            dict(getattr(obj, "registry_meta", {"name": key}))
            for key, obj in sorted(self._items.items())
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._items


DATASETS = Registry("dataset")
MODELS = Registry("model")
METRICS = Registry("metric")

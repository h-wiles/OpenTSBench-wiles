"""实验结果存储层。

文件型存储（JSON + NPZ），无 DB 依赖；接口保持稳定，未来可平滑换成 SQLite。

results/<run_id>/
├── config.yaml    # 完整实验配置
├── meta.json      # 运行状态/耗时/训练信息
├── metrics.json   # 指标结果
└── data.npz       # 全序列 + scaler + 预测（float32 压缩，API 按需重建窗口）
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..log import get_logger

logger = get_logger("evaluation.store")


class ResultStore:
    def __init__(self, root: str = "results"):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # ---------- 写 ----------
    def new_run_id(self, model: str, dataset: str, horizon: int) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        short = uuid.uuid4().hex[:6]
        return f"{ts}_{dataset}_{model}_h{horizon}_{short}"

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.root, run_id)

    def save_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        meta: Dict[str, Any],
        metrics: Dict[str, float],
        arrays: Dict[str, np.ndarray],
    ) -> None:
        d = self.run_dir(run_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        self._write_json(os.path.join(d, "meta.json"), meta)
        self._write_json(os.path.join(d, "metrics.json"), metrics)
        np.savez_compressed(os.path.join(d, "data.npz"), **arrays)
        logger.debug("run 已保存: %s (status=%s)", run_id, meta.get("status"))

    def update_meta(self, run_id: str, **fields) -> None:
        path = os.path.join(self.run_dir(run_id), "meta.json")
        meta = self._read_json(path) if os.path.exists(path) else {}
        meta.update(fields)
        self._write_json(path, meta)

    # ---------- 读 ----------
    def list_runs(
        self, dataset: Optional[str] = None, model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        runs = []
        for run_id in sorted(os.listdir(self.root), reverse=True):
            d = self.run_dir(run_id)
            meta_path = os.path.join(d, "meta.json")
            if not os.path.isdir(d) or not os.path.exists(meta_path):
                continue
            meta = self._read_json(meta_path)
            metrics_path = os.path.join(d, "metrics.json")
            metrics = self._read_json(metrics_path) if os.path.exists(metrics_path) else {}
            row = {**meta, "run_id": run_id, "metrics": metrics}
            if dataset and meta.get("dataset") != dataset:
                continue
            if model and meta.get("model") != model:
                continue
            runs.append(row)
        return runs

    def load_run(self, run_id: str) -> Dict[str, Any]:
        d = self.run_dir(run_id)
        if not os.path.isdir(d):
            raise KeyError(f"run '{run_id}' not found")
        out: Dict[str, Any] = {"run_id": run_id}
        for fname, key in [("meta.json", "meta"), ("metrics.json", "metrics"), ("config.yaml", "config")]:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                if fname.endswith(".yaml"):
                    with open(path) as f:
                        out[key] = yaml.safe_load(f)
                else:
                    out[key] = self._read_json(path)
        # 拍平 meta 到顶层，前端好用
        meta = out.pop("meta", {})
        return {**meta, **out}

    def load_arrays(self, run_id: str) -> Dict[str, np.ndarray]:
        path = os.path.join(self.run_dir(run_id), "data.npz")
        if not os.path.exists(path):
            raise KeyError(f"run '{run_id}' has no data.npz")
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def delete_run(self, run_id: str) -> bool:
        d = self.run_dir(run_id)
        if os.path.isdir(d):
            shutil.rmtree(d)
            logger.info("run 已删除: %s", run_id)
            return True
        return False

    def leaderboard(
        self, dataset: str, horizon: Optional[int] = None, context_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """同一 (dataset, horizon, context) 下、每个模型取最优 run 太主观；
        这里返回该口径下所有完成的 run，排序交给调用方按指标做。"""
        rows = []
        for r in self.list_runs(dataset=dataset):
            if r.get("status") != "completed":
                continue
            task = r.get("task") or {}
            if horizon is not None and task.get("horizon") != horizon:
                continue
            if context_length is not None and task.get("context_length") != context_length:
                continue
            rows.append(r)
        return rows

    # ---------- 工具 ----------
    @staticmethod
    def _write_json(path: str, obj: Any) -> None:
        def default(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(path, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=default)

    @staticmethod
    def _read_json(path: str) -> Any:
        with open(path) as f:
            return json.load(f)

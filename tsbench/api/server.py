"""FastAPI 后端：为前端提供插件信息、实验列表、排行榜、窗口曲线数据。

POST /api/runs 在后台线程中执行实验，前端轮询 run 状态即可。
"""
from __future__ import annotations

import glob
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core import DATASETS, METRICS, MODELS
from ..evaluation.store import ResultStore
from ..evaluation.runner import run_experiment
from ..log import get_logger, setup_logging
from ..plugins import load_builtin_plugins

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logger = get_logger("api.server")


class RunRequest(BaseModel):
    config: Dict[str, Any]


def _clean(obj: Any) -> Any:
    """递归把 NaN/Inf 转成 None —— 严格 JSON (前端 JSON.parse) 不接受 NaN 字面量。"""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def create_app(results_dir: str = "results") -> FastAPI:
    setup_logging()
    load_builtin_plugins()
    logger.info("OpenTSBench API 启动: results_dir=%s", results_dir)
    store = ResultStore(os.path.join(PROJECT_ROOT, results_dir))
    jobs: Dict[str, Dict[str, Any]] = {}  # run_id -> {status, error}

    app = FastAPI(title="OpenTSBench", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- 插件信息 ----------
    @app.get("/api/plugins")
    def plugins() -> Dict[str, Any]:
        return _clean({
            "datasets": DATASETS.describe(),
            "models": MODELS.describe(),
            "metrics": METRICS.describe(),
        })

    @app.get("/api/datasets")
    def datasets() -> List[Dict[str, Any]]:
        """扫描 data/ 下的 CSV，供前端"发起实验"选择。"""
        out = []
        for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "**", "*.csv"), recursive=True)):
            rel = os.path.relpath(path, PROJECT_ROOT)
            out.append({"name": os.path.splitext(os.path.basename(path))[0], "path": rel})
        return out

    # ---------- 实验查询 ----------
    @app.get("/api/runs")
    def runs(dataset: Optional[str] = None, model: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = store.list_runs(dataset=dataset, model=model)
        for r in rows:
            job = jobs.get(r["run_id"])
            if job:
                r["status"] = job["status"]
        return _clean(rows)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> Dict[str, Any]:
        try:
            row = store.load_run(run_id)
        except KeyError:
            raise HTTPException(404, f"run '{run_id}' not found")
        job = jobs.get(run_id)
        if job:
            row["status"] = job["status"]
        return _clean(row)

    @app.get("/api/runs/{run_id}/window")
    def run_window(run_id: str, index: int = 0, channel: Optional[str] = None) -> Dict[str, Any]:
        """单个评估窗口的 context / truth / prediction（已反标准化）。"""
        try:
            arrays = store.load_arrays(run_id)
            run = store.load_run(run_id)
        except KeyError as e:
            raise HTTPException(404, str(e))

        ctx = run["task"]["context_length"]
        horizon = run["task"]["horizon"]
        feature_names = [str(x) for x in arrays["feature_names"]]
        if channel is None or channel not in feature_names:
            channel = feature_names[-1]
        ch = feature_names.index(channel)

        y_true, y_pred = arrays["y_true"], arrays["y_pred"]
        n_windows = len(y_true)
        if n_windows == 0:
            raise HTTPException(400, "run has no stored predictions")
        index = max(0, min(index, n_windows - 1))

        full = arrays["full_series"]
        test_start = int(arrays["test_start"][0])
        origin = test_start + index
        mean, std = float(arrays["scaler_mean"][ch]), float(arrays["scaler_std"][ch])

        def unscale(x):
            return (np.asarray(x, dtype=np.float64) * std + mean).round(4).tolist()

        ts = arrays["timestamps"]
        return _clean({
            "index": index,
            "n_windows": n_windows,
            "channel": channel,
            "channels": feature_names,
            "context": unscale(full[origin - ctx : origin, ch]),
            "truth": unscale(y_true[index, :, ch]),
            "prediction": unscale(y_pred[index, :, ch]),
            "timestamps": [str(t) for t in ts[origin - ctx : origin + horizon]],
        })

    @app.get("/api/runs/{run_id}/series")
    def run_series(
        run_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        channel: Optional[str] = None,
        lead: int = 0,
    ) -> Dict[str, Any]:
        """任意时间区间的 真值 vs 预测 对比序列（已反标准化）。

        预测口径：时刻 u 的预测值取"起点为 u-lead 的窗口"的第 lead 步，
        即模型提前 lead+1 步对 u 做出的预测；lead=0 为滚动 1 步预测。
        """
        try:
            arrays = store.load_arrays(run_id)
            run = store.load_run(run_id)
        except KeyError as e:
            raise HTTPException(404, str(e))

        horizon = run["task"]["horizon"]
        lead = max(0, min(lead, horizon - 1))
        feature_names = [str(x) for x in arrays["feature_names"]]
        if channel not in feature_names:
            channel = feature_names[-1]
        ch = feature_names.index(channel)

        full = arrays["full_series"]
        ts = arrays["timestamps"]
        test_start = int(arrays["test_start"][0])
        n_windows = len(arrays["y_true"])
        first_origin, last_origin = test_start, test_start + n_windows - 1

        def ts_to_idx(s: Optional[str], default: int) -> int:
            if not s:
                return default
            target = np.datetime64(s)
            stamps = ts.astype("datetime64[s]")
            return int(np.argmin(np.abs(stamps - target)))

        # 有效评估区间：每个 u 都必须存在起点 o=u-lead ∈ [first_origin, last_origin] 的窗口
        lo = first_origin + lead
        hi = min(len(full) - 1, last_origin + lead)
        start_idx = max(lo, ts_to_idx(start, lo))
        end_idx = min(hi, ts_to_idx(end, hi))
        if end_idx < start_idx:
            raise HTTPException(
                400,
                f"区间内无可用预测（lead={lead} 时有效范围 "
                f"{str(ts[lo])} ~ {str(ts[hi])}）",
            )

        idx = np.arange(start_idx, end_idx + 1)
        mean, std = float(arrays["scaler_mean"][ch]), float(arrays["scaler_std"][ch])

        def unscale(x: np.ndarray) -> list:
            return (np.asarray(x, dtype=np.float64) * std + mean).round(4).tolist()

        origins = idx - lead  # 每个时刻的预测起点
        return _clean({
            "channel": channel,
            "channels": feature_names,
            "lead": lead,
            "horizon": horizon,
            "n_points": len(idx),
            "valid_range": [str(ts[lo]), str(ts[hi])],
            "timestamps": [str(t) for t in ts[start_idx : end_idx + 1]],
            "truth": unscale(full[start_idx : end_idx + 1, ch]),
            "prediction": unscale(arrays["y_pred"][origins - test_start, lead, ch]),
        })

    @app.get("/api/leaderboard")
    def leaderboard(
        dataset: str, horizon: Optional[int] = None, context_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return _clean(store.leaderboard(dataset, horizon, context_length))

    # ---------- 实验操作 ----------
    @app.post("/api/runs")
    def create_run(req: RunRequest) -> Dict[str, Any]:
        placeholder: List[str] = []

        def on_progress(run_id: str, status: str) -> None:
            jobs[run_id] = {"status": status}

        def worker() -> None:
            try:
                logger.info("后台实验开始: %s", req.config.get("name", ""))
                ids = run_experiment(req.config, results_dir=store.root, on_progress=on_progress)
                placeholder.extend(ids)
                logger.info("后台实验完成: %s", ids)
            except Exception as e:
                logger.exception("后台实验失败: %s", e)
                jobs[f"__batch__{id(req)}"] = {"status": "failed", "error": str(e)}

        threading.Thread(target=worker, daemon=True).start()
        return {"accepted": True, "note": "实验已在后台执行，轮询 /api/runs 查看状态"}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> Dict[str, Any]:
        jobs.pop(run_id, None)
        if not store.delete_run(run_id):
            raise HTTPException(404, f"run '{run_id}' not found")
        logger.info("实验已删除: %s", run_id)
        return {"deleted": run_id}

    return app

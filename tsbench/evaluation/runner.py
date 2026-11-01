"""实验运行器：把 数据集 → 模型 → 滑窗预测 → 指标 → 落盘 串成一条流水线。

一份 YAML 配置可包含多个模型，会生成多个 run（每个模型一个 run_id），
保证排行榜/详情页的数据粒度统一。
"""
from __future__ import annotations

import platform
import time
import traceback
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core import DATASETS, METRICS, MODELS, Task
from ..log import get_logger
from .store import ResultStore
from .windows import sliding_eval_windows

logger = get_logger("evaluation.runner")


def _select_channels(arr: np.ndarray, idx: List[int]) -> np.ndarray:
    return arr[..., idx] if idx else arr


def run_experiment(
    config: Dict[str, Any],
    results_dir: str = "results",
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> List[str]:
    """执行一份实验配置，返回生成的 run_id 列表。

    on_progress(run_id, status) 可用于 API 层跟踪状态。
    """
    store = ResultStore(results_dir)

    # 1. 数据集
    ds_cfg = config["dataset"]
    logger.info("实验开始: name=%s, 数据集=%s", config.get("name", ""), ds_cfg["name"])
    dataset = DATASETS.get(ds_cfg["name"])(ds_cfg.get("config", {}))
    bundle = dataset.load()
    logger.info(
        "数据集就绪: %s train=%d val=%s test=%d 特征=%d freq=%s",
        bundle.name, len(bundle.train),
        len(bundle.val) if bundle.val is not None else 0,
        len(bundle.test), bundle.num_features, bundle.freq,
    )

    # 2. 任务
    task = Task(**config.get("task", {}))
    target_idx: List[int] = []
    if task.target is not None:
        if task.target not in bundle.feature_names:
            raise ValueError(f"target '{task.target}' not in features {bundle.feature_names}")
        target_idx = [bundle.feature_names.index(task.target)]

    # 3. 评估窗口（全模型共用，保证可比）
    full = bundle.full_series()
    test_start = bundle.test_start_index()
    eval_end = len(full) - task.horizon + 1
    contexts, y_true = sliding_eval_windows(full, test_start, eval_end, task.context_length, task.horizon)
    logger.info(
        "评估窗口: %d 个 (context=%d, horizon=%d, test_start=%d)",
        len(contexts), task.context_length, task.horizon, test_start,
    )
    if len(contexts) == 0:
        raise ValueError(
            f"test 段长度 {len(bundle.test)} 不足以评估 "
            f"context({task.context_length})+horizon({task.horizon})"
        )

    # 4. 指标
    metric_names = config.get("metrics") or METRICS.list()
    metric_specs = [METRICS.get(m) for m in metric_names]

    # 5. 逐模型训练 + 预测 + 指标
    models_cfg = config.get("models") or []
    if not models_cfg:
        raise ValueError("config.models 不能为空")

    run_ids: List[str] = []
    for m_cfg in models_cfg:
        m_name = m_cfg["name"]
        m_conf = m_cfg.get("config", {})
        run_id = store.new_run_id(m_name, bundle.name, task.horizon)
        run_ids.append(run_id)
        logger.info("[%s] 开始: model=%s config=%s", run_id, m_name, m_conf)
        if on_progress:
            on_progress(run_id, "running")

        meta: Dict[str, Any] = {
            "run_id": run_id,
            "experiment": config.get("name", ""),
            "model": m_name,
            "model_config": m_conf,
            "dataset": bundle.name,
            "task": asdict(task),
            "status": "running",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        t0 = time.time()
        # 先落盘 running 状态，训练中的 run 也能在列表/API 中看到
        store.save_run(run_id, {**config, "models": [m_cfg]}, meta, {}, {})
        try:
            model = MODELS.get(m_name)(m_conf, task)
            train_info = model.fit(bundle)
            logger.info("[%s] 训练完成: %s", run_id, train_info)
            y_pred = model.predict(contexts)

            # 通道选择：单变量任务只评估目标通道
            yt = _select_channels(y_true, target_idx)
            yp = _select_channels(y_pred, target_idx)
            y_train_sel = _select_channels(bundle.train, target_idx)

            metrics = {}
            for spec in metric_specs:
                try:
                    metrics[spec.name] = spec.fn(
                        yt, yp, y_train=y_train_sel, seasonal_period=task.seasonal_period
                    )
                except Exception as e:  # 单个指标失败不拖垮整个 run
                    metrics[spec.name] = float("nan")
                    metrics[f"{spec.name}__error"] = str(e)

            meta.update(
                status="completed",
                duration_sec=round(time.time() - t0, 2),
                train_info=train_info,
                env={"python": platform.python_version(), "platform": platform.platform()},
                n_eval_windows=len(contexts),
            )
            arrays = {
                "full_series": full.astype(np.float32),
                "y_true": y_true.astype(np.float32),
                "y_pred": y_pred.astype(np.float32),
                "feature_names": np.array(bundle.feature_names),
                "test_start": np.array([test_start], dtype=np.int64),
                "scaler_mean": (
                    bundle.scaler_mean
                    if bundle.scaler_mean is not None
                    else np.zeros(bundle.num_features, dtype=np.float32)
                ),
                "scaler_std": (
                    bundle.scaler_std
                    if bundle.scaler_std is not None
                    else np.ones(bundle.num_features, dtype=np.float32)
                ),
                "timestamps": (
                    np.asarray(bundle.timestamps).astype(str)  # 强制 unicode dtype，避免 object array
                    if bundle.timestamps is not None
                    else np.array([str(i) for i in range(len(full))])
                ),
            }
            single_model_config = {**config, "models": [m_cfg]}
            store.save_run(run_id, single_model_config, meta, metrics, arrays)
            logger.info(
                "[%s] 完成: status=completed 耗时=%.1fs 指标=%s",
                run_id, time.time() - t0,
                {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
            )
        except Exception as e:
            meta.update(
                status="failed",
                duration_sec=round(time.time() - t0, 2),
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc()[-3000:],
            )
            store.save_run(run_id, {**config, "models": [m_cfg]}, meta, {}, {})
            logger.exception("[%s] 失败: %s", run_id, e)
        if on_progress:
            on_progress(run_id, meta["status"])

    return run_ids

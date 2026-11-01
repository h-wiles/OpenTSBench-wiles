#!/usr/bin/env python3
"""OpenTSBench 命令行入口。

  python cli.py run --config tsbench/configs/etth1_h96.yaml
  python cli.py list [--dataset ETTh1]
  python cli.py delete <run_id>
  python cli.py plugins
  python cli.py serve [--port 8000]
  python cli.py download-data
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from tsbench.log import setup_logging  # noqa: E402
from tsbench.plugins import load_builtin_plugins  # noqa: E402


def cmd_run(args) -> None:
    load_builtin_plugins()
    from tsbench.evaluation.runner import run_experiment

    with open(args.config) as f:
        config = yaml.safe_load(f)

    def on_progress(run_id: str, status: str) -> None:
        print(f"[{status:>10}] {run_id}")

    run_ids = run_experiment(config, results_dir=args.results, on_progress=on_progress)
    print(f"\n完成，共 {len(run_ids)} 个 run：")
    for rid in run_ids:
        print(" -", rid)


def cmd_list(args) -> None:
    from tsbench.evaluation.store import ResultStore

    store = ResultStore(args.results)
    rows = store.list_runs(dataset=args.dataset, model=args.model)
    if not rows:
        print("（暂无实验结果，先跑一个: python cli.py run --config ...）")
        return
    metric_names = sorted({m for r in rows for m in r.get("metrics", {}) if not m.endswith("__error")})
    header = ["run_id", "model", "status"] + metric_names[:6]
    print("\t".join(header))
    for r in rows:
        cells = [r["run_id"][:42], r.get("model", ""), r.get("status", "")]
        cells += [f"{r['metrics'].get(m, float('nan')):.4f}" for m in metric_names[:6]]
        print("\t".join(cells))


def cmd_delete(args) -> None:
    from tsbench.evaluation.store import ResultStore

    if ResultStore(args.results).delete_run(args.run_id):
        print(f"已删除 {args.run_id}")
    else:
        print(f"未找到 {args.run_id}", file=sys.stderr)
        sys.exit(1)


def cmd_plugins(args) -> None:
    load_builtin_plugins()
    from tsbench.core import DATASETS, METRICS, MODELS

    print(json.dumps(
        {"datasets": DATASETS.describe(), "models": MODELS.describe(), "metrics": METRICS.describe()},
        indent=2, ensure_ascii=False,
    ))


def cmd_serve(args) -> None:
    import uvicorn

    from tsbench.api.server import create_app

    app = create_app(results_dir=args.results)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_download_data(args) -> None:
    from scripts.download_data import main as download_main

    download_main()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="tsbench", description="OpenTSBench CLI")
    parser.add_argument("--results", default="results", help="结果目录 (默认 results/)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="按 YAML 配置运行实验")
    p.add_argument("--config", required=True)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("list", help="列出实验结果")
    p.add_argument("--dataset")
    p.add_argument("--model")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("delete", help="删除实验")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_delete)

    p = sub.add_parser("plugins", help="列出已注册插件")
    p.set_defaults(fn=cmd_plugins)

    p = sub.add_parser("serve", help="启动 API 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("download-data", help="下载公开数据集 (ETT)")
    p.set_defaults(fn=cmd_download_data)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

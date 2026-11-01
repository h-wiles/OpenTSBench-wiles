"""下载公开时序数据集（ETT 系列，GitHub 官方仓库直链）。"""
from __future__ import annotations

import os
import urllib.request

ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
DATASETS = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    out_dir = os.path.join(PROJECT_ROOT, "data", "ETT")
    os.makedirs(out_dir, exist_ok=True)
    for name in DATASETS:
        url = f"{ETT_BASE}/{name}"
        dest = os.path.join(out_dir, name)
        if os.path.exists(dest):
            print(f"已存在，跳过: {dest}")
            continue
        print(f"下载 {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("完成。")


if __name__ == "__main__":
    main()

"""PyTorch 模型的通用基类：把"训练循环"做成框架能力。

子类（原生实现或第三方 Adapter）只需实现 build_network()，
返回一个 (B, L, D_in) -> (B, H, D_out) 的 nn.Module，
即可获得：窗口数据集构建、Adam 训练、验证集早停、批量推理。

通用配置项（均可被子类默认或 YAML 覆盖）：
  epochs / batch_size / lr / patience / loss(mse|l1) / seed / device / num_workers
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..core import BaseModel, DatasetBundle
from ..evaluation.windows import sliding_eval_windows, train_windows
from ..log import get_logger

logger = get_logger("models.torch_base")


class TorchForecastModel(BaseModel):
    """所有 PyTorch 预测模型的基类。"""

    DEFAULTS: Dict[str, Any] = {
        "epochs": 10,
        "batch_size": 32,
        "lr": 1e-3,
        "patience": 3,
        "loss": "mse",
        "seed": 42,
        "device": "auto",
        "num_workers": 0,
    }

    def __init__(self, config: Dict[str, Any], task):
        merged = {**self.DEFAULTS, **(config or {})}
        super().__init__(merged, task)
        self.net: nn.Module | None = None

    # ---- 子类实现 ----
    def build_network(self, n_channels_in: int, n_channels_out: int) -> nn.Module:
        """返回 (B, L, D_in) -> (B, H, D_out) 的网络。"""
        raise NotImplementedError

    # ---- 训练 ----
    def _device(self) -> torch.device:
        dev = self.config["device"]
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev)

    def _loss_fn(self) -> nn.Module:
        return nn.L1Loss() if self.config["loss"] == "l1" else nn.MSELoss()

    def fit(self, bundle: DatasetBundle) -> Dict[str, Any]:
        cfg = self.config
        torch.manual_seed(int(cfg["seed"]))
        np.random.seed(int(cfg["seed"]))
        device = self._device()

        ctx, horizon = self.task.context_length, self.task.horizon
        x_train, y_train = train_windows(bundle.train, ctx, horizon)
        if len(x_train) == 0:
            raise ValueError(
                f"训练集长度 {len(bundle.train)} 不足以构成 "
                f"context({ctx})+horizon({horizon}) 的窗口"
            )

        # 验证窗口：context 可回溯到 train 末尾，预测起点落在 val 段内
        x_val = y_val = None
        if bundle.val is not None and len(bundle.val) >= horizon:
            full = bundle.full_series()
            start = len(bundle.train)
            end = start + len(bundle.val) - horizon + 1
            x_val, y_val = sliding_eval_windows(full, start, end, ctx, horizon)

        self.net = self.build_network(bundle.num_features, bundle.num_features).to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg["lr"]))
        loss_fn = self._loss_fn()
        logger.info(
            "训练开始: %s, device=%s, 参数量=%d, 训练窗口=%d, val窗口=%s, epochs=%d, batch=%d, lr=%g",
            type(self.net).__name__, device,
            sum(p.numel() for p in self.net.parameters()), len(x_train),
            len(x_val) if x_val is not None else 0,
            cfg["epochs"], cfg["batch_size"], cfg["lr"],
        )

        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_dl = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["num_workers"]),
            drop_last=True,
        )

        n_params = sum(p.numel() for p in self.net.parameters())
        best_val, best_state, bad_epochs = float("inf"), None, 0
        history = []
        t0 = time.time()
        epochs_run = 0
        for epoch in range(int(cfg["epochs"])):
            epochs_run += 1
            self.net.train()
            train_loss, n_batches = 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            val_loss = float("nan")
            if x_val is not None:
                val_loss = self._eval_loss(x_val, y_val, loss_fn, device)
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                logger.debug(
                    "epoch %d: train_loss=%.5f val_loss=%.5f (best=%.5f, bad=%d)",
                    epoch, train_loss, val_loss, best_val, bad_epochs,
                )
                if val_loss < best_val - 1e-6:
                    best_val, bad_epochs = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(cfg["patience"]):
                        break
            else:
                history.append({"epoch": epoch, "train_loss": train_loss})
                logger.debug("epoch %d: train_loss=%.5f (无验证集)", epoch, train_loss)

        if bad_epochs >= int(cfg["patience"]):
            logger.info("早停触发: epoch=%d (patience=%d)", epochs_run, cfg["patience"])
        if best_state is not None:
            self.net.load_state_dict(best_state)

        return {
            "device": str(device),
            "trainable_params": n_params,
            "epochs_run": epochs_run,
            "best_val_loss": None if best_val == float("inf") else best_val,
            "train_seconds": round(time.time() - t0, 2),
            "history": history,
        }

    @torch.no_grad()
    def _eval_loss(self, x: np.ndarray, y: np.ndarray, loss_fn, device) -> float:
        self.net.eval()
        total, count = 0.0, 0
        for i in range(0, len(x), 256):
            xb = torch.from_numpy(x[i : i + 256]).to(device)
            yb = torch.from_numpy(y[i : i + 256]).to(device)
            total += loss_fn(self.net(xb), yb).item() * len(xb)
            count += len(xb)
        return total / max(count, 1)

    @torch.no_grad()
    def predict(self, context: np.ndarray) -> np.ndarray:
        assert self.net is not None, "model must be fitted before predict"
        device = self._device()
        self.net.eval()
        context = np.ascontiguousarray(context, dtype=np.float32)
        outs = []
        for i in range(0, len(context), 256):
            xb = torch.from_numpy(context[i : i + 256]).to(device)
            outs.append(self.net(xb).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

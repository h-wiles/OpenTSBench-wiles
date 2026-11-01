# OpenTSBench 架构设计

> 目标：一个**高扩展、强解耦**的开源时序预测模型评测平台。
> 任意深度学习时序模型 × 任意开源数据集 × 多种评测指标 × Web 可视化。

## 1. 设计原则

1. **插件化（Registry Pattern）**：数据集、模型、指标都是插件，通过装饰器注册，
   新增任何一类插件都**不需要修改框架核心代码**。
2. **标准化接口（Adapter Pattern）**：第三方开源模型（如 Time-Series-Library 中的
   PatchTST / iTransformer）通过 Adapter 包装接入，**不改动第三方源码**。
3. **配置驱动（Config-driven）**：一次评测 = 一份 YAML 配置，可复现、可分享。
4. **分层解耦**：数据层 / 模型层 / 评测层 / 存储层 / API 层 / 前端层，
   层与层之间只通过标准数据结构通信（`DatasetBundle` / `Task` / `RunResult`）。
5. **进程隔离（可选）**：第三方模型既可以进程内接入（依赖兼容时），
   也可以通过子进程方式在独立 venv 中运行，彻底隔离依赖冲突。

## 2. 总体分层

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React + ECharts)  排行榜 / 曲线对比 / 发起实验 │
├─────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)         /api/runs /api/leaderboard │
├─────────────────────────────────────────────────────────┤
│  Evaluation Layer            Runner · 滑窗评估 · 指标聚合 │
├──────────────┬──────────────────────┬───────────────────┤
│ Dataset 插件  │  Model 插件           │ Metric 插件        │
│ CSV/ETT/M4…  │  Baseline/DLinear/    │ MSE/MAE/sMAPE/    │
│              │  TSLib Adapter/子进程  │ MASE/R² …         │
├──────────────┴──────────────────────┴───────────────────┤
│  Core 抽象层  Registry · BaseDataset · BaseModel · Metric │
├─────────────────────────────────────────────────────────┤
│  Result Store (JSON + NPZ 文件存储， 可换 SQLite)          │
└─────────────────────────────────────────────────────────┘
```

## 3. 目录结构

```
OpenTSBench-wiles/
├── tsbench/                    # 核心 Python 包
│   ├── core/                   # 抽象层：注册中心 + 基类 + 标准数据结构
│   │   ├── registry.py         #   DATASETS / MODELS / METRICS 三个注册表
│   │   ├── types.py            #   DatasetBundle / Task / RunResult
│   │   ├── base_dataset.py     #   数据集抽象: load() -> DatasetBundle
│   │   ├── base_model.py       #   模型抽象: fit() / predict()
│   │   └── base_metric.py      #   指标抽象: fn(y_true, y_pred, ...)
│   ├── datasets/               # 数据集插件
│   │   └── csv_dataset.py      #   通用 CSV (ETT 风格), 支持 ratio/ett 切分
│   ├── models/                 # 模型插件
│   │   ├── baselines.py        #   Naive / SeasonalNaive / MovingAverage
│   │   ├── torch_base.py       #   PyTorch 模型通用训练循环 (早停/验证)
│   │   ├── dlinear.py          #   原生 DLinear (DL 模型接入示例)
│   │   └── tslib_adapter.py    #   Time-Series-Library 第三方适配器
│   ├── metrics/                # 指标插件
│   │   └── forecasting.py      #   MSE/RMSE/MAE/MAPE/sMAPE/MASE/R²
│   ├── evaluation/             # 评测层
│   │   ├── windows.py          #   滑动窗口生成 (context 可回溯到 test 之前)
│   │   ├── runner.py           #   实验编排: 数据→训练→预测→指标→落盘
│   │   └── store.py            #   结果存储与排行榜查询
│   ├── plugins.py              # 一键加载全部内置插件
│   ├── log.py                  # 统一日志 (控制台 INFO + logs/ 文件 DEBUG 按天滚动)
│   ├── api/server.py           # FastAPI 后端
│   └── configs/                # YAML 实验配置示例
├── web/                        # React + Vite + ECharts 前端
├── data/                       # 数据集 (gitignored)
├── third_party/                # 克隆的开源模型仓库 (gitignored)
├── results/                    # 实验结果 (gitignored)
└── scripts/                    # 数据下载等脚本
```

## 4. 核心抽象

### 4.1 数据流标准结构

```python
DatasetBundle        # 数据集加载后的标准产物
├── train/val/test   # np.ndarray (T, D)，已按配置标准化(scaler 只在 train 上 fit)
├── feature_names    # 通道名, 如 ETT 的 HUFL..OT
├── timestamps       # 全序列时间戳 (前端画 x 轴)
└── freq             # "h" / "15min" ...

Task                 # 一次预测任务定义
├── context_length   # 回看窗口 L
├── horizon          # 预测步长 H
├── target           # None=多变量全通道; "OT"=单变量
└── seasonal_period  # MASE 等指标的季节周期
```

### 4.2 模型接口（任意 DL 模型只需实现两个方法）

```python
class BaseModel(ABC):
    def fit(self, bundle: DatasetBundle) -> dict: ...
    def predict(self, context: np.ndarray) -> np.ndarray:
        """(B, L, D) -> (B, H, D)。模型只需管"给历史, 出预测"。"""
```

- **评估口径统一**：框架用滑窗在 test 段上取 `(context, truth)`，
  所有模型面对完全相同的输入，指标完全可比。
- context 允许回溯到 test 起点之前（与 Time-Series-Library 的评估口径一致）。

### 4.3 指标接口

```python
@register_metric("smape", higher_is_better=False)
def smape(y_true, y_pred, *, y_train=None, seasonal_period=1) -> float: ...
```

- `y_true/y_pred`: `(N, H, D)`；指标自身声明方向（越高越好/越低越好），
  前端排行榜据此决定高亮方式。
- 需要训练集统计量的指标（如 MASE）通过 `y_train` 注入。

### 4.4 插件注册

```python
@register_dataset("csv")        # 新数据集
@register_model("dlinear")      # 新模型
@register_metric("smape")       # 新指标
```

新增插件 = 新建一个文件 + 一个装饰器，`cli.py` / API / 前端自动可见。

## 5. 实验配置（YAML 示例）

```yaml
name: etth1_h96
dataset:
  name: csv
  config:
    path: data/ETT/ETTh1.csv
    split: {type: ett}          # ETT 标准 12/4/4 月切分; 也支持 ratio
    scale: true
task: {context_length: 96, horizon: 96, seasonal_period: 24}
models:
  - {name: seasonal_naive, config: {period: 24}}
  - {name: dlinear, config: {epochs: 10, lr: 0.005}}
  - {name: tslib_patchtst, config: {epochs: 10}}   # 第三方模型
metrics: [mse, rmse, mae, mape, smape, mase, r2]
```

## 6. 结果存储

```
results/<run_id>/
├── config.yaml      # 完整实验配置 (可复现)
├── meta.json        # 状态/耗时/训练信息/环境信息
├── metrics.json     # 各指标值
└── data.npz         # 全序列+scaler 参数+预测结果 (float32 压缩)
```

- API 层按需重建任意评估窗口（context/truth/pred），前端单窗口拉取，流量极小。
- `index` 由存储层扫描重建，无中心化 DB 依赖；未来可平滑替换为 SQLite。

## 7. 第三方模型接入（关键扩展点）

| 方式 | 适用 | 实现 |
|---|---|---|
| 进程内 Adapter | 依赖兼容 (多数 PyTorch 模型) | `sys.path` 注入 third_party 仓库，包装其 `nn.Module` |
| 子进程 Adapter | 依赖冲突/独立 venv | 通过 npz/json 文件交换数据（预留扩展点） |

首批接入：`thuml/Time-Series-Library`（PatchTST、iTransformer、DLinear…）。
Adapter 负责把我们统一的 `(B,L,D)->(B,H,D)` 约定映射到 TSLib 的
`forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` 签名，并自动补全其 configs 命名空间。

## 8. API 设计

| 端点 | 说明 |
|---|---|
| `GET /api/plugins` | 已注册的数据集/模型/指标（含指标方向） |
| `GET /api/runs` | 实验列表（可按 dataset/model 过滤） |
| `GET /api/runs/{id}` | 实验详情（配置+指标+训练信息） |
| `GET /api/runs/{id}/window?index=&channel=` | 单窗口 context/truth/pred（反标准化后） |
| `GET /api/leaderboard?dataset=&horizon=` | 排行榜（按指标排序） |
| `POST /api/runs` | 从前端发起实验（后台线程执行） |
| `DELETE /api/runs/{id}` | 删除实验 |

## 9. 前端页面

1. **Leaderboard**：选数据集/预测步长，指标表格 + 每列最优高亮，点击进入详情。
2. **Run Detail**：指标卡片 + 交互式预测曲线（选窗口、选通道，
   历史 context + 真值 vs 预测，ECharts 折线图）。
3. **New Run**：选数据集/模型/参数 → 提交 → 轮询状态 → 自动跳到详情。

## 10. 扩展路线

- 新数据集：继承 `BaseDataset`（如 M4、GluonTS 适配器）。
- 新模型：进程内 20 行 Adapter，或子进程适配器。
- 新指标：一个函数 + 装饰器。
- 存储层换 SQLite/Postgres：仅改 `store.py`，接口不变。

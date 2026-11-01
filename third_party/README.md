# third_party — 第三方开源项目目录

本目录存放框架依赖的外部开源仓库，**内容不纳入本仓库版本管理**（见 `.gitignore`），
clone 到对应位置即可，框架会自动发现并使用，不修改其任何源码。

| 目录 | 来源 | 用途 | 获取方式 |
|---|---|---|---|
| `Time-Series-Library/` | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | TSLib Adapter 接入的第三方模型库（iTransformer / PatchTST / DLinear 等） | `git clone https://github.com/thuml/Time-Series-Library.git Time-Series-Library` |
| `ETDataset/` | [thuml/ETDataset](https://github.com/thuml/ETDataset) | ETT 数据集官方仓库（原始数据与说明） | `git clone https://github.com/thuml/ETDataset.git ETDataset` |

说明：

- ETT 四个 CSV 已随本仓库提交至 [data/ETT/](../data/ETT/)，`ETDataset` 仅作参考，可选。
- `Time-Series-Library` 为运行 TSLib 系列模型（`tslib_*`）所必需，缺失时相关插件报错。

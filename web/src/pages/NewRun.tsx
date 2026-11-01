// 发起实验：选数据集/任务参数/模型(带参数)/指标 → 生成 YAML 等价配置提交后端
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { DatasetFile, Plugins } from "../types";

const DEFAULT_MODEL_CONFIG: Record<string, string> = {
  naive: "{}",
  seasonal_naive: '{ "period": 24 }',
  moving_average: '{ "window": 24 }',
  dlinear: '{ "epochs": 10, "lr": 0.005 }',
  tslib_dlinear: '{ "epochs": 10, "lr": 0.005 }',
  tslib_patchtst: '{ "epochs": 10, "lr": 0.001, "batch_size": 64 }',
  tslib_itransformer: '{ "epochs": 10, "lr": 0.001, "batch_size": 64 }',
};

export default function NewRun() {
  const navigate = useNavigate();
  const [plugins, setPlugins] = useState<Plugins | null>(null);
  const [datasets, setDatasets] = useState<DatasetFile[]>([]);
  const [error, setError] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);

  // 表单状态
  const [dsPath, setDsPath] = useState<string>("");
  const [splitType, setSplitType] = useState<"ett" | "ratio">("ett");
  const [scale, setScale] = useState(true);
  const [contextLength, setContextLength] = useState(96);
  const [horizon, setHorizon] = useState(96);
  const [seasonalPeriod, setSeasonalPeriod] = useState(24);
  const [target, setTarget] = useState<string>("");
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [modelCfgs, setModelCfgs] = useState<Record<string, string>>({});
  const [metricSel, setMetricSel] = useState<Record<string, boolean>>({});

  useEffect(() => {
    api
      .plugins()
      .then((p) => {
        setPlugins(p);
        setMetricSel(Object.fromEntries(p.metrics.map((m) => [m.name, true])));
      })
      .catch((e) => setError(String(e)));
    api
      .datasets()
      .then((ds) => {
        setDatasets(ds);
        if (ds.length > 0) setDsPath(ds[0].path);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const toggleModel = (name: string) => {
    setSelected((s) => {
      const next = { ...s, [name]: !s[name] };
      if (next[name] && !modelCfgs[name]) {
        setModelCfgs((c) => ({ ...c, [name]: DEFAULT_MODEL_CONFIG[name] ?? "{}" }));
      }
      return next;
    });
  };

  const chosenModels = useMemo(
    () => (plugins?.models ?? []).filter((m) => selected[m.name]),
    [plugins, selected]
  );

  const submit = async () => {
    setError("");
    if (!dsPath) return setError("请选择数据集");
    if (chosenModels.length === 0) return setError("请至少选择一个模型");

    let models: { name: string; config: Record<string, unknown> }[];
    try {
      models = chosenModels.map((m) => ({
        name: m.name,
        config: JSON.parse(modelCfgs[m.name] || "{}") as Record<string, unknown>,
      }));
    } catch (e) {
      return setError(`模型参数 JSON 解析失败: ${e}`);
    }

    const dsName = dsPath.split("/").pop()?.replace(".csv", "") ?? "dataset";
    const config = {
      name: `web_${dsName}_h${horizon}`,
      dataset: {
        name: "csv",
        config: { path: dsPath, split: { type: splitType }, scale },
      },
      task: {
        context_length: contextLength,
        horizon,
        seasonal_period: seasonalPeriod,
        ...(target.trim() ? { target: target.trim() } : {}),
      },
      models,
      metrics: Object.keys(metricSel).filter((k) => metricSel[k]),
    };

    setSubmitting(true);
    try {
      await api.createRun(config);
      navigate("/leaderboard");
    } catch (e) {
      setError(String(e));
      setSubmitting(false);
    }
  };

  const isTorch = (name: string) =>
    !["naive", "seasonal_naive", "moving_average"].includes(name);

  return (
    <>
      <h1>发起实验</h1>
      <p className="subtitle">
        选择数据集、任务与模型，后端将异步执行；完成后自动出现在排行榜。
      </p>

      {error && <div className="error-box">{error}</div>}

      <div className="card">
        <h2>数据集</h2>
        <div className="filter-row">
          <div className="field">
            <label>CSV 数据集（data/ 目录）</label>
            <select value={dsPath} onChange={(e) => setDsPath(e.target.value)}>
              {datasets.length === 0 && <option value="">（请先把 CSV 放入 data/）</option>}
              {datasets.map((d) => (
                <option key={d.path} value={d.path}>
                  {d.name} — {d.path}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>切分方式</label>
            <select value={splitType} onChange={(e) => setSplitType(e.target.value as "ett" | "ratio")}>
              <option value="ett">ETT 标准 (12/4/4 月)</option>
              <option value="ratio">按比例 (7/1/2)</option>
            </select>
          </div>
          <div className="field">
            <label>标准化</label>
            <input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} />
          </div>
        </div>
      </div>

      <div className="card">
        <h2>任务</h2>
        <div className="filter-row">
          <div className="field">
            <label>回看窗口 context</label>
            <input
              type="number"
              min={8}
              value={contextLength}
              onChange={(e) => setContextLength(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>预测步长 horizon</label>
            <input
              type="number"
              min={1}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>季节周期（MASE 用）</label>
            <input
              type="number"
              min={1}
              value={seasonalPeriod}
              onChange={(e) => setSeasonalPeriod(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>目标通道（留空 = 全通道）</label>
            <input
              type="text"
              placeholder="例如 OT"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="card">
        <h2>模型（勾选后可编辑参数 JSON）</h2>
        {(plugins?.models ?? []).map((m) => (
          <div className="model-pick" key={m.name}>
            <div className="head">
              <input
                type="checkbox"
                checked={!!selected[m.name]}
                onChange={() => toggleModel(m.name)}
              />
              <strong>{m.name}</strong>
              {m.family && <span className="hint">{m.family}</span>}
            </div>
            {m.description && <div className="desc">{m.description}</div>}
            {selected[m.name] && (
              <textarea
                rows={isTorch(m.name) ? 2 : 1}
                value={modelCfgs[m.name] ?? "{}"}
                onChange={(e) =>
                  setModelCfgs((c) => ({ ...c, [m.name]: e.target.value }))
                }
              />
            )}
          </div>
        ))}
      </div>

      <div className="card">
        <h2>指标</h2>
        <div className="filter-row">
          {(plugins?.metrics ?? []).map((m) => (
            <label key={m.name} style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={!!metricSel[m.name]}
                onChange={() => setMetricSel((s) => ({ ...s, [m.name]: !s[m.name] }))}
              />
              {m.name}
              <span className="hint">{m.higher_is_better ? "↑" : "↓"}</span>
            </label>
          ))}
        </div>
      </div>

      <button className="btn" onClick={submit} disabled={submitting}>
        {submitting ? "提交中…" : "提交实验"}
      </button>
    </>
  );
}

// 实验详情：指标卡片 + 交互式预测曲线（选窗口 / 选通道）
import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import ForecastChart from "../components/ForecastChart";
import StatusBadge from "../components/StatusBadge";
import type { RunDetail, WindowData } from "../types";

function fmt(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return Math.abs(v) >= 1000 ? v.toFixed(1) : v.toFixed(4);
}

export default function RunDetailPage() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [win, setWin] = useState<WindowData | null>(null);
  const [index, setIndex] = useState(0);
  const [channel, setChannel] = useState<string>("");
  const [error, setError] = useState<string>("");

  // 加载详情；运行中则轮询
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .run(runId)
        .then((r) => {
          if (!cancelled) setRun(r);
        })
        .catch((e) => !cancelled && setError(String(e)));
    load();
    const t = setInterval(() => {
      if (!cancelled && run?.status === "running") load();
    }, 2000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, run?.status]);

  const loadWindow = useCallback(
    (idx: number, ch?: string) => {
      api
        .window(runId, idx, ch)
        .then((w) => {
          setWin(w);
          setIndex(w.index);
          setChannel(w.channel);
        })
        .catch((e) => setError(String(e)));
    },
    [runId]
  );

  useEffect(() => {
    if (run?.status === "completed" && !win) loadWindow(0);
  }, [run?.status, win, loadWindow]);

  if (error) return <div className="error-box">{error}</div>;
  if (!run) return <p className="hint">加载中…</p>;

  const metricEntries = Object.entries(run.metrics ?? {}).filter(
    ([k]) => !k.endsWith("__error")
  );

  return (
    <>
      <h1>
        {run.model} <span className="hint">on</span> {run.dataset}
      </h1>
      <p className="subtitle mono">{run.run_id}</p>

      <div className="card">
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "center" }}>
          <StatusBadge status={run.status} />
          <span className="hint">
            context {run.task.context_length} → horizon {run.task.horizon}
            {run.task.target ? ` · 目标通道 ${run.task.target}` : " · 全通道"}
          </span>
          <span className="hint">{run.created_at?.replace("T", " ")}</span>
          {run.duration_sec != null && <span className="hint">耗时 {run.duration_sec}s</span>}
          {run.train_info?.trainable_params != null && (
            <span className="hint">参数量 {run.train_info.trainable_params.toLocaleString()}</span>
          )}
          {run.train_info?.best_val_loss != null && (
            <span className="hint">best val {fmt(run.train_info.best_val_loss)}</span>
          )}
          <span style={{ marginLeft: "auto" }}>
            <Link to="/leaderboard">← 返回排行榜</Link>
          </span>
        </div>
        {run.error && <div className="error-box" style={{ marginTop: 12 }}>{run.error}</div>}
      </div>

      {metricEntries.length > 0 && (
        <div className="card">
          <h2>指标</h2>
          <div className="metric-grid">
            {metricEntries.map(([name, value]) => (
              <div className="metric-tile" key={name}>
                <div className="m-name">{name}</div>
                <div className="m-value">{fmt(value)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {run.status === "completed" && win && (
        <div className="card">
          <h2>预测曲线（测试集滑窗）</h2>
          <div className="filter-row">
            <div className="field">
              <label>通道</label>
              <select value={channel} onChange={(e) => loadWindow(index, e.target.value)}>
                {win.channels.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div className="field" style={{ flex: 1, minWidth: 220 }}>
              <label>
                窗口 {index} / {win.n_windows - 1}
              </label>
              <input
                type="range"
                min={0}
                max={win.n_windows - 1}
                value={index}
                onChange={(e) => loadWindow(Number(e.target.value), channel)}
                style={{ width: "100%" }}
              />
            </div>
            <div className="field">
              <label>跳转</label>
              <input
                type="number"
                min={0}
                max={win.n_windows - 1}
                value={index}
                onChange={(e) => loadWindow(Number(e.target.value), channel)}
                style={{ minWidth: 90 }}
              />
            </div>
          </div>
          <ForecastChart data={win} />
          <p className="hint">
            灰线为模型可见的历史 context，蓝线（实线）为真值，橙线（虚线）为模型预测；数值已反标准化回原始量纲。
          </p>
        </div>
      )}
    </>
  );
}

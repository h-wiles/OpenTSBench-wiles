// 排行榜：同一 (数据集, 预测步长) 口径下所有完成的 run，按指标列排序，每列最优高亮
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import StatusBadge from "../components/StatusBadge";
import type { Plugins, RunSummary } from "../types";

function fmt(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(1);
  return v.toFixed(4);
}

export default function Leaderboard() {
  const [plugins, setPlugins] = useState<Plugins | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [horizon, setHorizon] = useState<number | "">("");
  const [sortBy, setSortBy] = useState<string>("mse");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    api.plugins().then(setPlugins).catch((e) => setError(String(e)));
  }, []);

  // 有运行中的实验时轮询
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .runs()
        .then((rows) => {
          if (cancelled) return;
          setRuns(rows);
          if (!dataset && rows.length > 0) {
            const completed = rows.find((r) => r.status === "completed");
            if (completed) setDataset(completed.dataset);
          }
        })
        .catch((e) => setError(String(e)));
    load();
    const t = setInterval(() => {
      if (!cancelled && runs.some((r) => r.status === "running")) load();
    }, 2000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs.some((r) => r.status === "running")]);

  const datasets = useMemo(
    () => [...new Set(runs.map((r) => r.dataset))].sort(),
    [runs]
  );
  const horizons = useMemo(
    () =>
      [...new Set(runs.filter((r) => r.dataset === dataset).map((r) => r.task.horizon))].sort(
        (a, b) => a - b
      ),
    [runs, dataset]
  );

  const rows = useMemo(() => {
    const filtered = runs.filter(
      (r) =>
        r.status !== "failed" &&
        r.dataset === dataset &&
        (horizon === "" || r.task.horizon === horizon)
    );
    const meta = plugins?.metrics.find((m) => m.name === sortBy);
    const asc = !(meta?.higher_is_better ?? false);
    return [...filtered].sort((a, b) => {
      const va = a.metrics[sortBy] ?? Number.POSITIVE_INFINITY;
      const vb = b.metrics[sortBy] ?? Number.POSITIVE_INFINITY;
      return asc ? (va as number) - (vb as number) : (vb as number) - (va as number);
    });
  }, [runs, dataset, horizon, sortBy, plugins]);

  // 本表出现过的指标列（并集，保持注册表顺序）
  const metricCols = useMemo(() => {
    const present = new Set(rows.flatMap((r) => Object.keys(r.metrics)));
    const ordered = (plugins?.metrics ?? [])
      .map((m) => m.name)
      .filter((n) => present.has(n));
    const extra = [...present].filter((n) => !ordered.includes(n)).sort();
    return [...ordered, ...extra];
  }, [rows, plugins]);

  // 每列最优值（用于高亮）
  const bestOf = useMemo(() => {
    const out: Record<string, number> = {};
    for (const col of metricCols) {
      const meta = plugins?.metrics.find((m) => m.name === col);
      const higher = meta?.higher_is_better ?? false;
      let best: number | null = null;
      for (const r of rows) {
        const v = r.metrics[col];
        if (v == null || Number.isNaN(v)) continue;
        if (best == null || (higher ? v > best : v < best)) best = v;
      }
      if (best != null) out[col] = best;
    }
    return out;
  }, [rows, metricCols, plugins]);

  const handleDelete = async (runId: string) => {
    if (!window.confirm(`删除实验 ${runId}？`)) return;
    await api.deleteRun(runId);
    setRuns((rs) => rs.filter((r) => r.run_id !== runId));
  };

  return (
    <>
      <h1>模型排行榜</h1>
      <p className="subtitle">同一数据集与预测步长下，所有模型的指标对比；点击列名切换排序依据。</p>

      {error && <div className="error-box">{error}</div>}

      <div className="card">
        <div className="filter-row">
          <div className="field">
            <label>数据集</label>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {datasets.length === 0 && <option value="">（暂无数据）</option>}
              {datasets.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>预测步长</label>
            <select
              value={horizon}
              onChange={(e) => setHorizon(e.target.value === "" ? "" : Number(e.target.value))}
            >
              <option value="">全部</option>
              {horizons.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>　</label>
            <Link to="/new">
              <button className="btn">+ 发起实验</button>
            </Link>
          </div>
        </div>

        {rows.length === 0 ? (
          <p className="hint">
            暂无实验结果。可先用 CLI 跑一个：
            <code className="mono"> python cli.py run --config tsbench/configs/etth1_h96.yaml </code>
            或点击右上角"发起实验"。
          </p>
        ) : (
          <table className="board">
            <thead>
              <tr>
                <th>#</th>
                <th>模型</th>
                <th>状态</th>
                {metricCols.map((col) => {
                  const meta = plugins?.metrics.find((m) => m.name === col);
                  const arrow = meta?.higher_is_better ? "↑" : "↓";
                  return (
                    <th
                      key={col}
                      style={{ cursor: "pointer" }}
                      title={`${col}（${meta?.higher_is_better ? "越高越好" : "越低越好"}），点击排序`}
                      onClick={() => setSortBy(col)}
                    >
                      {col} {arrow}
                      {sortBy === col ? " ·" : ""}
                    </th>
                  );
                })}
                <th>耗时</th>
                <th>时间</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.run_id}>
                  <td>{i + 1}</td>
                  <td>
                    <Link to={`/runs/${encodeURIComponent(r.run_id)}`}>{r.model}</Link>
                  </td>
                  <td>
                    <StatusBadge status={r.status} />
                  </td>
                  {metricCols.map((col) => {
                    const v = r.metrics[col];
                    const isBest = v != null && bestOf[col] === v;
                    return (
                      <td key={col} className={`num${v == null ? " na" : ""}${isBest ? " best" : ""}`}>
                        {fmt(v)}
                      </td>
                    );
                  })}
                  <td>{r.duration_sec != null ? `${r.duration_sec}s` : "—"}</td>
                  <td className="hint">{r.created_at?.replace("T", " ").slice(5, 16)}</td>
                  <td>
                    <button className="btn danger" onClick={() => handleDelete(r.run_id)}>
                      删
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}

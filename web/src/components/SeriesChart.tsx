// 时间区间对比图：真值 vs 预测（用户选定起止时间与前瞻步数）
// 交互：dataZoom 缩放（默认聚焦最近 7 天）、LTTB 降采样、十字准线 tooltip
import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import { api } from "../api";
import { currentPalette } from "../colors";
import type { SeriesData } from "../types";

const DAY_MS = 24 * 3600 * 1000;

// "2017-10-28 03:00:00" <-> datetime-local 的 "2017-10-28T03:00"
const tsToLocal = (ts: string) => ts.slice(0, 16).replace(" ", "T");
const localToTs = (v: string) => v.replace("T", " ") + ":00";

export default function SeriesChart({ runId, horizon }: { runId: string; horizon: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const [data, setData] = useState<SeriesData | null>(null);
  const [channel, setChannel] = useState("");
  const [lead, setLead] = useState(0);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [valid, setValid] = useState<[string, string] | null>(null);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(
    () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const fn = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", fn);
    return () => mq.removeEventListener("change", fn);
  }, []);

  // 首次加载：取默认（全有效区间）并初始化选择器
  useEffect(() => {
    let cancelled = false;
    setData(null);
    api
      .series(runId, { lead })
      .then((d) => {
        if (cancelled) return;
        setValid(d.valid_range);
        setChannel(d.channel);
        setStart(tsToLocal(d.valid_range[0]));
        setEnd(tsToLocal(d.valid_range[1]));
        setData(d);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [runId, lead]);

  // 用户改动区间/通道后重新拉取（跳过首次，避免与上面重复请求）
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    if (!start || !end) return;
    api
      .series(runId, { start: localToTs(start), end: localToTs(end), channel, lead })
      .then(setData)
      .catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start, end, channel, lead]);

  // ECharts 渲染
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !data) return;
    const p = currentPalette();

    // dataZoom 初始聚焦最后 7 天（本地时区计算；数据不足则全区间）
    const n = data.timestamps.length;
    const d = new Date(Date.parse(data.timestamps[n - 1]) - 7 * DAY_MS);
    const pad = (x: number) => String(x).padStart(2, "0");
    const weekStart =
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
      ` ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    let zoomStartIdx = data.timestamps.findIndex((t) => t.slice(0, 16) >= weekStart);
    if (zoomStartIdx < 0) zoomStartIdx = 0;

    chart.setOption(
      {
        backgroundColor: "transparent",
        animation: false,
        legend: { top: 0, textStyle: { color: p.secondary, fontSize: 12 }, itemWidth: 18 },
        grid: { left: 56, right: 20, top: 34, bottom: 64 },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "cross", label: { backgroundColor: p.axis } },
          valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "-"),
        },
        dataZoom: [
          { type: "inside", startValue: zoomStartIdx, endValue: n - 1 },
          {
            type: "slider",
            startValue: zoomStartIdx,
            endValue: n - 1,
            height: 20,
            bottom: 8,
            borderColor: p.grid,
          },
        ],
        xAxis: {
          type: "category",
          data: data.timestamps,
          axisLabel: { color: p.muted, fontSize: 11, hideOverlap: true },
          axisLine: { lineStyle: { color: p.axis } },
          axisTick: { show: false },
        },
        yAxis: {
          type: "value",
          scale: true,
          axisLabel: { color: p.muted, fontSize: 11 },
          splitLine: { lineStyle: { color: p.grid, width: 1 } },
        },
        series: [
          {
            name: "真值",
            type: "line",
            data: data.truth,
            showSymbol: false,
            sampling: "lttb",
            lineStyle: { width: 2, color: p.series1 },
            itemStyle: { color: p.series1 },
          },
          {
            name: "预测",
            type: "line",
            data: data.prediction,
            showSymbol: false,
            sampling: "lttb",
            lineStyle: { width: 2, color: p.series2, type: "dashed" },
            itemStyle: { color: p.series2 },
          },
        ],
      },
      { notMerge: true }
    );
  }, [data, dark]);

  const leadOptions = useMemo(
    () => [0, 23, 47, 95].filter((v) => v < horizon).map((v) => ({ v, label: v === 0 ? "1 步（滚动）" : `${v + 1} 步` })),
    [horizon]
  );

  if (error) return <div className="error-box">{error}</div>;

  return (
    <>
      <div className="filter-row">
        <div className="field">
          <label>开始时间</label>
          <input
            type="datetime-local"
            value={start}
            min={valid ? tsToLocal(valid[0]) : undefined}
            max={valid ? tsToLocal(valid[1]) : undefined}
            onChange={(e) => setStart(e.target.value)}
          />
        </div>
        <div className="field">
          <label>结束时间</label>
          <input
            type="datetime-local"
            value={end}
            min={valid ? tsToLocal(valid[0]) : undefined}
            max={valid ? tsToLocal(valid[1]) : undefined}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
        <div className="field">
          <label>通道</label>
          <select value={channel} onChange={(e) => setChannel(e.target.value)}>
            {(data?.channels ?? []).map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>前瞻步数</label>
          <select value={lead} onChange={(e) => setLead(Number(e.target.value))}>
            {leadOptions.map((o) => (
              <option key={o.v} value={o.v}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {valid && (
        <p className="hint">
          可用区间 {valid[0]} ~ {valid[1]}（测试段）；"前瞻 N 步"表示每个时刻显示模型提前 N
          步做出的预测，N 越大长程误差越明显。图内可滚轮缩放、拖动下方滑块平移。
        </p>
      )}
      <div ref={ref} style={{ width: "100%", height: 440 }} />
    </>
  );
}

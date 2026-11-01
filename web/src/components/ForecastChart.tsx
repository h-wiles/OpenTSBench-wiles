// 预测对比图：历史 context（中性灰）+ 真值（分类色1 蓝）+ 预测（分类色2 橙, 虚线作次级编码）
// 规范要点：2px 细线、十字准线 tooltip、常驻图例、隐形网格、文本用墨色、暗色模式独立色板
import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { currentPalette } from "../colors";
import type { WindowData } from "../types";

function useDarkMode(): boolean {
  const [dark, setDark] = useState(
    () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark;
}

export default function ForecastChart({ data }: { data: WindowData }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const dark = useDarkMode();

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const p = currentPalette();
    const ctxLen = data.context.length;

    // context 的最后一个点同时作为真值/预测序列的起点，保证折线连续
    const anchor = data.context[ctxLen - 1];
    const contextSeries: (number | null)[] = [...data.context, ...data.truth.map(() => null)];
    const truthSeries: (number | null)[] = [
      ...data.context.slice(0, ctxLen - 1).map(() => null),
      anchor,
      ...data.truth,
    ];
    const predSeries: (number | null)[] = [
      ...data.context.slice(0, ctxLen - 1).map(() => null),
      anchor,
      ...data.prediction,
    ];

    chart.setOption(
      {
        backgroundColor: "transparent",
        animation: false,
        legend: {
          top: 0,
          data: ["历史 (context)", "真值", "预测"], // 排除 markLine 辅助 series
          textStyle: { color: p.secondary, fontSize: 12 },
          itemWidth: 18,
        },
        grid: { left: 56, right: 20, top: 36, bottom: 30 },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "cross", label: { backgroundColor: p.axis } },
          valueFormatter: (v: unknown) =>
            typeof v === "number" ? v.toFixed(3) : "-",
        },
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
            name: "历史 (context)",
            type: "line",
            data: contextSeries,
            showSymbol: false,
            lineStyle: { width: 2, color: p.muted },
            itemStyle: { color: p.muted },
            emphasis: { disabled: true },
          },
          {
            name: "真值",
            type: "line",
            data: truthSeries,
            showSymbol: false,
            lineStyle: { width: 2, color: p.series1 },
            itemStyle: { color: p.series1 },
          },
          {
            name: "预测",
            type: "line",
            data: predSeries,
            showSymbol: false,
            lineStyle: { width: 2, color: p.series2, type: "dashed" },
            itemStyle: { color: p.series2 },
          },
          {
            // 预测起点分隔线
            name: "forecast-origin",
            type: "line",
            data: [],
            markLine: {
              symbol: "none",
              silent: true,
              label: {
                formatter: "预测起点",
                color: p.secondary,
                fontSize: 11,
                position: "insideEndTop",
              },
              lineStyle: { color: p.axis, type: "dashed", width: 1 },
              data: [{ xAxis: data.timestamps[ctxLen - 1] }],
            },
          },
        ],
      },
      { notMerge: true }
    );
  }, [data, dark]);

  return <div ref={ref} style={{ width: "100%", height: 420 }} />;
}

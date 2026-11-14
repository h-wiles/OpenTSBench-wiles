// 指标说明：公式与解释来自后端指标插件元数据，新增指标可自动出现在本页。
import { useEffect, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { api } from "../api";
import type { PluginMeta } from "../types";

function Formula({ source }: { source: string }) {
  // 公式来自项目内置插件，不接收用户输入。
  const html = katex.renderToString(source, {
    displayMode: true,
    throwOnError: false,
  });
  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />;
}

const NOTATION = [
  ["N", "测试窗口数量"],
  ["H", "预测步长 horizon"],
  ["D", "参与评测的通道数"],
  ["T", "训练集时间长度"],
  ["s", "季节周期 seasonal_period"],
  ["y / ŷ", "真实值 / 模型预测值"],
  ["x", "MASE 分母使用的训练集序列"],
];

export default function MetricsGuide() {
  const [metrics, setMetrics] = useState<PluginMeta[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .plugins()
      .then((p) => {
        if (!cancelled) setMetrics(p.metrics);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <h1>指标说明</h1>
      <p className="subtitle">
        所有模型使用相同的测试窗口和通道。以下公式与当前 Python 指标实现一一对应。
      </p>

      <section className="card">
        <h2>统一计算口径</h2>
        <p className="metric-guide-text">
          测试集按步长 1 滑动取样。每个窗口用过去 L=context_length 个点预测未来
          H=horizon 个点；第一个测试窗口的历史可以回溯到验证集末尾。除特别说明外，
          指标直接对全部 N×H×D 个误差项汇总，不是先逐通道计算再平均。
        </p>
        <Formula source={String.raw`\mathrm{context}_t=\mathrm{full}_{t-L:t},\qquad y_t=\mathrm{full}_{t:t+H},\qquad N=T_{\mathrm{test}}-H+1`} />
        <div className="notation-grid">
          {NOTATION.map(([symbol, meaning]) => (
            <div className="notation-item" key={symbol}>
              <strong>{symbol}</strong>
              <span>{meaning}</span>
            </div>
          ))}
        </div>
        <p className="metric-guide-note">
          若实验启用标准化，公式中的 y、ŷ 和 x 均指标准化后的数值；预测曲线为了便于阅读，
          会反标准化回原始量纲。以 ETTh1、H=96、全 7 通道为例，N=2785，共计算
          2785×96×7=1,871,520 个误差项。
        </p>
      </section>

      {error && <div className="error-box">{error}</div>}
      {metrics === null && !error && <p className="hint">指标说明加载中…</p>}

      <div className="metric-guide-grid">
        {(metrics ?? []).map((metric) => (
          <section className="card metric-doc" id={metric.name} key={metric.name}>
            <div className="metric-doc-head">
              <div>
                <h2>{metric.title ?? metric.name}</h2>
                <span className="metric-code">{metric.name}</span>
              </div>
              <span className="metric-direction">
                {metric.higher_is_better ? "↑ 越高越好" : "↓ 越低越好"}
              </span>
            </div>
            {metric.description && <p className="metric-guide-text">{metric.description}</p>}
            {metric.formula ? (
              <Formula source={metric.formula} />
            ) : (
              <p className="hint">该指标尚未提供公式说明。</p>
            )}
            {metric.calculation && (
              <div className="metric-calculation">
                <strong>怎么计算</strong>
                <p>{metric.calculation}</p>
              </div>
            )}
            {metric.note && <p className="metric-guide-note">{metric.note}</p>}
          </section>
        ))}
      </div>
    </>
  );
}

// API 客户端：所有与 FastAPI 后端的通信都收口在这里
import type {
  DatasetFile,
  Plugins,
  RunDetail,
  RunSummary,
  SeriesData,
  WindowData,
} from "./types";

const BASE = "/api";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 300)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  plugins: () => http<Plugins>("/plugins"),
  datasets: () => http<DatasetFile[]>("/datasets"),
  runs: (dataset?: string) =>
    http<RunSummary[]>(`/runs${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  leaderboard: (dataset: string, horizon?: number) =>
    http<RunSummary[]>(
      `/leaderboard?dataset=${encodeURIComponent(dataset)}` +
        (horizon != null ? `&horizon=${horizon}` : "")
    ),
  run: (id: string) => http<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  window: (id: string, index: number, channel?: string) =>
    http<WindowData>(
      `/runs/${encodeURIComponent(id)}/window?index=${index}` +
        (channel ? `&channel=${encodeURIComponent(channel)}` : "")
    ),
  series: (
    id: string,
    params: { start?: string; end?: string; channel?: string; lead?: number } = {}
  ) =>
    http<SeriesData>(
      `/runs/${encodeURIComponent(id)}/series?` +
        new URLSearchParams(
          Object.entries({
            start: params.start ?? "",
            end: params.end ?? "",
            channel: params.channel ?? "",
            lead: params.lead != null ? String(params.lead) : "",
          }).filter(([, v]) => v !== "")
        ).toString()
    ),
  createRun: (config: unknown) =>
    http<{ accepted: boolean }>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }),
  deleteRun: (id: string) =>
    http<{ deleted: string }>(`/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
};

// 与后端 API 对应的类型定义

export interface PluginMeta {
  name: string;
  description?: string;
  family?: string;
  framework?: string;
  higher_is_better?: boolean;
}

export interface Plugins {
  datasets: PluginMeta[];
  models: PluginMeta[];
  metrics: PluginMeta[];
}

export interface DatasetFile {
  name: string;
  path: string;
}

export interface Task {
  context_length: number;
  horizon: number;
  target?: string | null;
  seasonal_period: number;
}

export interface RunSummary {
  run_id: string;
  experiment?: string;
  model: string;
  dataset: string;
  status: "running" | "completed" | "failed" | string;
  created_at?: string;
  duration_sec?: number;
  n_eval_windows?: number;
  task: Task;
  metrics: Record<string, number | null>;
}

export interface RunDetail extends RunSummary {
  model_config?: Record<string, unknown>;
  train_info?: {
    trainable_params?: number;
    epochs_run?: number;
    best_val_loss?: number | null;
    train_seconds?: number;
    device?: string;
    history?: { epoch: number; train_loss: number; val_loss?: number }[];
  };
  error?: string;
  config?: Record<string, unknown>;
}

export interface WindowData {
  index: number;
  n_windows: number;
  channel: string;
  channels: string[];
  context: number[];
  truth: number[];
  prediction: number[];
  timestamps: string[];
}

export interface SeriesData {
  channel: string;
  channels: string[];
  lead: number;
  horizon: number;
  n_points: number;
  valid_range: [string, string];
  timestamps: string[];
  truth: number[];
  prediction: number[];
}

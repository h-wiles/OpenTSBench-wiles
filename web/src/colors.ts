// 图表配色：取自项目 dataviz 规范的已验证参考调色板（CVD 安全，固定顺序）。
// 与 styles.css 中的 CSS 变量保持一致；ECharts 在 JS 侧需要显式色值。
// - 分类色（固定顺序，不循环）：slot1 蓝 / slot2 橙 / slot3 青
// - 状态色仅用于状态（成功/警告/失败），绝不充当序列色

export interface ChartPalette {
  surface: string;
  ink: string;
  secondary: string;
  muted: string;
  grid: string;
  axis: string;
  series1: string; // 真值
  series2: string; // 预测
  series3: string;
}

export const LIGHT: ChartPalette = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  series1: "#2a78d6",
  series2: "#eb6834",
  series3: "#1baf7a",
};

export const DARK: ChartPalette = {
  surface: "#1a1a19",
  ink: "#f0efec",
  secondary: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
  series1: "#3987e5",
  series2: "#d95926",
  series3: "#199e70",
};

export function currentPalette(): ChartPalette {
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return DARK;
  }
  return LIGHT;
}

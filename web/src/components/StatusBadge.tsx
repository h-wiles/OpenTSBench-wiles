// 状态徽标：图标 + 文字 + 状态色（信息不只靠颜色传达）
const LABELS: Record<string, string> = {
  completed: "完成",
  running: "运行中",
  failed: "失败",
};

export default function StatusBadge({ status }: { status: string }) {
  const cls = ["completed", "running", "failed"].includes(status) ? status : "";
  return (
    <span className={`badge ${cls}`}>
      <span className="dot" />
      {LABELS[status] ?? status}
    </span>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  color?: "blue" | "green" | "red" | "yellow";
  icon?: React.ReactNode;
}

const colorMap = {
  blue: "border-l-blue",
  green: "border-l-green",
  red: "border-l-red",
  yellow: "border-l-warning",
};

export default function MetricCard({
  label,
  value,
  subtext,
  color = "blue",
  icon,
}: MetricCardProps) {
  return (
    <div
      className={`rounded-lg border border-border bg-bg-surface p-3 border-l-4 ${colorMap[color]}`}
    >
      <div className="flex items-center gap-2 mb-1">
        {icon && <span className="text-text-secondary">{icon}</span>}
        <span className="text-xs text-text-secondary">{label}</span>
      </div>
      <div className="text-lg font-bold text-text-primary">{value}</div>
      {subtext && <div className="text-xs text-text-secondary mt-0.5">{subtext}</div>}
    </div>
  );
}

import { clsx } from "clsx";

const CONFIG: Record<string, { label: string; cls: string }> = {
  LOW:      { label: "LOW",      cls: "bg-green-50 text-green-700 border-green-200" },
  MEDIUM:   { label: "MEDIUM",   cls: "bg-amber-50 text-amber-800 border-amber-200" },
  HIGH:     { label: "HIGH",     cls: "bg-orange-50 text-orange-700 border-orange-200" },
  CRITICAL: { label: "CRITICAL", cls: "bg-red-50 text-red-700 border-red-200" },
};

export default function RiskBadge({ level }: { level?: string }) {
  const cfg = CONFIG[level ?? "LOW"] ?? CONFIG.LOW;
  return (
    <span className={clsx("inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border", cfg.cls)}>
      {cfg.label}
    </span>
  );
}

import { clsx } from "clsx";
import type { ReviewStatus } from "@/lib/api";

const CONFIG: Record<ReviewStatus, { label: string; cls: string }> = {
  PENDING:   { label: "PENDING",   cls: "bg-slate-50 text-slate-600 border-slate-200" },
  APPROVED:  { label: "APPROVED",  cls: "bg-green-50 text-green-700 border-green-200" },
  HELD:      { label: "HELD",      cls: "bg-amber-50 text-amber-800 border-amber-200" },
  ESCALATED: { label: "ESCALATED", cls: "bg-red-50 text-red-700 border-red-200" },
};

export default function StatusBadge({ status }: { status?: ReviewStatus | string }) {
  const cfg = CONFIG[(status as ReviewStatus) ?? "PENDING"] ?? CONFIG.PENDING;
  return (
    <span className={clsx("inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border", cfg.cls)}>
      {cfg.label}
    </span>
  );
}

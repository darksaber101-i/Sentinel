import { CheckCircle2, PauseCircle, AlertOctagon } from "lucide-react";
import type { OrderAction } from "@/lib/api";

const ICON: Record<string, React.ElementType> = {
  APPROVE: CheckCircle2, HOLD: PauseCircle, ESCALATE: AlertOctagon,
};
const COLOR: Record<string, string> = {
  APPROVE: "text-green-600", HOLD: "text-amber-700", ESCALATE: "text-red-600",
};

export default function ActionTimeline({ actions }: { actions: OrderAction[] }) {
  if (actions.length === 0) {
    return <p className="text-xs text-text-muted">No actions taken yet — this order is pending review.</p>;
  }

  return (
    <div className="space-y-3">
      {actions.map((a, i) => {
        const Icon = ICON[a.action] ?? CheckCircle2;
        return (
          <div key={i} className="flex gap-3">
            <div className={`mt-0.5 ${COLOR[a.action] ?? "text-text-secondary"}`}>
              <Icon size={14} />
            </div>
            <div className="flex-1 pb-3 border-b border-border last:border-0 last:pb-0">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-text-primary">{a.status}</p>
                <p className="text-[10px] text-text-muted">{new Date(a.created_at).toLocaleString()}</p>
              </div>
              {a.note && <p className="text-xs text-text-secondary mt-1">{a.note}</p>}
              {a.actor && <p className="text-[10px] text-text-muted mt-0.5">by {a.actor}</p>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

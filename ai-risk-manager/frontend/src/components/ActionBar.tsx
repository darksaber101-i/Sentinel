"use client";
import { useState } from "react";
import { CheckCircle2, PauseCircle, AlertOctagon } from "lucide-react";
import { api, type ReviewAction, type ReviewStatus } from "@/lib/api";
import StatusBadge from "./StatusBadge";

const ACTIONS: { action: ReviewAction; label: string; icon: React.ElementType; cls: string }[] = [
  { action: "APPROVE",  label: "Approve",              icon: CheckCircle2,  cls: "hover:bg-green-50 hover:text-green-700 hover:border-green-300" },
  { action: "HOLD",     label: "Hold for verification", icon: PauseCircle,  cls: "hover:bg-amber-50 hover:text-amber-800 hover:border-amber-300" },
  { action: "ESCALATE", label: "Escalate",              icon: AlertOctagon, cls: "hover:bg-red-50 hover:text-red-700 hover:border-red-300" },
];

interface Props {
  orderId: string;
  status: ReviewStatus;
  onActed?: (status: ReviewStatus, note: string | null) => void;
  compact?: boolean;
}

export default function ActionBar({ orderId, status, onActed, compact }: Props) {
  const [current, setCurrent] = useState(status);
  const [note, setNote]       = useState("");
  const [busy, setBusy]       = useState<ReviewAction | null>(null);

  const act = async (action: ReviewAction) => {
    setBusy(action);
    try {
      const res = await api.postOrderAction(orderId, action, note || undefined);
      setCurrent(res.review_status);
      onActed?.(res.review_status, res.note);
      setNote("");
    } finally {
      setBusy(null);
    }
  };

  if (compact) {
    return (
      <div className="flex items-center gap-1.5">
        {current !== "PENDING" ? (
          <StatusBadge status={current} />
        ) : (
          ACTIONS.map(({ action, label, icon: Icon, cls }) => (
            <button
              key={action}
              title={label}
              disabled={busy !== null}
              onClick={() => act(action)}
              className={`p-1.5 rounded-lg border border-border text-text-muted transition-colors disabled:opacity-40 ${cls}`}
            >
              <Icon size={14} />
            </button>
          ))
        )}
      </div>
    );
  }

  return (
    <div className="bg-card border border-border rounded-2xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-text-primary">Review Action</h3>
        <StatusBadge status={current} />
      </div>

      {current === "PENDING" ? (
        <>
          <textarea
            value={note}
            onChange={e => setNote(e.target.value)}
            placeholder="Optional note (e.g. reason for hold)…"
            rows={2}
            className="w-full bg-surface border border-border rounded-xl p-2.5 text-xs text-text-primary placeholder:text-text-muted resize-none focus:outline-none focus:border-amber/50 mb-3"
          />
          <div className="grid grid-cols-3 gap-2">
            {ACTIONS.map(({ action, label, icon: Icon, cls }) => (
              <button
                key={action}
                disabled={busy !== null}
                onClick={() => act(action)}
                className={`flex flex-col items-center gap-1 py-2.5 rounded-xl border border-border text-text-secondary text-[11px] font-medium transition-colors disabled:opacity-40 ${cls}`}
              >
                <Icon size={16} />
                {busy === action ? "…" : label}
              </button>
            ))}
          </div>
        </>
      ) : (
        <p className="text-xs text-text-muted">
          This order has already been reviewed. See the action history below.
        </p>
      )}
    </div>
  );
}

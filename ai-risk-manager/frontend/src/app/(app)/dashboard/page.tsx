"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import {
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer,
  XAxis, YAxis, Tooltip, Legend,
} from "recharts";
import {
  ShieldAlert, TrendingUp, Package, Target, Activity,
  ArrowRight, AlertTriangle, Sparkles,
} from "lucide-react";
import { api, type DashboardStats, type Alert, type Order } from "@/lib/api";
import { RISK_COLORS, SEVERITY_COLORS, AXIS_TICK, tooltipStyle, inr } from "@/lib/chartTheme";
import RiskBadge from "@/components/RiskBadge";

function KpiChip({ label, value, sub, icon: Icon }: {
  label: string; value: string; sub?: string; icon: React.ElementType;
}) {
  return (
    <div className="bg-card border border-border rounded-2xl p-4 flex items-center gap-3">
      <div className="p-2 rounded-lg bg-white/5">
        <Icon size={16} className="text-text-secondary" />
      </div>
      <div>
        <p className="text-text-muted text-[10px] uppercase tracking-wider">{label}</p>
        <p className="text-lg font-bold text-text-primary tabular-nums">{value}</p>
        {sub && <p className="text-text-muted text-[10px]">{sub}</p>}
      </div>
    </div>
  );
}

function AlertRow({ a }: { a: Alert }) {
  return (
    <div className="flex items-start gap-2.5 py-2.5 border-b border-border last:border-0">
      <span className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: SEVERITY_COLORS[a.severity] }} />
      <div>
        <p className="text-xs font-medium text-text-primary leading-snug">{a.title}</p>
        <p className="text-[11px] text-text-muted mt-0.5 leading-snug">{a.detail}</p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [data, setData]     = useState<DashboardStats | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [queue, setQueue]   = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState("");

  useEffect(() => {
    Promise.all([
      api.getDashboardStats(),
      api.getAlerts().catch(() => []),
      api.getReviewQueue({ status: "PENDING", page: 1, page_size: 5 }).catch(() => ({ orders: [] as Order[] })),
    ])
      .then(([d, a, q]) => { setData(d); setAlerts(a); setQueue(q.orders); })
      .catch(() => setError("Backend not running. Start with: uvicorn backend.main:app --reload"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-text-secondary">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-amber border-t-transparent rounded-full animate-spin" />
        <p>Loading control center…</p>
      </div>
    </div>
  );

  if (error) return (
    <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-red-700">
      <p className="font-semibold mb-1">Backend Unavailable</p>
      <p className="text-sm font-mono">{error}</p>
    </div>
  );

  const k    = data!.kpis;
  const cost = data!.cost_summary;
  const dist = data!.risk_distribution;
  const pieData = Object.entries(dist).map(([name, value]) => ({ name, value }));
  const catData = data!.category_data.slice(0, 8).map(c => ({
    name: c.category.substring(0, 12),
    rate: Math.round(c.return_rate * 100),
  }));
  const modelData = Object.entries(data!.model_comparison).map(([name, m]) => ({
    name: name.replace(" ", "\n"),
    F1: Math.round(m.f1 * 100) / 100,
    "ROC-AUC": Math.round(m.roc_auc * 100) / 100,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
            <ShieldAlert className="text-amber" size={24} />
            Control Center
          </h1>
          <p className="text-text-secondary text-sm mt-1">
            Sentinel — detect risk before it becomes loss. {data!.model_comparison ? Object.keys(data!.model_comparison).length : 0} models trained and compared
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-green-700 bg-green-50 border border-green-200 px-3 py-1.5 rounded-full">
          <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
          Live
        </div>
      </div>

      {/* Hero row: Net Savings + Alerts */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 hero-glow border border-border rounded-2xl p-6 animate-count-in">
          <div className="flex items-center gap-2 text-money mb-1">
            <Sparkles size={14} />
            <p className="text-xs font-semibold uppercase tracking-wider">Estimated impact · test set</p>
          </div>
          <p className="text-5xl font-bold text-text-primary tabular-nums">{inr(cost.best_net_savings)}</p>
          <p className="text-text-secondary text-sm mt-2">
            saved vs. flagging nothing ({inr(cost.baseline_cost_flag_nothing)} baseline loss), by operating at the{" "}
            <span className="text-amber font-semibold">{(cost.best_threshold * 100).toFixed(0)}% threshold</span> instead of the default 50%.
          </p>
          <Link href="/threshold-simulator" className="inline-flex items-center gap-1 text-xs text-amber hover:underline mt-3">
            Tune the policy <ArrowRight size={12} />
          </Link>
        </div>

        <div className="bg-card border border-border rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle size={14} className="text-amber" />
            <h3 className="text-sm font-semibold text-text-primary">Risk Signals</h3>
          </div>
          {alerts.length === 0 ? (
            <p className="text-xs text-text-muted mt-3">No concentration risk signals right now.</p>
          ) : (
            <div className="mt-1">
              {alerts.slice(0, 3).map((a, i) => <AlertRow key={i} a={a} />)}
            </div>
          )}
        </div>
      </div>

      {/* Action Queue preview */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-text-primary">Needs Your Decision</h3>
          <Link href="/review-queue" className="inline-flex items-center gap-1 text-xs text-amber hover:underline">
            {data!.queue_count.toLocaleString()} pending <ArrowRight size={12} />
          </Link>
        </div>
        {queue.length === 0 ? (
          <p className="text-xs text-text-muted">Queue is clear — nothing pending review.</p>
        ) : (
          <div className="space-y-0">
            {queue.map((o, i) => (
              <Link
                key={o.order_id}
                href={`/orders/${o.order_id}`}
                className={`flex items-center justify-between py-2.5 text-xs hover:bg-surface/50 -mx-2 px-2 rounded-lg transition-colors ${i !== queue.length - 1 ? "border-b border-border" : ""}`}
              >
                <span className="font-mono text-amber w-28">{o.order_id}</span>
                <span className="text-text-secondary flex-1">{o.product_category}</span>
                <RiskBadge level={o.risk_level} />
                <span className="text-amber font-semibold tabular-nums w-24 text-right">{inr(o.cost_at_stake ?? 0)}</span>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-4 gap-4">
        <KpiChip label="Orders Analyzed" value={k.total_predicted.toLocaleString()} icon={Package} />
        <KpiChip label="Return Rate" value={`${(k.return_rate * 100).toFixed(1)}%`} sub="of predicted orders" icon={TrendingUp} />
        <KpiChip label="Precision" value={`${(k.precision * 100).toFixed(1)}%`} sub="flagged orders correct" icon={Target} />
        <KpiChip label="Recall" value={`${(k.recall * 100).toFixed(1)}%`} sub="returns caught" icon={Activity} />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Risk Distribution</h3>
          <div className="flex items-center gap-6">
            <ResponsiveContainer width={160} height={160}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value" paddingAngle={3}>
                  {pieData.map(entry => (
                    <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip {...tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2">
              {pieData.map(e => (
                <div key={e.name} className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ background: RISK_COLORS[e.name] }} />
                  <span className="text-xs text-text-secondary w-16">{e.name}</span>
                  <span className="text-xs font-semibold text-text-primary">{e.value.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Return Rate by Category (%)</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={catData} layout="vertical" margin={{ left: 0, right: 16 }}>
              <XAxis type="number" domain={[0, 50]} tick={AXIS_TICK} />
              <YAxis type="category" dataKey="name" tick={AXIS_TICK} width={72} />
              <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`, "Return Rate"]} />
              <Bar dataKey="rate" fill="#f59e0b" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Model Comparison (Test Set)</h3>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={modelData} margin={{ top: 0, right: 24, bottom: 0, left: 0 }}>
            <XAxis dataKey="name" tick={AXIS_TICK} />
            <YAxis domain={[0, 1]} tick={AXIS_TICK} />
            <Tooltip {...tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="ROC-AUC" fill="#6366f1" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Importance */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Top Risk Factors (Feature Importance)</h3>
        <div className="space-y-3">
          {data!.feature_importance.map((f, i) => (
            <div key={f.feature}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-text-secondary">{f.feature.replace(/_/g, " ")}</span>
                <span className="text-text-primary font-medium">{(f.importance * 100).toFixed(1)}%</span>
              </div>
              <div className="h-1.5 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${f.importance * 100}%`,
                    background: i === 0 ? "#f59e0b" : i < 3 ? "#f97316" : "#6366f1",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

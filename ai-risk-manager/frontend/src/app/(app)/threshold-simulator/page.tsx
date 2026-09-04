"use client";
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from "recharts";
import { api, type ThresholdRow, type CostAnalysis } from "@/lib/api";
import { AXIS_TICK, tooltipStyle, inr } from "@/lib/chartTheme";

export default function ThresholdSimulator() {
  const [rows, setRows]         = useState<ThresholdRow[]>([]);
  const [cost, setCost]         = useState<CostAnalysis | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    Promise.all([api.getThresholdAnalysis(), api.getCostAnalysis()])
      .then(([t, c]) => { setRows(t); setCost(c); })
      .finally(() => setLoading(false));
  }, []);

  const selected = rows.find(r => Math.abs(r.threshold - threshold) < 0.03) ?? rows[0];

  if (loading) return <div className="text-text-secondary py-20 text-center">Loading…</div>;

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Policy Simulator</h1>
        <p className="text-text-secondary text-sm mt-1">
          Adjust the classification threshold and see how precision, recall, flagged orders, and ₹ savings change.
          Model deployment is a <em>business decision</em>, not just a technical one.
        </p>
      </div>

      {/* Concept card */}
      <div className="bg-card border border-amber/20 rounded-2xl p-5 text-xs text-text-secondary space-y-2">
        <p className="text-sm font-semibold text-amber">Why does the threshold matter?</p>
        <p>
          The model outputs a probability (e.g., 0.73 = 73% chance of return). We still need to choose
          a <strong className="text-text-primary">threshold</strong> to convert that into an action.
        </p>
        <p>
          <strong className="text-text-primary">Low threshold (e.g., 30%):</strong> Flag more orders →
          higher recall (catch more returns) but lower precision (more false alarms).
        </p>
        <p>
          <strong className="text-text-primary">High threshold (e.g., 70%):</strong> Flag fewer orders →
          higher precision (more confident flags) but lower recall (miss more returns).
        </p>
        <p>The right threshold depends on the cost of a missed return vs. the cost of a false alarm.</p>
      </div>

      {/* Cost assumptions + optimal threshold */}
      {cost && (
        <div className="bg-card border border-border rounded-2xl p-5 text-xs text-text-secondary space-y-2">
          <p className="text-sm font-semibold text-text-primary">₹ cost model (test set)</p>
          <p>
            False alarm (flag a good order): <strong className="text-text-primary">{inr(cost.assumptions.review_cost_per_flag)}</strong> manual
            review cost. Missed return (unflagged): <strong className="text-text-primary">{Math.round(cost.assumptions.return_cost_pct_of_order_value * 100)}%
            </strong> of order value in reverse logistics + refund processing, only{" "}
            <strong className="text-text-primary">{Math.round((1 - cost.assumptions.intervention_effectiveness) * 100)}%</strong> of
            that cost still lands even when caught, since intervention isn&apos;t perfect.
          </p>
          <div className="flex flex-wrap gap-x-6 gap-y-1 pt-1">
            <span>Flag-nothing baseline loss: <strong className="text-text-primary">{inr(cost.assumptions.baseline_cost_flag_nothing)}</strong></span>
            <span>Best threshold by ₹ savings: <strong className="text-amber">{(cost.best_threshold * 100).toFixed(0)}%</strong></span>
            <span>Max net savings: <strong className="text-green-500">{inr(cost.best_net_savings)}</strong></span>
          </div>
        </div>
      )}

      {/* Slider */}
      <div className="bg-card border border-border rounded-2xl p-6">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-semibold text-text-primary">Classification Threshold</p>
          <p className="text-2xl font-bold text-amber">{(threshold * 100).toFixed(0)}%</p>
        </div>
        <input
          type="range" min={0.10} max={0.90} step={0.05}
          value={threshold}
          onChange={e => setThreshold(Number(e.target.value))}
          className="w-full accent-amber"
        />
        <div className="flex justify-between text-xs text-text-muted mt-1">
          <span>10% (Aggressive)</span>
          <span>90% (Conservative)</span>
        </div>
      </div>

      {/* Selected metrics */}
      {selected && (
        <div className="grid grid-cols-5 gap-3">
          {[
            { label: "Precision",      value: `${(selected.precision * 100).toFixed(1)}%`, sub: "flagged correctly" },
            { label: "Recall",         value: `${(selected.recall * 100).toFixed(1)}%`,    sub: "returns caught" },
            { label: "F1 Score",       value: selected.f1.toFixed(3),                       sub: "balanced metric" },
            { label: "Flagged Orders", value: selected.flagged_orders.toLocaleString(),     sub: `${(selected.flagged_pct * 100).toFixed(1)}% of all` },
            { label: "Net Savings",    value: inr(selected.net_savings),                    sub: "vs. flag nothing" },
          ].map(c => (
            <div key={c.label} className="bg-card border border-border rounded-2xl p-4 text-center">
              <p className="text-xs text-text-muted uppercase tracking-wider mb-1">{c.label}</p>
              <p className="text-2xl font-bold text-text-primary">{c.value}</p>
              <p className="text-[10px] text-text-muted mt-0.5">{c.sub}</p>
            </div>
          ))}
        </div>
      )}

      {/* Chart */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Precision · Recall · F1 vs Threshold</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={rows}>
            <XAxis
              dataKey="threshold"
              tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              tick={AXIS_TICK}
            />
            <YAxis domain={[0, 1]} tick={AXIS_TICK} />
            <Tooltip
              {...tooltipStyle}
              labelFormatter={v => `Threshold: ${(Number(v) * 100).toFixed(0)}%`}
              formatter={(v: number, name) => [`${(v * 100).toFixed(1)}%`, name]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine x={threshold} stroke="#f59e0b" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="precision" stroke="#6366f1" dot={false} strokeWidth={2} name="Precision" />
            <Line type="monotone" dataKey="recall"    stroke="#22c55e" dot={false} strokeWidth={2} name="Recall" />
            <Line type="monotone" dataKey="f1"        stroke="#f59e0b" dot={false} strokeWidth={2.5} name="F1" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Net savings chart */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Net ₹ Savings vs Threshold (vs. flag-nothing baseline)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rows}>
            <XAxis dataKey="threshold" tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={AXIS_TICK} />
            <YAxis tickFormatter={v => inr(v)} tick={AXIS_TICK} width={70} />
            <Tooltip
              {...tooltipStyle}
              labelFormatter={v => `Threshold: ${(Number(v) * 100).toFixed(0)}%`}
              formatter={(v: number) => [inr(v), "Net Savings"]}
            />
            <ReferenceLine x={threshold} stroke="#f59e0b" strokeDasharray="4 4" />
            {cost && <ReferenceLine x={cost.best_threshold} stroke="#22c55e" strokeDasharray="2 2" label={{ value: "optimal", fontSize: 10, fill: "#22c55e" }} />}
            <Line type="monotone" dataKey="net_savings" stroke="#22c55e" dot={false} strokeWidth={2} name="Net Savings" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Flagged volume chart */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Flagged Orders vs Threshold</h3>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={rows}>
            <XAxis dataKey="threshold" tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={AXIS_TICK} />
            <YAxis tick={AXIS_TICK} />
            <Tooltip
              {...tooltipStyle}
              labelFormatter={v => `Threshold: ${(Number(v) * 100).toFixed(0)}%`}
            />
            <ReferenceLine x={threshold} stroke="#f59e0b" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="flagged_orders" stroke="#f97316" dot={false} strokeWidth={2} name="Flagged Orders" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

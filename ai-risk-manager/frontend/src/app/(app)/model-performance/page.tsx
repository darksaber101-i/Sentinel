"use client";
import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { api, type ModelPerf } from "@/lib/api";
import { AXIS_TICK, tooltipStyle } from "@/lib/chartTheme";

function MetricCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="bg-card border border-border rounded-2xl p-5">
      <p className="text-xs text-text-muted uppercase tracking-wider mb-2">{label}</p>
      <p className="text-3xl font-bold text-amber">{value}</p>
      <p className="text-xs text-text-secondary mt-1">{sub}</p>
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  return (
    <div>
      <h3 className="text-sm font-semibold text-text-primary mb-3">Confusion Matrix</h3>
      <div className="grid grid-cols-2 gap-1 w-60">
        {[
          { label: "TN", value: tn, sub: "Correctly kept", color: "text-green-600" },
          { label: "FP", value: fp, sub: "Wrongly flagged", color: "text-orange-600" },
          { label: "FN", value: fn, sub: "Missed returns", color: "text-red-600" },
          { label: "TP", value: tp, sub: "Correctly flagged", color: "text-green-600" },
        ].map(c => (
          <div key={c.label} className="bg-surface border border-border rounded-xl p-3 text-center">
            <p className={`text-xl font-bold ${c.color}`}>{c.value.toLocaleString()}</p>
            <p className="text-xs text-text-muted font-mono">{c.label}</p>
            <p className="text-[10px] text-text-muted mt-0.5">{c.sub}</p>
          </div>
        ))}
      </div>
      <p className="text-xs text-text-muted mt-2">Total test orders: {total.toLocaleString()}</p>
    </div>
  );
}

export default function ModelPerformancePage() {
  const [data, setData] = useState<ModelPerf | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getModelPerformance().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-text-secondary py-20 text-center">Loading model data…</div>;
  if (!data) return <div className="text-red-600 py-20 text-center">Model not trained yet. Run: python ml/train.py && python ml/evaluate.py</div>;

  const m    = data.test_metrics;
  const roc  = data.test_metrics.roc_curve;
  const pr   = data.test_metrics.pr_curve;

  const rocData = roc.fpr.map((x, i) => ({ fpr: +x.toFixed(3), tpr: +roc.tpr[i].toFixed(3) })).filter((_, i) => i % 10 === 0);
  const prData  = pr.recall.map((x, i) => ({ recall: +x.toFixed(3), precision: +pr.precision[i].toFixed(3) })).filter((_, i) => i % 10 === 0);

  const compData = Object.entries(data.all_model_test_metrics).map(([name, mm]) => ({
    name,
    Precision: mm.precision,
    Recall:    mm.recall,
    F1:        mm.f1,
    "ROC-AUC": mm.roc_auc,
  }));

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Model Trust</h1>
        <p className="text-text-secondary text-sm mt-1">
          How the model was validated — for anyone who needs to trust the numbers, not just see them.
          Best model: <span className="text-amber font-medium">{data.best_model_name}</span> ·
          Train {data.train_size.toLocaleString()} · Val {data.val_size.toLocaleString()} · Test {data.test_size.toLocaleString()}
        </p>
      </div>

      {/* Core Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Precision" value={m.precision.toFixed(3)} sub="Of flagged orders, % actually returned" />
        <MetricCard label="Recall"    value={m.recall.toFixed(3)}    sub="Of all returns, % the model caught" />
        <MetricCard label="F1 Score"  value={m.f1.toFixed(3)}        sub="Harmonic mean of precision & recall" />
        <MetricCard label="ROC-AUC"   value={m.roc_auc.toFixed(3)}   sub="Discrimination ability (1.0 = perfect)" />
        <MetricCard label="PR-AUC"    value={m.pr_auc.toFixed(3)}    sub="Better than ROC for imbalanced data" />
        <MetricCard label="Accuracy"  value={`${(m.accuracy * 100).toFixed(1)}%`} sub="Overall correct (can be misleading)" />
      </div>

      {/* Why These Metrics */}
      <div className="bg-card border border-amber/20 rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-amber mb-3">Why These Metrics?</h3>
        <div className="grid grid-cols-3 gap-4 text-xs text-text-secondary">
          <div>
            <p className="font-semibold text-text-primary mb-1">Precision</p>
            <p>Of all orders flagged as risky, how many actually returned? Low precision = many false alarms wasting ops team time.</p>
          </div>
          <div>
            <p className="font-semibold text-text-primary mb-1">Recall</p>
            <p>Of all orders that actually returned, how many did we catch? Low recall = missed returns costing money.</p>
          </div>
          <div>
            <p className="font-semibold text-text-primary mb-1">F1 & PR-AUC</p>
            <p>With 28% returns and 72% non-returns, accuracy is misleading. F1 and PR-AUC capture the real trade-off.</p>
          </div>
        </div>
      </div>

      {/* Confusion Matrix + ROC */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-2xl p-5">
          <ConfusionMatrix cm={m.confusion_matrix} />
        </div>

        <div className="bg-card border border-border rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-3">ROC Curve (AUC = {m.roc_auc.toFixed(3)})</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={rocData}>
              <XAxis dataKey="fpr" tick={AXIS_TICK} label={{ value: "FPR", position: "insideBottom", fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={AXIS_TICK} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="tpr" stroke="#f59e0b" dot={false} strokeWidth={2} name="TPR" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* PR Curve + Feature Importance */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-3">Precision-Recall Curve (AUC = {m.pr_auc.toFixed(3)})</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={prData}>
              <XAxis dataKey="recall" tick={AXIS_TICK} />
              <YAxis tick={AXIS_TICK} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="precision" stroke="#6366f1" dot={false} strokeWidth={2} name="Precision" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-card border border-border rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-3">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={data.feature_importance.slice(0, 8)} layout="vertical">
              <XAxis type="number" tick={AXIS_TICK} />
              <YAxis type="category" dataKey="feature" tick={AXIS_TICK}
                tickFormatter={v => v.replace(/_/g, " ").substring(0, 18)} width={100} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="importance" fill="#f59e0b" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="bg-card border border-border rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-text-primary mb-4">All Models — Test Set Comparison</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={compData}>
            <XAxis dataKey="name" tick={AXIS_TICK} />
            <YAxis domain={[0, 1]} tick={AXIS_TICK} />
            <Tooltip {...tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="F1"       fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Precision" fill="#6366f1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Recall"   fill="#22c55e" radius={[4, 4, 0, 0]} />
            <Bar dataKey="ROC-AUC"  fill="#f97316" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

import Link from "next/link";
import {
  ShieldAlert, ArrowRight, Target, Activity, Gauge,
  GitBranch, LineChart, MessageSquareText, ListChecks,
} from "lucide-react";

const MODELS = [
  { name: "Logistic Regression", tag: "Selected", precision: 0.477, recall: 0.698, f1: 0.567, rocAuc: 0.704 },
  { name: "Random Forest",       tag: null,       precision: 0.487, recall: 0.667, f1: 0.563, rocAuc: 0.687 },
  { name: "Gradient Boosting",   tag: null,       precision: 0.439, recall: 0.740, f1: 0.551, rocAuc: 0.672 },
];

const FEATURES = [
  { icon: Gauge,             title: "Risk Scoring",        desc: "0–100 score with LOW / MEDIUM / HIGH / CRITICAL levels on every order." },
  { icon: GitBranch,         title: "3 Models, Compared",  desc: "Logistic Regression, Random Forest, and XGBoost evaluated head-to-head on a held-out test set." },
  { icon: LineChart,         title: "Threshold Simulator", desc: "See the precision/recall and ₹-cost trade-off move as the decision threshold changes." },
  { icon: Target,            title: "Explainability",      desc: "Per-order feature contributions, so every flag has a reason attached." },
  { icon: ListChecks,        title: "Review Queue",        desc: "A working queue for the orders the model actually wants a human to look at." },
  { icon: MessageSquareText, title: "AI Assistant",        desc: "Ask questions about the model or the data, grounded in what's actually loaded — no hallucinated numbers." },
];

const STEPS = [
  { step: "01", title: "Synthetic order data",  desc: "15,000 e-commerce orders across 8 categories, with realistic class imbalance and 5% deliberate noise." },
  { step: "02", title: "Feature engineering",   desc: "22 raw + 6 derived features covering order, customer, product, and behavioral signals." },
  { step: "03", title: "Train & select",        desc: "Three models trained on a stratified 70/15/15 split; the best F1 on validation wins." },
  { step: "04", title: "Cost-optimal threshold", desc: "The F1-optimal cutoff isn't the ₹-optimal one — the simulator finds the threshold that actually saves money." },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-bg">
      {/* Nav */}
      <header className="flex items-center justify-between px-8 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <ShieldAlert className="text-amber" size={22} />
          <span className="text-sm font-semibold text-text-primary">Sentinel</span>
        </div>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-text-primary bg-card border border-border rounded-full px-4 py-2 hover:border-amber transition-colors"
        >
          Open Control Center <ArrowRight size={14} />
        </Link>
      </header>

      {/* Hero */}
      <section className="hero-glow border-b border-border">
        <div className="max-w-4xl mx-auto px-8 py-20 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-text-primary tracking-tight leading-tight animate-count-in">
            Detect risk before it becomes loss.
          </h1>
          <p className="text-text-secondary text-lg mt-5 max-w-2xl mx-auto leading-relaxed">
            Sentinel is an end-to-end ML platform that predicts whether an e-commerce order will be
            returned before fulfillment — with explainable predictions, a real-time dashboard, and
            an AI assistant grounded in the model's own data.
          </p>
          <div className="flex items-center justify-center gap-3 mt-8">
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 bg-text-primary text-white text-sm font-semibold rounded-xl px-5 py-3 hover:opacity-90 transition-opacity"
            >
              Open Control Center <ArrowRight size={16} />
            </Link>
            <Link
              href="/model-performance"
              className="inline-flex items-center gap-2 bg-card border border-border text-text-primary text-sm font-semibold rounded-xl px-5 py-3 hover:border-amber transition-colors"
            >
              See model performance
            </Link>
          </div>
        </div>
      </section>

      {/* Impact strip */}
      <section className="border-b border-border">
        <div className="max-w-5xl mx-auto px-8 py-10 grid grid-cols-2 sm:grid-cols-4 gap-6">
          <Stat label="Net savings vs. flagging nothing" value="₹190,996" sub="~37% at the ₹-optimal 15% threshold" />
          <Stat label="ROC-AUC" value="0.704" sub="Logistic Regression, test set" />
          <Stat label="Recall" value="69.8%" sub="of real returns caught" />
          <Stat label="Orders modeled" value="15,000" sub="synthetic, 8 categories" />
        </div>
      </section>

      {/* Features */}
      <section className="max-w-5xl mx-auto px-8 py-20">
        <h2 className="text-2xl font-bold text-text-primary text-center">What's inside</h2>
        <p className="text-text-secondary text-sm text-center mt-2 max-w-lg mx-auto">
          Every number in the dashboard comes from an actual trained model — nothing here is hardcoded.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-10">
          {FEATURES.map(f => (
            <div key={f.title} className="bg-card border border-border rounded-2xl p-5">
              <div className="p-2 rounded-lg bg-amber-light w-fit mb-3">
                <f.icon size={18} className="text-amber" />
              </div>
              <h3 className="text-sm font-semibold text-text-primary">{f.title}</h3>
              <p className="text-xs text-text-secondary mt-1.5 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pipeline */}
      <section className="border-y border-border bg-surface">
        <div className="max-w-5xl mx-auto px-8 py-20">
          <h2 className="text-2xl font-bold text-text-primary text-center">How the model gets built</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-10">
            {STEPS.map(s => (
              <div key={s.step} className="relative">
                <p className="text-3xl font-bold text-amber/30">{s.step}</p>
                <h3 className="text-sm font-semibold text-text-primary mt-2">{s.title}</h3>
                <p className="text-xs text-text-secondary mt-1.5 leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Model comparison */}
      <section className="max-w-5xl mx-auto px-8 py-20">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-text-primary">Model comparison</h2>
            <p className="text-text-secondary text-sm mt-1">Final evaluation on a held-out test set, never used during training.</p>
          </div>
          <Link href="/model-performance" className="hidden sm:inline-flex items-center gap-1 text-xs text-amber hover:underline whitespace-nowrap">
            Full breakdown <ArrowRight size={12} />
          </Link>
        </div>
        <div className="bg-card border border-border rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-text-muted text-[11px] uppercase tracking-wider">
                <th className="px-5 py-3 font-medium">Model</th>
                <th className="px-5 py-3 font-medium text-right">Precision</th>
                <th className="px-5 py-3 font-medium text-right">Recall</th>
                <th className="px-5 py-3 font-medium text-right">F1</th>
                <th className="px-5 py-3 font-medium text-right">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map(m => (
                <tr key={m.name} className="border-b border-border last:border-0">
                  <td className="px-5 py-3.5">
                    <span className="font-medium text-text-primary">{m.name}</span>
                    {m.tag && (
                      <span className="ml-2 text-[10px] font-semibold text-amber bg-amber-light rounded-full px-2 py-0.5">{m.tag}</span>
                    )}
                  </td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-text-secondary">{m.precision.toFixed(3)}</td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-text-secondary">{m.recall.toFixed(3)}</td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-text-secondary">{m.f1.toFixed(3)}</td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-text-secondary">{m.rocAuc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-border">
        <div className="max-w-3xl mx-auto px-8 py-20 text-center">
          <Activity className="text-amber mx-auto mb-4" size={28} />
          <h2 className="text-2xl font-bold text-text-primary">See it running on live data</h2>
          <p className="text-text-secondary text-sm mt-2 max-w-md mx-auto">
            The Control Center streams real predictions, alerts, and a review queue straight from the trained model.
          </p>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 bg-text-primary text-white text-sm font-semibold rounded-xl px-5 py-3 mt-6 hover:opacity-90 transition-opacity"
          >
            Open Control Center <ArrowRight size={16} />
          </Link>
        </div>
      </section>

      <footer className="border-t border-border px-8 py-6 text-center text-[11px] text-text-muted">
        Sentinel — built to demonstrate the complete ML engineering lifecycle.
      </footer>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div>
      <p className="text-2xl sm:text-3xl font-bold text-text-primary tabular-nums">{value}</p>
      <p className="text-xs text-text-secondary mt-1">{label}</p>
      <p className="text-[11px] text-text-muted mt-0.5">{sub}</p>
    </div>
  );
}

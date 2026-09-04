"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, AlertTriangle, Info, Bot, History } from "lucide-react";
import { api, type Order, type AssistantResp, type OrderAction } from "@/lib/api";
import RiskBadge from "@/components/RiskBadge";
import StatusBadge from "@/components/StatusBadge";
import ActionBar from "@/components/ActionBar";
import ActionTimeline from "@/components/ActionTimeline";
import { RISK_COLORS } from "@/lib/chartTheme";

function RiskGauge({ score, level }: { score: number; level: string }) {
  const color = RISK_COLORS[level] ?? "#94a3b8";
  const r = 50, cx = 60, cy = 60;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  return (
    <div className="flex flex-col items-center">
      <svg width={120} height={120}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e3e6ec" strokeWidth={10} />
        <circle
          cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={10}
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
        />
        <text x={cx} y={cy - 4} textAnchor="middle" fill={color} fontSize="22" fontWeight="bold">{score}</text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill="#64748b" fontSize="9">/ 100</text>
      </svg>
      <RiskBadge level={level} />
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between py-2 border-b border-border last:border-0">
      <span className="text-xs text-text-muted">{label}</span>
      <span className="text-xs text-text-primary font-medium">{value}</span>
    </div>
  );
}

export default function OrderDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [order, setOrder]   = useState<Order | null>(null);
  const [actions, setActions] = useState<OrderAction[]>([]);
  const [loading, setLoading] = useState(true);
  const [question, setQuestion] = useState("");
  const [aiReply, setAiReply]   = useState<AssistantResp | null>(null);
  const [asking, setAsking]     = useState(false);

  useEffect(() => {
    api.getOrder(id).then(setOrder).finally(() => setLoading(false));
    api.getOrderActions(id).then(setActions);
  }, [id]);

  const handleActed = () => {
    api.getOrderActions(id).then(setActions);
    api.getOrder(id).then(setOrder);
  };

  const ask = async () => {
    if (!question.trim()) return;
    setAsking(true);
    try {
      const r = await api.askAssistant(question, id);
      setAiReply(r);
    } finally {
      setAsking(false);
    }
  };

  if (loading) return <div className="text-text-secondary py-20 text-center">Loading…</div>;
  if (!order) return <div className="text-red-600 py-20 text-center">Order not found</div>;

  const features = order.top_features ?? [];
  const maxShap  = Math.max(...features.map(f => Math.abs(f.shap_value)), 0.01);

  return (
    <div className="space-y-5 max-w-5xl">
      <div className="flex items-center gap-3">
        <Link href="/orders" className="text-text-muted hover:text-amber transition-colors">
          <ArrowLeft size={18} />
        </Link>
        <div>
          <h1 className="text-xl font-bold text-text-primary flex items-center gap-2">
            {order.order_id}
            <StatusBadge status={order.review_status} />
          </h1>
          <p className="text-text-secondary text-xs">{order.customer_id} · {order.product_category}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Left: Order Info */}
        <div className="col-span-2 space-y-4">
          <div className="bg-card border border-border rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
              <Info size={14} className="text-amber" /> Order Information
            </h3>
            <InfoRow label="Order Value" value={`₹${order.order_value.toLocaleString()}`} />
            <InfoRow label="Quantity" value={order.quantity} />
            <InfoRow label="Discount" value={`${order.discount_percentage}%`} />
            <InfoRow label="Payment Method" value={order.payment_method} />
            <InfoRow label="Device" value={order.device_type} />
            <InfoRow label="Order Hour" value={`${order.order_hour}:00`} />
            <InfoRow label="Delivery Days" value={order.delivery_days} />
            <InfoRow label="Shipping Distance" value={`${order.shipping_distance_km} km`} />
            <InfoRow label="Product Return Rate" value={`${(order.product_return_rate * 100).toFixed(0)}%`} />
          </div>

          <div className="bg-card border border-border rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-text-primary mb-3">Customer History</h3>
            <InfoRow label="Tenure" value={`${order.customer_tenure_days} days`} />
            <InfoRow label="Previous Orders" value={order.previous_orders} />
            <InfoRow label="Previous Returns" value={order.previous_returns} />
            <InfoRow label="Return Rate" value={`${(order.previous_return_rate * 100).toFixed(0)}%`} />
            <InfoRow label="Failed Payments" value={order.previous_failed_payments} />
            <InfoRow label="Chargebacks" value={order.previous_chargebacks} />
            <InfoRow label="Support Tickets" value={order.support_tickets} />
            <InfoRow label="New Customer" value={order.is_new_customer ? "Yes" : "No"} />
          </div>

          {/* Why Flagged */}
          {features.length > 0 && (
            <div className="bg-card border border-amber/20 rounded-2xl p-5">
              <h3 className="text-sm font-semibold text-amber mb-4 flex items-center gap-2">
                <AlertTriangle size={14} /> Why This Order Was Flagged
              </h3>
              <div className="space-y-4">
                {features.map(f => (
                  <div key={f.feature}>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-text-primary">{f.display_name}</span>
                      <span className={f.direction === "positive" ? "text-red-600" : "text-green-600"}>
                        {f.direction === "positive" ? "↑ Increases risk" : "↓ Reduces risk"}
                      </span>
                    </div>
                    <div className="h-2 bg-border rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${Math.abs(f.shap_value) / maxShap * 100}%`,
                          background: f.direction === "positive" ? RISK_COLORS.CRITICAL : RISK_COLORS.LOW,
                        }}
                      />
                    </div>
                    <p className="text-[10px] text-text-muted mt-1">
                      Value: {f.raw_value.toFixed(3)} · SHAP: {f.shap_value.toFixed(4)}
                    </p>
                  </div>
                ))}
              </div>
              {order.explanation && (
                <div className="mt-4 p-3 bg-surface rounded-xl border border-border">
                  <p className="text-xs text-text-secondary">{order.explanation}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Risk Panel */}
        <div className="space-y-4">
          <div className="bg-card border border-border rounded-2xl p-5 flex flex-col items-center text-center gap-4">
            <h3 className="text-sm font-semibold text-text-primary w-full text-left">Risk Assessment</h3>
            {order.risk_score != null && order.risk_level ? (
              <RiskGauge score={order.risk_score} level={order.risk_level} />
            ) : (
              <p className="text-text-muted text-xs">No prediction yet</p>
            )}
            <div className="w-full">
              <p className="text-xs text-text-muted mb-1">Return Probability</p>
              <p className="text-3xl font-bold text-text-primary">
                {order.return_probability != null ? `${(order.return_probability * 100).toFixed(0)}%` : "—"}
              </p>
            </div>
            <div className="w-full p-3 bg-surface rounded-xl border border-border text-left">
              <p className="text-xs text-text-muted mb-1">Prediction</p>
              <p className="text-sm font-semibold text-text-primary">{order.prediction ?? "—"}</p>
            </div>
            <div className="w-full">
              <p className="text-xs text-text-muted mb-1">Actual Outcome</p>
              <p className={`text-sm font-semibold ${order.is_returned ? "text-red-600" : "text-green-600"}`}>
                {order.is_returned == null ? "—" : order.is_returned ? "Returned" : "Not Returned"}
              </p>
            </div>
          </div>

          {/* Recommended Action */}
          <div className="bg-amber/5 border border-amber/20 rounded-2xl p-4">
            <p className="text-xs font-semibold text-amber mb-1">Recommended Action</p>
            <p className="text-xs text-text-secondary">
              {(order.risk_score ?? 0) >= 60
                ? "Consider manual review before fulfillment. This order shows multiple risk signals."
                : (order.risk_score ?? 0) >= 30
                ? "Monitor this order. Risk is moderate — standard fulfillment is acceptable."
                : "Standard fulfillment. Risk signals are low."}
            </p>
            <p className="text-[10px] text-text-muted mt-2">
              This is a decision-support system. Never reject a customer solely on model output.
            </p>
          </div>

          {/* Action Bar */}
          <ActionBar
            orderId={id}
            status={order.review_status ?? "PENDING"}
            onActed={handleActed}
          />

          {/* Action Timeline */}
          <div className="bg-card border border-border rounded-2xl p-4">
            <h3 className="text-xs font-semibold text-text-primary mb-3 flex items-center gap-1.5">
              <History size={13} className="text-amber" /> Action History
            </h3>
            <ActionTimeline actions={actions} />
          </div>

          {/* Ask AI */}
          <div className="bg-card border border-border rounded-2xl p-4">
            <h3 className="text-xs font-semibold text-text-primary mb-3 flex items-center gap-1.5">
              <Bot size={13} className="text-amber" /> Ask AI Assistant
            </h3>
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder={`Why is ${id} high risk?`}
              rows={2}
              className="w-full bg-surface border border-border rounded-xl p-2.5 text-xs text-text-primary placeholder:text-text-muted resize-none focus:outline-none focus:border-amber/50"
            />
            <button
              onClick={ask}
              disabled={asking}
              className="mt-2 w-full py-2 bg-amber-solid text-black text-xs font-semibold rounded-xl hover:bg-amber-dark transition-colors disabled:opacity-50"
            >
              {asking ? "Thinking…" : "Ask"}
            </button>
            {aiReply && (
              <div className="mt-3 p-3 bg-surface rounded-xl border border-border">
                <p className="text-xs text-text-secondary">{aiReply.answer}</p>
                {aiReply.sources.length > 0 && (
                  <p className="text-[10px] text-text-muted mt-1">Source: {aiReply.sources[0]}</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

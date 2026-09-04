"use client";
import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight, ExternalLink, CheckCircle2, PauseCircle, AlertOctagon, ListChecks } from "lucide-react";
import { api, type Order, type ReviewAction } from "@/lib/api";
import RiskBadge from "@/components/RiskBadge";
import ActionBar from "@/components/ActionBar";
import { inr } from "@/lib/chartTheme";

const STATUS_FILTERS = ["PENDING", "APPROVED", "HELD", "ESCALATED", "ALL"];

export default function ReviewQueuePage() {
  const [orders, setOrders]     = useState<Order[]>([]);
  const [total, setTotal]       = useState(0);
  const [page, setPage]         = useState(1);
  const [status, setStatus]     = useState("PENDING");
  const [loading, setLoading]   = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState<ReviewAction | null>(null);
  const PAGE_SIZE = 25;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getReviewQueue({ status, page, page_size: PAGE_SIZE });
      setOrders(res.orders);
      setTotal(res.total);
      setSelected(new Set());
    } finally {
      setLoading(false);
    }
  }, [status, page]);

  useEffect(() => { load(); }, [load]);

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const totalAtStake = orders.reduce((s, o) => s + (o.cost_at_stake ?? 0), 0);

  const toggle = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    setSelected(prev => prev.size === orders.length ? new Set() : new Set(orders.map(o => o.order_id)));
  };

  const bulkAct = async (action: ReviewAction) => {
    setBulkBusy(action);
    try {
      await Promise.all(Array.from(selected).map(id => api.postOrderAction(id, action)));
      await load();
    } finally {
      setBulkBusy(null);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
            <ListChecks className="text-amber" size={22} /> Review Queue
          </h1>
          <p className="text-text-secondary text-sm mt-1">
            HIGH / CRITICAL orders sorted by ₹ at stake · {total.toLocaleString()} orders
            {status === "PENDING" && orders.length > 0 && (
              <> · <span className="text-amber font-medium">{inr(totalAtStake)}</span> at stake on this page</>
            )}
          </p>
        </div>
      </div>

      {/* Status filter */}
      <div className="flex gap-1">
        {STATUS_FILTERS.map(f => (
          <button
            key={f}
            onClick={() => { setStatus(f); setPage(1); }}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              status === f
                ? "bg-amber-solid text-black font-semibold"
                : "bg-card border border-border text-text-secondary hover:text-text-primary"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Bulk action toolbar */}
      {selected.size > 0 && (
        <div className="flex items-center gap-3 bg-amber/5 border border-amber/20 rounded-xl px-4 py-2.5">
          <span className="text-xs text-text-primary font-medium">{selected.size} selected</span>
          <div className="flex-1" />
          <button onClick={() => bulkAct("APPROVE")}  disabled={bulkBusy !== null} className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-border text-text-secondary hover:bg-green-50 hover:text-green-700 disabled:opacity-40">
            <CheckCircle2 size={13} /> Approve
          </button>
          <button onClick={() => bulkAct("HOLD")}     disabled={bulkBusy !== null} className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-border text-text-secondary hover:bg-amber-50 hover:text-amber-800 disabled:opacity-40">
            <PauseCircle size={13} /> Hold
          </button>
          <button onClick={() => bulkAct("ESCALATE")} disabled={bulkBusy !== null} className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-border text-text-secondary hover:bg-red-50 hover:text-red-700 disabled:opacity-40">
            <AlertOctagon size={13} /> Escalate
          </button>
        </div>
      )}

      {/* Table */}
      <div className="bg-card border border-border rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-left">
                <th className="px-4 py-3 w-8">
                  <input type="checkbox" checked={orders.length > 0 && selected.size === orders.length} onChange={toggleAll} />
                </th>
                {["Order ID", "Customer", "Category", "Value", "Risk Score", "Level", "₹ At Stake", "Status"].map(h => (
                  <th key={h} className="px-4 py-3 text-xs text-text-muted font-medium uppercase tracking-wider">{h}</th>
                ))}
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={10} className="text-center py-16 text-text-muted">Loading…</td></tr>
              ) : orders.length === 0 ? (
                <tr><td colSpan={10} className="text-center py-16 text-text-muted">No orders in this queue — nice and clear.</td></tr>
              ) : orders.map((o, i) => (
                <tr key={o.order_id} className={`border-b border-border hover:bg-surface/50 transition-colors ${i % 2 === 0 ? "" : "bg-surface/20"}`}>
                  <td className="px-4 py-3">
                    <input type="checkbox" checked={selected.has(o.order_id)} onChange={() => toggle(o.order_id)} />
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-amber">{o.order_id}</td>
                  <td className="px-4 py-3 text-text-secondary text-xs">{o.customer_id}</td>
                  <td className="px-4 py-3 text-text-primary">{o.product_category}</td>
                  <td className="px-4 py-3 text-text-primary font-medium">₹{o.order_value.toLocaleString()}</td>
                  <td className="px-4 py-3 text-text-primary font-medium">{o.risk_score}</td>
                  <td className="px-4 py-3"><RiskBadge level={o.risk_level} /></td>
                  <td className="px-4 py-3 text-amber font-semibold tabular-nums">{inr(o.cost_at_stake ?? 0)}</td>
                  <td className="px-4 py-3">
                    <ActionBar orderId={o.order_id} status={o.review_status ?? "PENDING"} compact onActed={load} />
                  </td>
                  <td className="px-4 py-3">
                    <Link href={`/orders/${o.order_id}`} className="text-text-muted hover:text-amber transition-colors">
                      <ExternalLink size={14} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between px-4 py-3 border-t border-border">
          <p className="text-xs text-text-muted">Page {page} of {Math.max(totalPages, 1)} · {total.toLocaleString()} orders</p>
          <div className="flex gap-2">
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="p-1.5 rounded-lg border border-border text-text-secondary hover:text-text-primary disabled:opacity-30">
              <ChevronLeft size={14} />
            </button>
            <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages} className="p-1.5 rounded-lg border border-border text-text-secondary hover:text-text-primary disabled:opacity-30">
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

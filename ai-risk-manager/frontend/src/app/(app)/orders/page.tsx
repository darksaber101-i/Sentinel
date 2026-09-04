"use client";
import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Search, ChevronLeft, ChevronRight, ExternalLink } from "lucide-react";
import { api, type Order } from "@/lib/api";
import RiskBadge from "@/components/RiskBadge";
import StatusBadge from "@/components/StatusBadge";
import { RISK_COLORS } from "@/lib/chartTheme";

const FILTERS = ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"];

export default function OrdersPage() {
  const [orders, setOrders]       = useState<Order[]>([]);
  const [total, setTotal]         = useState(0);
  const [page, setPage]           = useState(1);
  const [riskFilter, setRisk]     = useState("ALL");
  const [search, setSearch]       = useState("");
  const [loading, setLoading]     = useState(true);
  const PAGE_SIZE = 50;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getOrders({
        page, page_size: PAGE_SIZE,
        risk_level: riskFilter === "ALL" ? undefined : riskFilter,
        search: search || undefined,
      });
      setOrders(res.orders);
      setTotal(res.total);
    } finally {
      setLoading(false);
    }
  }, [page, riskFilter, search]);

  useEffect(() => { load(); }, [load]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Orders</h1>
        <p className="text-text-secondary text-sm">{total.toLocaleString()} orders · click any row for full risk detail</p>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1); }}
            placeholder="Search order / customer ID…"
            className="w-full bg-card border border-border rounded-xl pl-9 pr-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-amber/50"
          />
        </div>
        <div className="flex gap-1">
          {FILTERS.map(f => (
            <button
              key={f}
              onClick={() => { setRisk(f); setPage(1); }}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                riskFilter === f
                  ? "bg-amber-solid text-black font-semibold"
                  : "bg-card border border-border text-text-secondary hover:text-text-primary"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="bg-card border border-border rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-left">
                {["Order ID","Customer","Category","Value","Discount","Risk Score","Level","Status","Prediction","Actual"].map(h => (
                  <th key={h} className="px-4 py-3 text-xs text-text-muted font-medium uppercase tracking-wider">{h}</th>
                ))}
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={11} className="text-center py-16 text-text-muted">Loading…</td></tr>
              ) : orders.length === 0 ? (
                <tr><td colSpan={11} className="text-center py-16 text-text-muted">No orders found</td></tr>
              ) : orders.map((o, i) => (
                <tr
                  key={o.order_id}
                  className={`border-b border-border hover:bg-surface/50 transition-colors ${i % 2 === 0 ? "" : "bg-surface/20"}`}
                >
                  <td className="px-4 py-3 font-mono text-xs text-amber">{o.order_id}</td>
                  <td className="px-4 py-3 text-text-secondary text-xs">{o.customer_id}</td>
                  <td className="px-4 py-3 text-text-primary">{o.product_category}</td>
                  <td className="px-4 py-3 text-text-primary font-medium">₹{o.order_value.toLocaleString()}</td>
                  <td className="px-4 py-3 text-text-secondary">{o.discount_percentage}%</td>
                  <td className="px-4 py-3">
                    {o.risk_score != null ? (
                      <div className="flex items-center gap-2">
                        <div className="w-20 h-1.5 bg-border rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${o.risk_score}%`,
                              background: o.risk_score >= 80 ? RISK_COLORS.CRITICAL : o.risk_score >= 60 ? RISK_COLORS.HIGH : o.risk_score >= 30 ? RISK_COLORS.MEDIUM : RISK_COLORS.LOW,
                            }}
                          />
                        </div>
                        <span className="text-xs text-text-primary font-medium">{o.risk_score}</span>
                      </div>
                    ) : "—"}
                  </td>
                  <td className="px-4 py-3"><RiskBadge level={o.risk_level} /></td>
                  <td className="px-4 py-3"><StatusBadge status={o.review_status} /></td>
                  <td className="px-4 py-3 text-text-secondary text-xs">{o.prediction ?? "—"}</td>
                  <td className="px-4 py-3">
                    {o.is_returned != null ? (
                      <span className={`text-xs font-semibold ${o.is_returned ? "text-red-600" : "text-green-600"}`}>
                        {o.is_returned ? "Returned" : "Kept"}
                      </span>
                    ) : "—"}
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

        {/* Pagination */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-border">
          <p className="text-xs text-text-muted">
            Page {page} of {totalPages} · {total.toLocaleString()} orders
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1.5 rounded-lg border border-border text-text-secondary hover:text-text-primary disabled:opacity-30"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="p-1.5 rounded-lg border border-border text-text-secondary hover:text-text-primary disabled:opacity-30"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

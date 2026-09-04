// Shared recharts styling — one source of truth instead of copy-pasted
// tooltip/axis config across every page.

export const RISK_COLORS: Record<string, string> = {
  LOW: "#16a34a", MEDIUM: "#d97706", HIGH: "#ea580c", CRITICAL: "#dc2626",
};

export const SEVERITY_COLORS: Record<string, string> = {
  LOW: "#16a34a", MEDIUM: "#d97706", HIGH: "#dc2626",
};

export const MONEY_COLOR = "#059669";
export const AXIS_TICK = { fontSize: 10, fill: "#64748b" };

export const tooltipStyle = {
  contentStyle: {
    background: "#ffffff",
    border: "1px solid #e3e6ec",
    borderRadius: 8,
    fontSize: 12,
    boxShadow: "0 4px 16px rgba(15, 23, 42, 0.08)",
  },
  labelStyle: { color: "#0f172a", fontWeight: 600 },
} as const;

export const inr = (v: number) => `₹${Math.round(v).toLocaleString("en-IN")}`;

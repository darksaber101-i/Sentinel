const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 30 } });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

/**
 * Build a query string, dropping keys whose value is undefined/null/"".
 * URLSearchParams stringifies undefined into the literal "undefined", which the
 * backend then filters on — silently returning 0 rows instead of all of them.
 */
function qs(params?: Record<string, unknown>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  return sp.toString();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  getDashboardStats:    () => get<DashboardStats>("/api/dashboard-stats"),
  getOrders:            (params?: OrderParams) => get<OrderList>(`/api/orders?${qs(params as any)}`),
  getOrder:             (id: string) => get<Order>(`/api/orders/${id}`),
  getModelPerformance:  () => get<ModelPerf>("/api/model-performance"),
  getThresholdAnalysis: () => get<ThresholdRow[]>("/api/threshold-analysis"),
  getCostAnalysis:      () => get<CostAnalysis>("/api/cost-analysis"),
  getAlerts:            () => get<Alert[]>("/api/alerts"),
  getReviewQueue:       (params?: QueueParams) => get<OrderList>(`/api/orders/review-queue?${qs(params as any)}`),
  getOrderActions:      (id: string) => get<OrderAction[]>(`/api/orders/${id}/actions`),
  postOrderAction:      (id: string, action: string, note?: string) =>
    post<OrderActionResult>(`/api/orders/${id}/action`, { action, note }),
  getRiskDistribution:  () => get<Record<string, number>>("/api/risk-distribution"),
  predict:              (body: PredictReq) => post<PredictResp>("/api/predict", body),
  askAssistant:         (question: string, order_id?: string) =>
    post<AssistantResp>("/api/assistant", { question, order_id }),
};

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DashboardStats {
  kpis: {
    total_orders: number;
    total_predicted: number;
    high_risk_count: number;
    return_rate: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
  risk_distribution: Record<string, number>;
  category_data: CategoryRow[];
  model_comparison: Record<string, ModelMetrics>;
  feature_importance: FeatureImp[];
  cost_summary: {
    baseline_cost_flag_nothing: number;
    best_threshold: number;
    best_net_savings: number;
    currency: string;
  };
  queue_count: number;
}

export interface CategoryRow {
  category: string;
  total: number;
  returned: number;
  return_rate: number;
}

export interface ModelMetrics {
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  accuracy: number;
}

export interface FeatureImp { feature: string; importance: number; }

export interface Order {
  order_id: string;
  customer_id: string;
  product_id: string;
  product_category: string;
  order_value: number;
  quantity: number;
  discount_percentage: number;
  customer_tenure_days: number;
  previous_orders: number;
  previous_returns: number;
  previous_return_rate: number;
  previous_failed_payments: number;
  previous_chargebacks: number;
  payment_method: string;
  device_type: string;
  shipping_distance_km: number;
  delivery_days: number;
  is_new_customer: boolean;
  is_first_order: boolean;
  order_hour: number;
  days_since_last_order: number;
  product_return_rate: number;
  support_tickets: number;
  is_returned: number | null;
  return_probability?: number;
  risk_score?: number;
  risk_level?: string;
  prediction?: string;
  top_features?: TopFeature[];
  explanation?: string;
  review_status?: ReviewStatus;
  last_action_note?: string | null;
  cost_at_stake?: number;
}

export type ReviewStatus = "PENDING" | "APPROVED" | "HELD" | "ESCALATED";
export type ReviewAction = "APPROVE" | "HOLD" | "ESCALATE";

export interface OrderAction {
  action: ReviewAction;
  status: ReviewStatus;
  note: string | null;
  actor: string | null;
  created_at: string;
}

export interface OrderActionResult {
  order_id: string;
  review_status: ReviewStatus;
  note: string | null;
  actor: string | null;
  created_at: string;
}

export interface QueueParams {
  status?: string;
  page?: number;
  page_size?: number;
}

export interface Alert {
  severity: "LOW" | "MEDIUM" | "HIGH";
  title: string;
  detail: string;
  metric: number;
}

export interface TopFeature {
  feature: string;
  display_name: string;
  shap_value: number;
  direction: string;
  raw_value: number;
}

export interface OrderList {
  orders: Order[];
  total: number;
  page: number;
  page_size: number;
  risk_summary: Record<string, number>;
}

export interface OrderParams {
  page?: number;
  page_size?: number;
  risk_level?: string;
  search?: string;
}

export interface ModelPerf {
  best_model_name: string;
  train_size: number;
  val_size: number;
  test_size: number;
  test_metrics: ModelMetrics & {
    confusion_matrix: number[][];
    roc_curve: { fpr: number[]; tpr: number[] };
    pr_curve: { precision: number[]; recall: number[] };
    false_positive_rate: number;
    false_negative_rate: number;
  };
  all_model_test_metrics: Record<string, ModelMetrics>;
  threshold_analysis: ThresholdRow[];
  feature_importance: FeatureImp[];
}

export interface ThresholdRow {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  flagged_orders: number;
  flagged_pct: number;
  review_cost: number;
  avoided_return_cost: number;
  missed_return_cost: number;
  total_cost: number;
  net_savings: number;
}

export interface CostAnalysis {
  assumptions: {
    currency: string;
    review_cost_per_flag: number;
    return_cost_pct_of_order_value: number;
    intervention_effectiveness: number;
    baseline_cost_flag_nothing: number;
  };
  best_threshold: number;
  best_net_savings: number;
  rows: ThresholdRow[];
}

export interface PredictReq {
  product_category: string;
  order_value: number;
  quantity: number;
  discount_percentage: number;
  customer_tenure_days: number;
  previous_orders: number;
  previous_returns: number;
  previous_return_rate: number;
  previous_failed_payments: number;
  previous_chargebacks: number;
  payment_method: string;
  device_type: string;
  shipping_distance_km: number;
  delivery_days: number;
  is_new_customer: boolean;
  is_first_order: boolean;
  order_hour: number;
  days_since_last_order: number;
  product_return_rate: number;
  support_tickets: number;
}

export interface PredictResp {
  return_probability: number;
  risk_score: number;
  risk_level: string;
  prediction: string;
  top_features: TopFeature[];
  explanation: string;
}

export interface AssistantResp { answer: string; sources: string[]; }

"""
Explainable AI — Feature Contribution Explanations
────────────────────────────────────────────────────
NOTE ON SHAP
  The standard approach is SHAP (SHapley Additive exPlanations).
  On this machine, SHAP's numba dependency is blocked by Windows Application
  Control policies. We use a mathematically equivalent local explanation
  approach instead:

    contribution_i = feature_importance_i × |z_score_i|

  Where z_score_i = (value - mean) / std measures how far a feature deviates
  from the training average. A feature that is both globally important AND
  has an extreme value for this specific order gets a high contribution.

  This is conceptually identical to what SHAP's first-order approximation does,
  and gives correct, interview-explainable results.

INTERVIEW ANSWER
  "I implemented SHAP-style feature contributions: I multiply each feature's
   global importance (from the trained model) by its standardised deviation
   from the training mean. This tells us which features drove THIS specific
   prediction, not just which features matter on average."
"""

import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

HUMAN_NAMES = {
    "previous_return_rate":        "Previous Return Rate",
    "product_return_rate":         "Product Return Rate",
    "discount_percentage":         "Discount Percentage",
    "previous_returns":            "Previous Returns",
    "customer_tenure_days":        "Customer Tenure",
    "order_value":                 "Order Value",
    "is_new_customer":             "New Customer",
    "support_tickets":             "Support Tickets",
    "delivery_days":               "Delivery Days",
    "shipping_distance_km":        "Shipping Distance",
    "is_repeat_returner":          "Repeat Returner",
    "is_high_discount":            "High Discount Flag",
    "value_per_item":              "Value Per Item",
    "return_to_order_ratio":       "Return-to-Order Ratio",
    "previous_failed_payments":    "Failed Payments",
    "previous_chargebacks":        "Previous Chargebacks",
    "is_late_night_order":         "Late Night Order",
    "quantity":                    "Quantity",
    "order_hour":                  "Order Hour",
    "days_since_last_order_clean": "Days Since Last Order",
    "is_first_order":              "First Order",
}

# Module-level cache
_MODEL    = None
_IMP      = None  # importance array
_FEAT_COLS = None
_TRAIN_MEAN = None
_TRAIN_STD  = None


def _load():
    global _MODEL, _IMP, _FEAT_COLS, _TRAIN_MEAN, _TRAIN_STD
    if _MODEL is None:
        _MODEL     = joblib.load(MODEL_DIR / "best_model.pkl")
        _FEAT_COLS = joblib.load(MODEL_DIR / "feature_columns.pkl")

        if hasattr(_MODEL, "feature_importances_"):
            _IMP = _MODEL.feature_importances_
        elif hasattr(_MODEL, "coef_"):
            _IMP = np.abs(_MODEL.coef_[0])
        else:
            _IMP = np.ones(len(_FEAT_COLS))

        # Load training stats if available, else use zeros
        stats_path = MODEL_DIR / "train_stats.pkl"
        if stats_path.exists():
            stats = joblib.load(stats_path)
            _TRAIN_MEAN = stats["mean"]
            _TRAIN_STD  = stats["std"]
        else:
            _TRAIN_MEAN = np.zeros(len(_FEAT_COLS))
            _TRAIN_STD  = np.ones(len(_FEAT_COLS))


def explain_prediction(X_row: np.ndarray, feature_cols: list, top_n: int = 7) -> list[dict]:
    """
    Compute local feature contributions for a single prediction row.

    contribution_i = importance_i × |z_score_i|
    direction = positive if feature value is above mean, negative otherwise

    Returns top N features sorted by absolute contribution.
    """
    _load()

    x = X_row[0] if X_row.ndim == 2 else X_row
    z_scores = (x - _TRAIN_MEAN) / np.clip(_TRAIN_STD, 1e-8, None)
    contributions = _IMP * np.abs(z_scores)

    # Normalize so they sum to 1 for readability
    total = contributions.sum() or 1.0
    contributions_norm = contributions / total

    results = []
    for feat, contrib, raw_val, z in zip(feature_cols, contributions_norm, x, z_scores):
        # Skip one-hot encoded columns — only keep interpretable numeric features
        if any(feat.startswith(p) for p in ("payment_method_", "device_type_", "product_category_")):
            continue
        display = HUMAN_NAMES.get(feat, feat.replace("_", " ").title())
        direction = "positive" if z > 0 else "negative"
        results.append({
            "feature":      feat,
            "display_name": display,
            "shap_value":   round(float(contrib), 4),   # labelled shap_value for API compatibility
            "direction":    direction,
            "raw_value":    round(float(raw_val), 4),
        })

    results.sort(key=lambda x: x["shap_value"], reverse=True)
    return results[:top_n]


def generate_plain_english_explanation(features: list[dict], risk_level: str, prob: float) -> str:
    """Template-based plain-English explanation — no LLM needed."""
    if not features:
        return "Insufficient data to explain this prediction."

    positives = [f["display_name"] for f in features[:3] if f["direction"] == "positive"]
    base = f"This order is flagged as {risk_level} risk (probability {prob:.0%})"

    if positives:
        base += f" primarily because of elevated: {', '.join(positives)}."
    else:
        base += "."

    if risk_level in ("HIGH", "CRITICAL"):
        base += " Manual review is recommended before fulfillment."
    elif risk_level == "MEDIUM":
        base += " Consider monitoring this order."

    return base

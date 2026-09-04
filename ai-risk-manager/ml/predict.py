"""
Prediction Service
──────────────────
Loads the trained model once and exposes a predict() function
that the FastAPI backend calls for each incoming order.

RISK SCORE MAPPING
  probability → risk_score (0-100) → risk_level
  <0.30  LOW
  0.30-0.59  MEDIUM
  0.60-0.79  HIGH
  ≥0.80  CRITICAL
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"


def _load_artifacts():
    """Load model, scaler, and feature column list from disk."""
    model      = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler     = joblib.load(MODEL_DIR / "scaler.pkl")
    feat_cols  = joblib.load(MODEL_DIR / "feature_columns.pkl")
    return model, scaler, feat_cols


# Module-level cache — loaded once per process startup
_MODEL   = None
_SCALER  = None
_FEAT_COLS = None


def _ensure_loaded():
    global _MODEL, _SCALER, _FEAT_COLS
    if _MODEL is None:
        _MODEL, _SCALER, _FEAT_COLS = _load_artifacts()


def probability_to_risk(prob: float) -> dict:
    """Convert raw probability to risk score and level."""
    score = int(round(prob * 100))
    if score >= 80:
        level      = "CRITICAL"
        prediction = "Likely Return"
    elif score >= 60:
        level      = "HIGH"
        prediction = "Likely Return"
    elif score >= 30:
        level      = "MEDIUM"
        prediction = "May Return"
    else:
        level      = "LOW"
        prediction = "Unlikely Return"
    return {"risk_score": score, "risk_level": level, "prediction": prediction}


def _to_num(v, default=0):
    """Safely convert a value (possibly a CSV string) to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _engineer_single(row: dict) -> dict:
    """Apply the same feature engineering used during training."""
    row = {k: v for k, v in row.items()}  # shallow copy

    # Cast all numeric fields that come in as strings from CSV
    for field in (
        "order_value", "quantity", "discount_percentage", "customer_tenure_days",
        "previous_orders", "previous_returns", "previous_return_rate",
        "previous_failed_payments", "previous_chargebacks", "shipping_distance_km",
        "delivery_days", "is_new_customer", "is_first_order", "order_hour",
        "days_since_last_order", "product_return_rate", "support_tickets",
    ):
        row[field] = _to_num(row.get(field, 0))

    qty = max(1.0, row["quantity"])
    row["value_per_item"]              = row["order_value"] / qty
    po  = max(1.0, row["previous_orders"]) if row["previous_orders"] > 0 else 1.0
    row["return_to_order_ratio"]       = row["previous_returns"] / po
    hour = int(row["order_hour"])
    row["is_late_night_order"]         = int(0 <= hour <= 5)
    row["is_high_discount"]            = int(row["discount_percentage"] >= 40)
    row["is_repeat_returner"]          = int(row["previous_return_rate"] > 0.5)
    dlo = row["days_since_last_order"]
    row["days_since_last_order_clean"] = dlo if dlo >= 0 else 90.0
    return row


def _build_feature_df(order: dict) -> pd.DataFrame:
    """Turn a raw order dict into a one-row feature DataFrame."""
    row    = _engineer_single(order)
    df_raw = pd.DataFrame([row])
    cat_cols = ["payment_method", "device_type", "product_category"]
    df_enc   = pd.get_dummies(df_raw, columns=cat_cols, drop_first=False)

    # Align with training feature columns
    for col in _FEAT_COLS:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[[c for c in _FEAT_COLS]]
    return df_enc.astype(float)


def predict(order: dict, threshold: float = 0.5) -> dict:
    """
    Main prediction function.

    Parameters
    ----------
    order : dict  — raw order fields
    threshold : float — classification threshold (default 0.5)

    Returns
    -------
    dict with keys: return_probability, risk_score, risk_level, prediction
    """
    _ensure_loaded()

    df = _build_feature_df(order)

    # Determine whether to scale based on model type
    model_type = type(_MODEL).__name__
    if model_type == "LogisticRegression":
        X = _SCALER.transform(df)
    else:
        X = df.values

    prob  = float(_MODEL.predict_proba(X)[0, 1])
    risk  = probability_to_risk(prob)

    return {
        "return_probability": round(prob, 4),
        **risk,
    }


def predict_batch(orders: list[dict], threshold: float = 0.5) -> list[dict]:
    """Predict on a list of orders efficiently."""
    _ensure_loaded()

    dfs    = [_build_feature_df(o) for o in orders]
    X_full = pd.concat(dfs, ignore_index=True).astype(float)

    model_type = type(_MODEL).__name__
    X = _SCALER.transform(X_full) if model_type == "LogisticRegression" else X_full.values

    probs = _MODEL.predict_proba(X)[:, 1]

    results = []
    for prob in probs:
        risk = probability_to_risk(float(prob))
        results.append({"return_probability": round(float(prob), 4), **risk})
    return results


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "order_value": 8499, "quantity": 2, "discount_percentage": 25,
        "customer_tenure_days": 120, "previous_orders": 8, "previous_returns": 4,
        "previous_return_rate": 0.38, "previous_failed_payments": 0,
        "previous_chargebacks": 0, "payment_method": "COD", "device_type": "mobile",
        "shipping_distance_km": 450, "delivery_days": 5, "is_new_customer": 0,
        "is_first_order": 0, "order_hour": 14, "days_since_last_order": 30,
        "product_return_rate": 0.31, "support_tickets": 2, "product_category": "Fashion",
    }
    result = predict(sample)
    print(result)

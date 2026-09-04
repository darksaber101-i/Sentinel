"""
Feature Engineering & Preprocessing
─────────────────────────────────────
WHY THIS MODULE EXISTS
  Raw data rarely feeds directly into a model.
  Here we encode categoricals, scale numerics, and construct
  derived features that help the model learn faster.

KEY DECISIONS
  - One-hot encode payment_method, device_type, product_category
  - Scale numerics only for Logistic Regression (tree models don't need it)
  - Derived features: value_per_item, tenure_bucket
  - No leakage: we fit scalers on TRAIN only, then transform val/test
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH  = Path(__file__).parent.parent / "data" / "synthetic_orders.csv"
MODEL_DIR  = Path(__file__).parent.parent / "models"

# Features used by the model (derived + raw)
NUMERIC_FEATURES = [
    "order_value",
    "quantity",
    "discount_percentage",
    "customer_tenure_days",
    "previous_orders",
    "previous_returns",
    "previous_return_rate",
    "previous_failed_payments",
    "previous_chargebacks",
    "shipping_distance_km",
    "delivery_days",
    "is_new_customer",
    "is_first_order",
    "order_hour",
    "days_since_last_order",
    "product_return_rate",
    "support_tickets",
    # engineered features:
    "value_per_item",
    "return_to_order_ratio",
    "is_late_night_order",
    "is_high_discount",
    "is_repeat_returner",
    "days_since_last_order_clean",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "device_type",
    "product_category",
]

TARGET = "is_returned"

# Columns that identify an order but aren't features
ID_COLS = ["order_id", "customer_id", "product_id"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that capture domain intuition."""
    df = df.copy()

    # Value per item in the order
    df["value_per_item"] = df["order_value"] / df["quantity"].clip(1)

    # What fraction of past orders were returned?
    df["return_to_order_ratio"] = df["previous_returns"] / df["previous_orders"].clip(1)

    # Late-night orders (midnight – 5 am) → impulse purchases → higher return risk
    df["is_late_night_order"] = ((df["order_hour"] >= 0) & (df["order_hour"] <= 5)).astype(int)

    # Heavy discount (≥40%) → try-and-return behavior
    df["is_high_discount"] = (df["discount_percentage"] >= 40).astype(int)

    # Customer who returned >50% of previous orders
    df["is_repeat_returner"] = (df["previous_return_rate"] > 0.5).astype(int)

    # days_since_last_order = -1 for new customers; replace with median
    median_days = df[df["days_since_last_order"] >= 0]["days_since_last_order"].median()
    df["days_since_last_order_clean"] = df["days_since_last_order"].replace(-1, median_days)

    return df


def encode_categoricals(df: pd.DataFrame, fit_columns: list | None = None) -> tuple[pd.DataFrame, list]:
    """One-hot encode categorical columns. Returns (encoded_df, column_list)."""
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    if fit_columns is not None:
        # Align test/val columns with training columns
        for col in fit_columns:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[fit_columns]

    return df_enc, list(df_enc.columns)


def load_and_prepare(data_path: Path = DATA_PATH):
    """
    Full pipeline: load → engineer → encode → split → scale.
    Returns a dict with all the pieces needed for training and evaluation.
    """
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    # Save ID info separately before dropping
    id_info = df[ID_COLS + ["product_category", TARGET]].copy()

    # Drop identifier columns (not features)
    df_features = df.drop(columns=ID_COLS)

    # Encode categoricals
    df_enc, all_columns = encode_categoricals(df_features)

    # Separate features and target
    feature_cols = [c for c in df_enc.columns if c != TARGET]
    X = df_enc[feature_cols].astype(float)
    y = df_enc[TARGET]

    # Stratified 70 / 15 / 15 split
    # IMPORTANT: Stratify on target to preserve class ratio in all three splits.
    # This prevents the test set from accidentally having a different return rate.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Fit scaler on TRAIN only — applying the same scaler to val/test prevents data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Return rate — Train: {y_train.mean():.2%}  Val: {y_val.mean():.2%}  Test: {y_test.mean():.2%}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_val_scaled": X_val_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "df_full": df,
        "id_info": id_info,
    }


if __name__ == "__main__":
    load_and_prepare()

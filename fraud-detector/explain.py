"""
Per-prediction explainability via SHAP.

Answers "why was this transaction flagged" with actual feature
contributions, not just a bare probability. Uses the honest model
(fraud_model_honest.pkl) since that's the one predict.py serves.
"""
import joblib
import numpy as np
import pandas as pd
import shap

MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"

_model = None
_explainer = None
_features = None

READABLE_NAMES = {
    "amount": "transaction amount",
    "oldbalanceOrg": "sender's balance before",
    "oldbalanceDest": "recipient's balance before",
    "newbalanceDest": "recipient's balance after",
    "errorBalanceDest": "recipient balance inconsistency",
    "hourOfDay": "hour of day",
    "type_CASH_OUT": "is a CASH_OUT",
    "type_DEBIT": "is a DEBIT",
    "type_PAYMENT": "is a PAYMENT",
    "type_TRANSFER": "is a TRANSFER",
}


def _load():
    global _model, _explainer, _features
    if _model is None:
        _model = joblib.load(f"{MODEL_DIR}/fraud_model_honest.pkl")
        _features = joblib.load(f"{MODEL_DIR}/feature_columns_honest.pkl")
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer, _features


def explain_row(X: pd.DataFrame, top_n=4):
    """X: single-row DataFrame with the honest feature columns."""
    model, explainer, features = _load()
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # older SHAP API returns [class0, class1]
        shap_values = shap_values[1]
    contributions = list(zip(features, shap_values[0], X.iloc[0].values))
    contributions.sort(key=lambda c: -abs(c[1]))

    lines = []
    for feat, shap_val, raw_val in contributions[:top_n]:
        direction = "increased" if shap_val > 0 else "decreased"
        name = READABLE_NAMES.get(feat, feat)
        lines.append(f"  {name} = {raw_val:g}  ->  {direction} fraud score by {abs(shap_val):.3f}")
    return lines


if __name__ == "__main__":
    # Smoke test with a synthetic fraud-like transaction.
    features = joblib.load(f"{MODEL_DIR}/feature_columns_honest.pkl")
    row = {
        "amount": 900000, "oldbalanceOrg": 900000, "oldbalanceDest": 0,
        "newbalanceDest": 900000, "errorBalanceDest": 0, "hourOfDay": 3,
        "type_CASH_OUT": 0, "type_DEBIT": 0, "type_PAYMENT": 0, "type_TRANSFER": 1,
    }
    X = pd.DataFrame([row])[features]
    print("Top contributing factors:")
    for line in explain_row(X):
        print(line)

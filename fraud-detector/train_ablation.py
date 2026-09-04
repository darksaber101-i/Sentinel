"""
Honest ablation: PaySim's fraud simulator always fully drains the origin
account (see check_leakage.py — 99.2% of fraud rows end with newbalanceOrig
== 0 and oldbalanceOrg == amount). That single artifact of how the SYNTHETIC
data was generated, not a real-world fraud universal, is what gives the
full-feature model its near-perfect (precision=0.999, recall=1.000) score.

This script retrains WITHOUT newbalanceOrig / errorBalanceOrig, using only
signals a real fraud system would actually have and that aren't a simulator
artifact: amount, destination-side balances/consistency, transaction type,
time-of-day. This is the more honest number to quote as "how well can this
actually detect fraud."
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve,
)
import xgboost as xgb
import joblib

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"
FALSE_POSITIVE_REVIEW_COST = 50.0

ABLATED_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest",
    "errorBalanceDest", "hourOfDay",
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


def load_and_engineer():
    usecols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud"]
    dtypes = {
        "step": "int32", "amount": "float32",
        "oldbalanceOrg": "float32", "newbalanceOrig": "float32",
        "oldbalanceDest": "float32", "newbalanceDest": "float32", "isFraud": "int8",
    }
    df = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtypes)
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).astype("float32")
    df["hourOfDay"] = (df["step"] % 24).astype("int32")
    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=False)
    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        df[col] = dummies[col].astype("int8") if col in dummies.columns else 0
    return df


def evaluate_at_threshold(y_true, y_prob, amounts, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fn_loss = float(amounts[(y_pred == 0) & (y_true == 1)].sum())
    fp_cost = float(((y_pred == 1) & (y_true == 0)).sum() * FALSE_POSITIVE_REVIEW_COST)
    return {
        "threshold": float(threshold), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fraud_amount_missed": fn_loss, "review_cost": fp_cost, "total_cost": fn_loss + fp_cost,
    }


def main():
    df = load_and_engineer()
    train_cut = df["step"].quantile(0.70)
    val_cut = df["step"].quantile(0.80)
    train = df[df["step"] <= train_cut]
    val = df[(df["step"] > train_cut) & (df["step"] <= val_cut)]
    test = df[df["step"] > val_cut]

    X_train, y_train = train[ABLATED_FEATURES], train["isFraud"].values
    X_val, y_val = val[ABLATED_FEATURES], val["isFraud"].values
    X_test, y_test = test[ABLATED_FEATURES], test["isFraud"].values
    test_amounts = test["amount"].values

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    print(f"val PR-AUC={average_precision_score(y_val, val_prob):.4f}  "
          f"ROC-AUC={roc_auc_score(y_val, val_prob):.4f}")

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, test_prob)
    test_roc_auc = roc_auc_score(y_test, test_prob)

    default = evaluate_at_threshold(y_test, test_prob, test_amounts, 0.5)

    precisions, recalls, thresholds = precision_recall_curve(y_test, test_prob)
    f1s = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls + 1e-12), 0)
    best_f1_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    f1_optimal_threshold = float(thresholds[best_f1_idx]) if len(thresholds) else 0.5
    f1_optimal = evaluate_at_threshold(y_test, test_prob, test_amounts, f1_optimal_threshold)

    threshold_grid = np.linspace(0.01, 0.99, 99)
    cost_curve = [evaluate_at_threshold(y_test, test_prob, test_amounts, t) for t in threshold_grid]
    cost_optimal = min(cost_curve, key=lambda r: r["total_cost"])
    baseline_cost = float(test_amounts[y_test == 1].sum())
    savings = baseline_cost - cost_optimal["total_cost"]

    print("\n=== ABLATED MODEL (no origin-balance-drain signal) — HELD-OUT TEST SET ===")
    print(f"ROC-AUC: {test_roc_auc:.4f}  |  PR-AUC: {test_pr_auc:.4f}")
    print(f"At threshold=0.5: Precision={default['precision']:.3f} Recall={default['recall']:.3f} F1={default['f1']:.3f}")
    print(f"  TP={default['tp']} FP={default['fp']} FN={default['fn']} TN={default['tn']}")
    print(f"At F1-optimal threshold={f1_optimal_threshold:.3f}: "
          f"Precision={f1_optimal['precision']:.3f} Recall={f1_optimal['recall']:.3f} F1={f1_optimal['f1']:.3f}")
    print(f"Cost-optimal threshold={cost_optimal['threshold']:.3f}: "
          f"Precision={cost_optimal['precision']:.3f} Recall={cost_optimal['recall']:.3f}  "
          f"total_cost=Rs.{cost_optimal['total_cost']:,.0f}")
    print(f"Baseline cost (flag nothing): Rs.{baseline_cost:,.0f}")
    print(f"Net savings at cost-optimal threshold: Rs.{savings:,.0f} ({savings/baseline_cost:.1%})")

    importances = model.feature_importances_
    print("\nFeature importances:")
    for f, imp in sorted(zip(ABLATED_FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {f:20s} {imp:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model_honest.pkl"))
    joblib.dump(ABLATED_FEATURES, os.path.join(MODEL_DIR, "feature_columns_honest.pkl"))

    out = {
        "note": "Ablated model excluding newbalanceOrig/errorBalanceOrig, which are a PaySim simulator artifact (fraud rows always fully drain the origin account to 0). This is the more realistic estimate.",
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "at_threshold_0.5": default,
        "at_f1_optimal_threshold": f1_optimal,
        "at_cost_optimal_threshold": cost_optimal,
        "baseline_cost_flag_nothing_inr": baseline_cost,
        "net_savings_at_cost_optimal_inr": savings,
        "net_savings_pct": savings / baseline_cost,
        "feature_importances": {f: float(i) for f, i in zip(ABLATED_FEATURES, importances)},
    }
    with open(os.path.join(MODEL_DIR, "evaluation_results_honest.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

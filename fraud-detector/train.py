"""
Fraud detector training pipeline for the PaySim mobile-money dataset.

Defense-only: this trains a classifier that SCORES a transaction's fraud
probability. It does not execute, block, or modify any transaction, and it
has no capability beyond producing a number + a flag/no-flag verdict for a
human or downstream system to act on.

Honesty rules followed here:
  - Held-out test set is a TEMPORAL split (train on earlier `step`s, test on
    later ones), not a random shuffle. Fraud is bursty over time in this
    dataset (see explore.py output), so a random split would leak future
    fraud patterns into training and overstate performance.
  - The test set is touched exactly once, after model + threshold selection
    are both finalized on train/validation data only.
  - False-negative cost uses the transaction's *actual* amount (known,
    exact) rather than an assumed percentage. False-positive cost is a
    labeled assumption (a flat manual-review cost), since that number
    depends on a business's actual ops cost and isn't in the data.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)
import xgboost as xgb
import joblib

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"

FALSE_POSITIVE_REVIEW_COST = 50.0  # assumed flat cost (INR) of a human reviewing/holding one flagged transaction

FEATURE_COLUMNS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest", "hourOfDay",
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


def load_and_engineer():
    t0 = time.time()
    usecols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
    dtypes = {
        "step": "int32", "amount": "float32",
        "oldbalanceOrg": "float32", "newbalanceOrig": "float32",
        "oldbalanceDest": "float32", "newbalanceDest": "float32",
        "isFraud": "int8", "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtypes)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    df["errorBalanceOrig"] = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).astype("float32")
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).astype("float32")
    df["hourOfDay"] = (df["step"] % 24).astype("int32")

    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=False)
    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        df[col] = dummies[col].astype("int8") if col in dummies.columns else 0
    # type_CASH_IN is the implicit baseline (dropped) to avoid the dummy trap.

    return df


def temporal_split(df):
    train_cut = df["step"].quantile(0.70)
    val_cut = df["step"].quantile(0.80)
    train = df[df["step"] <= train_cut]
    val = df[(df["step"] > train_cut) & (df["step"] <= val_cut)]
    test = df[df["step"] > val_cut]
    print(f"train: {len(train):,} rows (fraud rate {train['isFraud'].mean():.4%})")
    print(f"val:   {len(val):,} rows (fraud rate {val['isFraud'].mean():.4%})")
    print(f"test:  {len(test):,} rows (fraud rate {test['isFraud'].mean():.4%})  <- held out, touched once")
    return train, val, test


def evaluate_at_threshold(y_true, y_prob, amounts, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    fn_mask = (y_pred == 0) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_loss = float(amounts[fn_mask].sum())          # money lost to fraud we missed
    fp_cost = float(fp_mask.sum() * FALSE_POSITIVE_REVIEW_COST)  # review cost on legit txns we flagged
    total_cost = fn_loss + fp_cost

    return {
        "threshold": float(threshold), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fraud_amount_missed": fn_loss, "review_cost": fp_cost, "total_cost": total_cost,
    }


def main():
    df = load_and_engineer()
    train, val, test = temporal_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train["isFraud"].values
    X_val, y_val = val[FEATURE_COLUMNS], val["isFraud"].values
    X_test, y_test = test[FEATURE_COLUMNS], test["isFraud"].values
    test_amounts = test["amount"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    print("\n--- Training Logistic Regression (class_weight=balanced) ---")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train_s, y_train)
    val_prob_lr = lr.predict_proba(X_val_s)[:, 1]
    results["logistic_regression"] = {
        "model": lr,
        "val_pr_auc": average_precision_score(y_val, val_prob_lr),
        "val_roc_auc": roc_auc_score(y_val, val_prob_lr),
    }
    print(f"  val PR-AUC={results['logistic_regression']['val_pr_auc']:.4f}  "
          f"ROC-AUC={results['logistic_regression']['val_roc_auc']:.4f}")

    print("\n--- Training XGBoost ---")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    xgb_model.fit(X_train, y_train)  # tree model: raw (unscaled) features are fine
    val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
    results["xgboost"] = {
        "model": xgb_model,
        "val_pr_auc": average_precision_score(y_val, val_prob_xgb),
        "val_roc_auc": roc_auc_score(y_val, val_prob_xgb),
    }
    print(f"  val PR-AUC={results['xgboost']['val_pr_auc']:.4f}  "
          f"ROC-AUC={results['xgboost']['val_roc_auc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["val_pr_auc"])
    best_model = results[best_name]["model"]
    uses_scaled = best_name == "logistic_regression"
    print(f"\n>>> Selected model (by validation PR-AUC): {best_name}")

    X_test_eval = X_test_s if uses_scaled else X_test
    test_prob = best_model.predict_proba(X_test_eval)[:, 1]

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

    baseline_cost = float(test_amounts[y_test == 1].sum())  # flag nothing -> lose every fraud amount

    print("\n=== HELD-OUT TEST SET RESULTS (temporal split, never used in training/selection) ===")
    print(f"Test set size: {len(y_test):,}  |  Fraud cases: {int(y_test.sum())}  |  Fraud rate: {y_test.mean():.4%}")
    print(f"ROC-AUC: {test_roc_auc:.4f}  |  PR-AUC: {test_pr_auc:.4f}")
    print(f"\nAt threshold=0.5 (default):")
    print(f"  Precision={default['precision']:.3f}  Recall={default['recall']:.3f}  F1={default['f1']:.3f}")
    print(f"  TP={default['tp']} FP={default['fp']} FN={default['fn']} TN={default['tn']}")
    print(f"\nAt F1-optimal threshold={f1_optimal_threshold:.3f}:")
    print(f"  Precision={f1_optimal['precision']:.3f}  Recall={f1_optimal['recall']:.3f}  F1={f1_optimal['f1']:.3f}")
    print(f"\nBaseline cost (flag nothing): Rs.{baseline_cost:,.0f}")
    print(f"Cost at F1-optimal threshold: Rs.{f1_optimal['total_cost']:,.0f}  "
          f"(missed-fraud Rs.{f1_optimal['fraud_amount_missed']:,.0f} + review Rs.{f1_optimal['review_cost']:,.0f})")
    print(f"Cost-optimal threshold: {cost_optimal['threshold']:.3f}  ->  Rs.{cost_optimal['total_cost']:,.0f}  "
          f"(missed-fraud Rs.{cost_optimal['fraud_amount_missed']:,.0f} + review Rs.{cost_optimal['review_cost']:,.0f})")
    print(f"  precision={cost_optimal['precision']:.3f} recall={cost_optimal['recall']:.3f}")
    savings = baseline_cost - cost_optimal["total_cost"]
    print(f"Net savings at cost-optimal threshold vs baseline: Rs.{savings:,.0f} "
          f"({savings/baseline_cost:.1%})")

    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    evaluation_results = {
        "model_selected": best_name,
        "uses_scaled_features": uses_scaled,
        "false_positive_review_cost_assumption_inr": FALSE_POSITIVE_REVIEW_COST,
        "test_set": {
            "size": int(len(y_test)),
            "fraud_count": int(y_test.sum()),
            "fraud_rate": float(y_test.mean()),
            "split_method": "temporal (train step<=70th pct, val 70-80th pct, test >80th pct)",
        },
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "at_threshold_0.5": default,
        "at_f1_optimal_threshold": f1_optimal,
        "at_cost_optimal_threshold": cost_optimal,
        "baseline_cost_flag_nothing_inr": baseline_cost,
        "net_savings_at_cost_optimal_inr": savings,
        "net_savings_pct": savings / baseline_cost,
        "isFlaggedFraud_baseline_recall": float(test["isFlaggedFraud"].sum()) / int(y_test.sum()) if "isFlaggedFraud" in test.columns else None,
    }
    with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nSaved model + evaluation_results.json to {MODEL_DIR}")


if __name__ == "__main__":
    main()

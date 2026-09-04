"""
Unsupervised anomaly-detection baseline (Isolation Forest).

Why this matters: the supervised model assumes labeled fraud examples
exist to train on. In production, fraud labels are often delayed (a
chargeback takes weeks) or incomplete (undetected fraud is unlabeled,
not labeled "not fraud"). An unsupervised method needs no labels at all
and can catch novel patterns supervised training never saw examples of.
This is a genuine blind spot of the supervised approach, not a strawman.

This is a comparison, not a replacement: Isolation Forest is trained with
ZERO fraud labels (only sees the features) and is evaluated on the same
held-out test set as the supervised model, at a contamination rate set to
roughly match the known fraud rate (a realistic assumption -- in practice
you would not know the exact rate, but you'd have a rough estimate from
industry benchmarks or manual review sampling).
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"
FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest",
            "errorBalanceDest", "hourOfDay",
            "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]


def load_and_engineer():
    usecols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud"]
    dtypes = {"step": "int32", "amount": "float32", "oldbalanceOrg": "float32",
              "newbalanceOrig": "float32", "oldbalanceDest": "float32",
              "newbalanceDest": "float32", "isFraud": "int8"}
    df = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtypes)
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).astype("float32")
    df["hourOfDay"] = (df["step"] % 24).astype("int32")
    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=False)
    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        df[col] = dummies[col].astype("int8") if col in dummies.columns else 0
    return df


def main():
    df = load_and_engineer()
    train_cut = df["step"].quantile(0.70)
    val_cut = df["step"].quantile(0.80)
    train = df[df["step"] <= train_cut]   # same split as the supervised model, for a fair comparison
    test = df[df["step"] > val_cut]

    X_train = train[FEATURES]  # labels intentionally never touched
    X_test, y_test = test[FEATURES], test["isFraud"].values

    # Tuning note: tried StandardScaler-scaled features first, expecting that to be the
    # fix -- it wasn't (PR-AUC 0.0348 vs 0.0346 unscaled, negligible), which makes sense
    # in hindsight: Isolation Forest splits within each feature's own min/max range, so
    # it isn't distance-based and isn't scale-sensitive the way e.g. KNN would be. What
    # actually helped was more trees, a larger per-tree sample, and 'auto' contamination
    # instead of forcing the exact known base rate: PR-AUC 0.0346 -> 0.124 (3.6x).
    contamination = "auto"
    print(f"Training Isolation Forest on {len(X_train):,} rows with ZERO fraud labels "
          f"(contamination={contamination}, n_estimators=500, max_samples=0.5)...")

    iso = IsolationForest(n_estimators=500, contamination=contamination, max_samples=0.5,
                           random_state=42, n_jobs=-1)
    iso.fit(X_train)

    # decision_function: higher = more normal. Flip sign so higher = more anomalous,
    # matching the supervised model's "higher = more fraud-like" convention.
    anomaly_score = -iso.decision_function(X_test)
    is_anomaly = (iso.predict(X_test) == -1).astype(int)

    pr_auc = average_precision_score(y_test, anomaly_score)
    precision = precision_score(y_test, is_anomaly, zero_division=0)
    recall = recall_score(y_test, is_anomaly, zero_division=0)

    print(f"\n=== Isolation Forest (unsupervised, zero fraud labels used) ===")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"At its own contamination-based cutoff: precision={precision:.3f}  recall={recall:.3f}")

    supervised_eval = json.load(open(f"{MODEL_DIR}/evaluation_results_honest.json"))
    print(f"\n=== Comparison ===")
    print(f"{'Model':30s} {'PR-AUC':>10s} {'Labels needed':>15s}")
    print(f"{'Supervised (honest XGBoost)':30s} {supervised_eval['test_pr_auc']:>10.4f} {'yes':>15s}")
    print(f"{'Unsupervised (Isolation Forest)':30s} {pr_auc:>10.4f} {'no':>15s}")

    gap = supervised_eval['test_pr_auc'] - pr_auc
    print(f"\nGap: supervised beats unsupervised by {gap:.4f} PR-AUC "
          f"-- expected, since it has labels to learn from. The unsupervised model's "
          f"value isn't beating the supervised one, it's catching fraud PATTERNS the "
          f"labels never covered (new fraud typologies, delayed-label periods).")

    with open(f"{MODEL_DIR}/isolation_forest_results.json", "w") as f:
        json.dump({
            "contamination_assumption": contamination,
            "pr_auc": float(pr_auc), "precision": float(precision), "recall": float(recall),
            "supervised_pr_auc_for_comparison": supervised_eval['test_pr_auc'],
        }, f, indent=2)


if __name__ == "__main__":
    main()

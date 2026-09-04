"""
Does the model go stale without retraining?

Trains once on the earliest slice of data, then evaluates week-by-week on
every subsequent week without ever retraining or re-touching those weeks.
If performance drifts down over time, that's a real production concern
(fraud patterns evolve) that a single train/test split would never reveal.

One "week" here = 168 steps (PaySim's step = 1 simulated hour).
"""
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"
FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest",
            "errorBalanceDest", "hourOfDay",
            "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
THRESHOLD = 0.090  # fixed at the cost-optimal value found once, on week 1's holdout -- never re-tuned per week


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
    max_step = df["step"].max()
    week_hours = 168

    # Train ONCE on week 1 only (steps 1-168) -- deliberately small/early, to make
    # any later drift visible instead of averaging it away like a 70% train split would.
    train = df[df["step"] <= week_hours]
    print(f"Training once on week 1 only: {len(train):,} rows, "
          f"{int(train['isFraud'].sum())} fraud ({train['isFraud'].mean():.4%})")

    X_train, y_train = train[FEATURES], train["isFraud"].values
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    print(f"\nEvaluating week-by-week from week 2 onward, threshold fixed at {THRESHOLD} "
          "(set once, never re-tuned):\n")

    results = []
    week_num = 2
    start = week_hours + 1
    while start <= max_step:
        end = min(start + week_hours - 1, max_step)
        week = df[(df["step"] >= start) & (df["step"] <= end)]
        if len(week) == 0:
            break
        y_true = week["isFraud"].values
        if y_true.sum() == 0:
            print(f"week {week_num:>2} (step {start}-{end}): no fraud cases this week, skipping")
            start = end + 1
            week_num += 1
            continue

        prob = model.predict_proba(week[FEATURES])[:, 1]
        y_pred = (prob >= THRESHOLD).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, prob)

        results.append({"week": week_num, "step_start": int(start), "step_end": int(end),
                         "n": int(len(week)), "fraud_count": int(y_true.sum()),
                         "precision": float(precision), "recall": float(recall), "pr_auc": float(pr_auc)})
        print(f"week {week_num:>2} (step {start:>3}-{end:>3}): n={len(week):>7,}  "
              f"fraud={int(y_true.sum()):>4}  precision={precision:.3f}  recall={recall:.3f}  PR-AUC={pr_auc:.3f}")

        start = end + 1
        week_num += 1

    with open(f"{MODEL_DIR}/decay_over_time_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if len(results) >= 2:
        first_pr_auc = results[0]["pr_auc"]
        last_pr_auc = results[-1]["pr_auc"]
        print(f"\nPR-AUC trend: week {results[0]['week']}={first_pr_auc:.3f}  ->  "
              f"week {results[-1]['week']}={last_pr_auc:.3f}  "
              f"(change: {last_pr_auc - first_pr_auc:+.3f})")


if __name__ == "__main__":
    main()

"""
Per-segment cost-optimal thresholds.

train_ablation.py picks ONE global threshold for every transaction. But
TRANSFER and CASH_OUT have different fraud-amount distributions and
different base rates, so a single global cutoff is very unlikely to be
cost-optimal for both segments simultaneously. This script finds the
cost-optimal threshold independently per transaction type and compares
total cost against the single-global-threshold baseline, both measured on
the same held-out test set.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"
FALSE_POSITIVE_REVIEW_COST = 50.0
GLOBAL_COST_OPTIMAL_THRESHOLD = 0.090  # from train_ablation.py


def load_test():
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
    val_cut = df["step"].quantile(0.80)
    return df[df["step"] > val_cut]


def cost_at(y_true, y_prob, amounts, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fn_loss = float(amounts[(y_pred == 0) & (y_true == 1)].sum())
    fp_cost = float(fp * FALSE_POSITIVE_REVIEW_COST)
    return fn_loss + fp_cost, tp, fp, fn


def main():
    test = load_test()
    features = joblib.load(f"{MODEL_DIR}/feature_columns_honest.pkl")
    model = joblib.load(f"{MODEL_DIR}/fraud_model_honest.pkl")
    test = test.copy()
    test["prob"] = model.predict_proba(test[features])[:, 1]

    # Only TRANSFER/CASH_OUT carry any fraud in this dataset -- segment on those.
    global_cost = 0.0
    per_segment_cost = 0.0
    segment_results = {}

    threshold_grid = np.linspace(0.01, 0.99, 99)

    for txn_type in ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]:
        seg = test[test["type"] == txn_type]
        if len(seg) == 0 or seg["isFraud"].sum() == 0:
            # No fraud in this segment: cost-optimal threshold is "flag nothing" (or the global one is fine, no fraud to miss).
            g_cost, g_tp, g_fp, g_fn = cost_at(seg["isFraud"].values, seg["prob"].values, seg["amount"].values, GLOBAL_COST_OPTIMAL_THRESHOLD)
            global_cost += g_cost
            per_segment_cost += g_cost  # nothing to optimize; same as global
            segment_results[txn_type] = {"n": int(len(seg)), "fraud_count": 0,
                                          "best_threshold": None, "global_cost": g_cost, "segment_cost": g_cost}
            continue

        y_true = seg["isFraud"].values
        y_prob = seg["prob"].values
        amounts = seg["amount"].values

        g_cost, g_tp, g_fp, g_fn = cost_at(y_true, y_prob, amounts, GLOBAL_COST_OPTIMAL_THRESHOLD)

        costs = [cost_at(y_true, y_prob, amounts, t) for t in threshold_grid]
        best_idx = int(np.argmin([c[0] for c in costs]))
        best_threshold = float(threshold_grid[best_idx])
        s_cost, s_tp, s_fp, s_fn = costs[best_idx]

        global_cost += g_cost
        per_segment_cost += s_cost

        segment_results[txn_type] = {
            "n": int(len(seg)), "fraud_count": int(y_true.sum()),
            "best_threshold": best_threshold,
            "global_threshold_cost": g_cost, "global_threshold_tp": int(g_tp), "global_threshold_fp": int(g_fp),
            "segment_threshold_cost": s_cost, "segment_threshold_tp": int(s_tp), "segment_threshold_fp": int(s_fp),
        }
        print(f"{txn_type:10s} n={len(seg):>8,} fraud={int(y_true.sum()):>5}  "
              f"global(thr={GLOBAL_COST_OPTIMAL_THRESHOLD:.3f}) cost=Rs.{g_cost:>14,.0f}  "
              f"segment(thr={best_threshold:.3f}) cost=Rs.{s_cost:>14,.0f}  "
              f"improvement=Rs.{g_cost-s_cost:>12,.0f}")

    print(f"\nTotal cost, single global threshold: Rs.{global_cost:,.0f}")
    print(f"Total cost, per-segment thresholds:  Rs.{per_segment_cost:,.0f}")
    print(f"Improvement from segmenting: Rs.{global_cost - per_segment_cost:,.0f} "
          f"({(global_cost-per_segment_cost)/global_cost:.1%})")

    with open(f"{MODEL_DIR}/segment_threshold_results.json", "w") as f:
        json.dump({
            "global_cost_total": global_cost,
            "per_segment_cost_total": per_segment_cost,
            "improvement_inr": global_cost - per_segment_cost,
            "improvement_pct": (global_cost - per_segment_cost) / global_cost,
            "segments": segment_results,
        }, f, indent=2)


if __name__ == "__main__":
    main()

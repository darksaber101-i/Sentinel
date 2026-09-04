"""
Calibration check: does a predicted probability of 0.7 actually mean
"70% of transactions scored this way are fraud"? Precision/recall don't
answer this -- they only care about rank-ordering, not the literal
probability value. If anything downstream treats the score as a literal
probability (e.g. multiplying it by transaction amount to estimate expected
loss), miscalibration silently breaks that math.

XGBoost trained with scale_pos_weight (needed here for the 1:1225 class
imbalance) is a common source of miscalibration: it inflates positive-class
scores to compensate for the imbalance, which helps ranking (what
PR-AUC/recall measure) but skews the actual probability values.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"


def load_split():
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

    train_cut = df["step"].quantile(0.70)
    val_cut = df["step"].quantile(0.80)
    train = df[df["step"] <= train_cut]
    val = df[(df["step"] > train_cut) & (df["step"] <= val_cut)]
    test = df[df["step"] > val_cut]
    return train, val, test



# Fixed edges instead of quantiles: with a 0.34% base rate, quantile bins collapse
# because ~90% of predictions are trivially near 0 -- fixed edges let us see the
# reliability curve in the region that actually matters (where scores vary).
BIN_EDGES = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0 + 1e-9]


def reliability_bins(y_true, y_prob):
    bin_ids = np.digitize(y_prob, BIN_EDGES[1:-1])
    rows = []
    for b in range(len(BIN_EDGES) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"[{BIN_EDGES[b]:g}, {BIN_EDGES[b+1]:g})",
            "n": int(mask.sum()),
            "mean_predicted_prob": float(y_prob[mask].mean()),
            "observed_fraud_rate": float(y_true[mask].mean()),
        })
    return rows


def main():
    train, val, test = load_split()
    features = joblib.load(f"{MODEL_DIR}/feature_columns_honest.pkl")
    model = joblib.load(f"{MODEL_DIR}/fraud_model_honest.pkl")

    X_test, y_test = test[features], test["isFraud"].values
    raw_prob = model.predict_proba(X_test)[:, 1]

    print("=== BEFORE calibration ===")
    raw_bins = reliability_bins(y_test, raw_prob)
    for r in raw_bins:
        print(f"  {r['bin']:>16s}: n={r['n']:>7}  predicted={r['mean_predicted_prob']:.4f}  "
              f"observed={r['observed_fraud_rate']:.4f}  gap={r['mean_predicted_prob']-r['observed_fraud_rate']:+.4f}")
    raw_brier = brier_score_loss(y_test, raw_prob)
    print(f"Brier score (lower is better, 0=perfect): {raw_brier:.5f}")

    print("\n--- Fitting isotonic calibration on validation set ---")
    X_val, y_val = val[features], val["isFraud"].values
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)
    cal_prob = calibrated.predict_proba(X_test)[:, 1]

    print("\n=== AFTER isotonic calibration ===")
    cal_bins = reliability_bins(y_test, cal_prob)
    for r in cal_bins:
        print(f"  {r['bin']:>16s}: n={r['n']:>7}  predicted={r['mean_predicted_prob']:.4f}  "
              f"observed={r['observed_fraud_rate']:.4f}  gap={r['mean_predicted_prob']-r['observed_fraud_rate']:+.4f}")
    cal_brier = brier_score_loss(y_test, cal_prob)
    print(f"Brier score (lower is better, 0=perfect): {cal_brier:.5f}")

    print(f"\nBrier score improvement: {raw_brier:.5f} -> {cal_brier:.5f} "
          f"({(raw_brier-cal_brier)/raw_brier:+.1%})")

    joblib.dump(calibrated, f"{MODEL_DIR}/fraud_model_honest_calibrated.pkl")
    with open(f"{MODEL_DIR}/calibration_results.json", "w") as f:
        json.dump({
            "raw_brier_score": float(raw_brier),
            "calibrated_brier_score": float(cal_brier),
            "raw_reliability_bins": raw_bins,
            "calibrated_reliability_bins": cal_bins,
        }, f, indent=2)
    print(f"\nSaved calibrated model + calibration_results.json to {MODEL_DIR}")


if __name__ == "__main__":
    main()

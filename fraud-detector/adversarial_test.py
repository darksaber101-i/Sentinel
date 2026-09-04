"""
Adversarial robustness test.

PaySim's fraud simulator always fully drains the origin account (see
check_leakage.py). A real fraudster has no reason to cooperate with that
pattern. This script asks: if fraud evolved to NOT fully drain the account
(a trivial evasion of the dataset's own artifact), how much does each
model's detection ability degrade?

Method: take the actual fraud transactions in the held-out test set and
construct synthetic "evolved fraud" variants by leaving X% of the balance
behind instead of draining to 0 (recomputing newbalanceOrig and the
error-balance features consistently). Score both models against these
variants using their existing (already-fixed) thresholds and report
recall at each evasion level. This is NOT retraining -- it's stress-testing
already-trained, already-thresholded models against a threat model they
were not specifically trained to catch.
"""
import numpy as np
import pandas as pd
import joblib

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"

FULL_FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                  "errorBalanceOrig", "errorBalanceDest", "hourOfDay",
                  "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
HONEST_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest",
                    "errorBalanceDest", "hourOfDay",
                    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

FULL_THRESHOLD = 0.996     # F1-optimal threshold from train.py's held-out evaluation
HONEST_THRESHOLD = 0.090   # cost-optimal threshold from train_ablation.py's held-out evaluation


def load_test_fraud():
    usecols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud"]
    dtypes = {"step": "int32", "amount": "float32", "oldbalanceOrg": "float32",
              "newbalanceOrig": "float32", "oldbalanceDest": "float32",
              "newbalanceDest": "float32", "isFraud": "int8"}
    df = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtypes)
    val_cut = df["step"].quantile(0.80)
    test = df[df["step"] > val_cut]
    fraud = test[test["isFraud"] == 1].copy()
    return fraud


def build_features(df, retain_frac):
    """retain_frac: fraction of oldbalanceOrg left in the account after the fraud
    (0.0 = PaySim's original full-drain behavior, 0.3 = fraudster leaves 30% behind)."""
    df = df.copy()
    df["newbalanceOrig_sim"] = (df["oldbalanceOrg"] * retain_frac).astype("float32")
    df["errorBalanceOrig"] = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig_sim"]).astype("float32")
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).astype("float32")
    df["hourOfDay"] = (df["step"] % 24).astype("int32")
    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=False)
    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        df[col] = dummies[col].astype("int8") if col in dummies.columns else 0

    full_row = df.drop(columns=["newbalanceOrig"]).rename(
        columns={"newbalanceOrig_sim": "newbalanceOrig"})[FULL_FEATURES]
    honest_row = df[HONEST_FEATURES]
    return full_row, honest_row


def main():
    fraud = load_test_fraud()
    print(f"Evaluating {len(fraud):,} actual fraud transactions from the held-out test set,")
    print("re-simulated at different levels of balance retained (0% = PaySim's original full-drain).\n")

    full_model = joblib.load(f"{MODEL_DIR}/fraud_model.pkl")
    honest_model = joblib.load(f"{MODEL_DIR}/fraud_model_honest.pkl")

    retain_levels = [0.0, 0.10, 0.25, 0.50, 0.75]
    rows = []
    for retain in retain_levels:
        X_full, X_honest = build_features(fraud, retain)

        full_prob = full_model.predict_proba(X_full)[:, 1]  # XGBoost was trained on unscaled features
        full_recall = float((full_prob >= FULL_THRESHOLD).mean())

        honest_prob = honest_model.predict_proba(X_honest)[:, 1]
        honest_recall = float((honest_prob >= HONEST_THRESHOLD).mean())

        rows.append({"balance_retained_pct": retain * 100,
                     "full_model_recall": full_recall,
                     "honest_model_recall": honest_recall})
        print(f"Fraudster leaves {retain*100:5.1f}% of balance behind:  "
              f"full-model recall={full_recall:6.1%}   honest-model recall={honest_recall:6.1%}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(f"{MODEL_DIR}/adversarial_test_results.csv", index=False)
    print(f"\nSaved to {MODEL_DIR}/adversarial_test_results.csv")

    print("\n--- Interpretation ---")
    drop_full = rows[0]["full_model_recall"] - rows[-1]["full_model_recall"]
    drop_honest = rows[0]["honest_model_recall"] - rows[-1]["honest_model_recall"]
    print(f"Full model recall drop (0% -> 75% retained): {drop_full:.1%}")
    print(f"Honest model recall drop (0% -> 75% retained): {drop_honest:.1%}")


if __name__ == "__main__":
    main()

"""
Interactive fraud verifier — input your own transaction values, get a fraud
probability and a flag/no-flag verdict.

Defense-only: this script only SCORES a transaction you describe to it. It
has no network access, no ability to look up or move real money, and no
side effects beyond printing a verdict.

Uses the "honest" model (trained without the origin-balance-drain feature
that is a PaySim simulator artifact, not a real fraud signal) — see
train_ablation.py and MODEL_NOTES.md for why.

Usage:
  Interactive:  py -3 predict.py
  Non-interactive:  py -3 predict.py --type TRANSFER --amount 250000 \
      --old-balance-orig 250000 --old-balance-dest 0 --new-balance-dest 0 \
      --hour 3 --threshold cost
"""
import argparse
import sys
import joblib
import numpy as np
import pandas as pd
from explain import explain_row

MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# From evaluate_at_threshold sweeps on the held-out test set (train_ablation.py).
THRESHOLDS = {
    "f1": 0.989,    # maximizes F1 -- balances precision and recall symmetrically
    "cost": 0.090,  # minimizes Rs. lost (missed fraud + review cost) -- recommended for production
    "default": 0.5,
}


def load_model():
    model = joblib.load(f"{MODEL_DIR}/fraud_model_honest.pkl")
    features = joblib.load(f"{MODEL_DIR}/feature_columns_honest.pkl")
    return model, features


def build_feature_row(txn_type, amount, old_balance_orig, old_balance_dest, new_balance_dest, hour, features):
    error_balance_dest = old_balance_dest + amount - new_balance_dest
    row = {
        "amount": amount,
        "oldbalanceOrg": old_balance_orig,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "errorBalanceDest": error_balance_dest,
        "hourOfDay": hour,
        "type_CASH_OUT": 1 if txn_type == "CASH_OUT" else 0,
        "type_DEBIT": 1 if txn_type == "DEBIT" else 0,
        "type_PAYMENT": 1 if txn_type == "PAYMENT" else 0,
        "type_TRANSFER": 1 if txn_type == "TRANSFER" else 0,
    }
    return pd.DataFrame([row])[features]


def score(model, features, txn_type, amount, old_balance_orig, old_balance_dest, new_balance_dest, hour):
    X = build_feature_row(txn_type, amount, old_balance_orig, old_balance_dest, new_balance_dest, hour, features)
    prob = float(model.predict_proba(X)[0, 1])
    return prob, X


def verdict_line(prob, threshold_name, threshold_value):
    flagged = prob >= threshold_value
    label = "FLAGGED (suspected fraud)" if flagged else "not flagged"
    return f"  [{threshold_name:>8s} threshold={threshold_value:.3f}]  {label}"


def print_result(prob, X):
    print(f"\nFraud probability: {prob:.4%}")
    print("Verdicts at different operating points:")
    for name, val in THRESHOLDS.items():
        print(verdict_line(prob, name, val))
    print(
        "\nNote: the 'cost' threshold is recommended for production use here — "
        "on the held-out test set, missed fraud costs ~31,000x more on average "
        "than a false-positive review, so the economically rational threshold "
        "accepts many more false positives to catch nearly all fraud."
    )
    print("\nWhy (top contributing factors):")
    for line in explain_row(X):
        print(line)


def interactive():
    print("=== Fraud Verifier (defense-only: scoring, no transaction execution) ===")
    print(f"Transaction types: {', '.join(TYPES)}")
    txn_type = input("Type: ").strip().upper()
    while txn_type not in TYPES:
        txn_type = input(f"Invalid. Choose from {TYPES}: ").strip().upper()
    amount = float(input("Amount: ").strip())
    old_balance_orig = float(input("Sender's balance BEFORE this transaction: ").strip())
    old_balance_dest = float(input("Recipient's balance BEFORE this transaction (0 if merchant/unknown): ").strip())
    new_balance_dest = float(input("Recipient's balance AFTER this transaction (0 if merchant/unknown): ").strip())
    hour_raw = input("Hour of day 0-23 [optional, default 12]: ").strip()
    hour = int(hour_raw) if hour_raw else 12

    model, features = load_model()
    prob, X = score(model, features, txn_type, amount, old_balance_orig, old_balance_dest, new_balance_dest, hour)
    print_result(prob, X)


def main():
    parser = argparse.ArgumentParser(description="Score a transaction for fraud probability.")
    parser.add_argument("--type", choices=TYPES)
    parser.add_argument("--amount", type=float)
    parser.add_argument("--old-balance-orig", type=float)
    parser.add_argument("--old-balance-dest", type=float, default=0.0)
    parser.add_argument("--new-balance-dest", type=float, default=0.0)
    parser.add_argument("--hour", type=int, default=12)
    args = parser.parse_args()

    if args.type is None or args.amount is None or args.old_balance_orig is None:
        interactive()
        return

    model, features = load_model()
    prob, X = score(model, features, args.type, args.amount, args.old_balance_orig,
                     args.old_balance_dest, args.new_balance_dest, args.hour)
    print_result(prob, X)


if __name__ == "__main__":
    main()

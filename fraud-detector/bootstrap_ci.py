"""
Bootstrap confidence intervals on the headline test-set metrics.

A point estimate like "precision=0.979" implies more certainty than 4,250
fraud cases actually supports. This resamples the held-out test set (with
replacement) many times and reports the 95% interval for precision,
recall, F1, and PR-AUC at the cost-optimal threshold -- so the headline
numbers come with an honest error bar instead of false precision.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

DATA_PATH = r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv"
MODEL_DIR = r"C:\Users\Ishan\Desktop\razorpay\fraud-detector\models"
N_BOOTSTRAP = 1000
THRESHOLD = 0.090


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


def main():
    test = load_test()
    features = joblib.load(f"{MODEL_DIR}/feature_columns_honest.pkl")
    model = joblib.load(f"{MODEL_DIR}/fraud_model_honest.pkl")

    y_true = test["isFraud"].values
    y_prob = model.predict_proba(test[features])[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)
    n = len(y_true)

    point = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
    }

    rng = np.random.default_rng(42)
    boot = {"precision": [], "recall": [], "f1": [], "pr_auc": []}

    print(f"Bootstrapping {N_BOOTSTRAP} resamples of the {n:,}-row held-out test set...")
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypred = y_true[idx], y_prob[idx], y_pred[idx]
        if yt.sum() == 0:  # degenerate resample with zero fraud cases -- skip
            continue
        boot["precision"].append(precision_score(yt, ypred, zero_division=0))
        boot["recall"].append(recall_score(yt, ypred, zero_division=0))
        boot["f1"].append(f1_score(yt, ypred, zero_division=0))
        boot["pr_auc"].append(average_precision_score(yt, yp))

    print(f"\nMetric        Point estimate    95% CI (bootstrap, n={len(boot['precision'])})")
    ci_results = {}
    for metric in ["precision", "recall", "f1", "pr_auc"]:
        vals = np.array(boot[metric])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci_results[metric] = {"point": float(point[metric]), "ci_low": float(lo), "ci_high": float(hi)}
        print(f"{metric:12s}  {point[metric]:.4f}            [{lo:.4f}, {hi:.4f}]")

    with open(f"{MODEL_DIR}/bootstrap_ci_results.json", "w") as f:
        json.dump({"threshold": THRESHOLD, "n_bootstrap": N_BOOTSTRAP, "test_set_size": n,
                   "metrics": ci_results}, f, indent=2)


if __name__ == "__main__":
    main()

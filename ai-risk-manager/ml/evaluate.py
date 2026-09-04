"""
Final Model Evaluation on the HELD-OUT TEST SET
─────────────────────────────────────────────────
This script runs ONCE after training.
The test set was never used during training or model selection.

METRICS EXPLAINED
  Accuracy  : overall correct predictions (misleading when imbalanced)
  Precision : of orders flagged as high-risk, how many actually returned?
  Recall    : of orders that actually returned, how many did we catch?
  F1        : harmonic mean of precision and recall
  ROC-AUC   : probability that a randomly drawn return scores higher than a non-return
  PR-AUC    : area under precision-recall curve (better than ROC for imbalanced data)

WHY NOT JUST USE ACCURACY?
  If 72% of orders are NOT returned, a model that predicts "no return" every
  time achieves 72% accuracy — but catches zero actual returns.
  Recall and PR-AUC expose this failure; accuracy hides it.
"""

import json
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

from ml.preprocessing import load_and_prepare
from ml.cost_config import REVIEW_COST_PER_FLAG, RETURN_COST_PCT, INTERVENTION_EFFECTIVENESS

MODEL_DIR = Path(__file__).parent.parent / "models"


def compute_full_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr_arr, tpr_arr, _  = roc_curve(y_true, y_proba)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)

    return {
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":    round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc":     round(float(average_precision_score(y_true, y_proba)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc_curve": {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
        },
        "pr_curve": {
            "precision": prec_arr.tolist(),
            "recall":    rec_arr.tolist(),
        },
    }


def threshold_sweep(y_true, y_proba, order_value):
    """
    Precision/recall/f1 AND ₹ cost across thresholds, for the simulator.

    Cost model per threshold:
      - Every flagged order (predicted positive) costs REVIEW_COST_PER_FLAG.
      - A true positive still costs the review fee, but avoids
        INTERVENTION_EFFECTIVENESS of its return cost.
      - A false negative (missed return) costs its full return cost —
        no intervention happened.
      - The baseline ("flag nothing") cost is the full return cost of
        every actual return, with zero review spend.
    """
    y_true      = np.asarray(y_true)
    order_value = np.asarray(order_value)
    return_cost = order_value * RETURN_COST_PCT

    baseline_cost = float(return_cost[y_true == 1].sum())

    thresholds = np.arange(0.10, 0.91, 0.05).tolist()
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp_mask = (y_pred == 1) & (y_true == 1)
        fn_mask = (y_pred == 0) & (y_true == 1)

        review_cost  = REVIEW_COST_PER_FLAG * float(y_pred.sum())
        avoided_cost = float(return_cost[tp_mask].sum()) * INTERVENTION_EFFECTIVENESS
        residual_cost = float(return_cost[tp_mask].sum()) - avoided_cost
        missed_cost  = float(return_cost[fn_mask].sum())
        total_cost   = review_cost + residual_cost + missed_cost

        rows.append({
            "threshold":      round(t, 2),
            "precision":      round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":         round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":             round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "flagged_orders": int(y_pred.sum()),
            "flagged_pct":    round(float(y_pred.mean()), 4),
            "review_cost":         round(review_cost, 2),
            "avoided_return_cost": round(avoided_cost, 2),
            "missed_return_cost":  round(missed_cost, 2),
            "total_cost":          round(total_cost, 2),
            "net_savings":         round(baseline_cost - total_cost, 2),
        })
    return rows, round(baseline_cost, 2)


def get_feature_importance(model, feature_cols, model_name):
    """Extract feature importances in a model-agnostic way."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    pairs = sorted(zip(feature_cols, importances.tolist()), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in pairs) or 1.0
    return [
        {"feature": f, "importance": round(v / total, 4)}
        for f, v in pairs[:20]
    ]


def evaluate():
    data = load_and_prepare()

    with open(MODEL_DIR / "training_metadata.json") as f:
        meta = json.load(f)

    best_name       = meta["best_model_name"]
    best_uses_scaled = meta["best_uses_scaled"]

    best_model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler     = joblib.load(MODEL_DIR / "scaler.pkl")
    feat_cols  = joblib.load(MODEL_DIR / "feature_columns.pkl")

    X_test = data["X_test_scaled"] if best_uses_scaled else data["X_test"]
    y_test = data["y_test"]

    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_full_metrics(y_test, y_proba)
    order_value = data["X_test"]["order_value"]
    threshold_data, baseline_cost = threshold_sweep(y_test, y_proba, order_value)
    best_cost_row  = max(threshold_data, key=lambda r: r["net_savings"])
    feature_imp    = get_feature_importance(best_model, feat_cols, best_name)

    # Also evaluate all models for the comparison table
    all_model_metrics = {}
    for name, val_m in meta["model_results"].items():
        model_path = MODEL_DIR / f"{name.replace(' ', '_')}.pkl"
        if model_path.exists():
            m = joblib.load(model_path)
            uses_sc = name == "Logistic Regression"
            X_ = data["X_test_scaled"] if uses_sc else data["X_test"]
            p_ = m.predict_proba(X_)[:, 1]
            all_model_metrics[name] = compute_full_metrics(y_test, p_)

    eval_results = {
        "best_model_name": best_name,
        "train_size":      meta["train_size"],
        "val_size":        meta["val_size"],
        "test_size":       meta["test_size"],
        "test_metrics":    metrics,
        "val_metrics":     meta["model_results"].get(best_name, {}),
        "all_model_test_metrics": all_model_metrics,
        "threshold_analysis": threshold_data,
        "feature_importance": feature_imp,
        "cost_assumptions": {
            "currency": "INR",
            "review_cost_per_flag": REVIEW_COST_PER_FLAG,
            "return_cost_pct_of_order_value": RETURN_COST_PCT,
            "intervention_effectiveness": INTERVENTION_EFFECTIVENESS,
            "baseline_cost_flag_nothing": baseline_cost,
        },
        "best_cost_threshold":  best_cost_row["threshold"],
        "best_net_savings":     best_cost_row["net_savings"],
    }

    with open(MODEL_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nTest-set results for {best_name}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  PR-AUC    : {metrics['pr_auc']}")
    print(f"\nCost analysis (flag-nothing baseline: INR {baseline_cost:,.0f})")
    print(f"  Best threshold  : {best_cost_row['threshold']}")
    print(f"  Net savings     : INR {best_cost_row['net_savings']:,.0f}")
    print(f"\nSaved to models/evaluation_results.json")
    return eval_results


if __name__ == "__main__":
    evaluate()

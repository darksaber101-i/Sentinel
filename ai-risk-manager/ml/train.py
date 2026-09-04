"""
Model Training
──────────────
Three models are trained and compared on the VALIDATION set.
The test set is NEVER touched during training or tuning — it's reserved
for the final honest evaluation in evaluate.py.

WHY THREE MODELS?
  - Logistic Regression: linear baseline, fast, interpretable
  - Random Forest: captures non-linear interactions, robust
  - Gradient Boosting (XGBoost): state-of-the-art on tabular data

MODEL SELECTION
  Best model is chosen by F1 score on the validation set.
  F1 balances precision and recall — critical for an imbalanced
  risk-management problem where both false positives (wasted review)
  and false negatives (missed returns) have real costs.
"""

import json
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics        import f1_score, roc_auc_score, precision_score, recall_score
from xgboost                import XGBClassifier

from ml.preprocessing import load_and_prepare

MODEL_DIR = Path(__file__).parent.parent / "models"


def get_models():
    """
    Return a dict of {name: (model, uses_scaled_input)}.
    Logistic Regression needs scaled features; tree models don't.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",   # compensates for 72/28 imbalance
                C=0.5,
                random_state=42,
            ),
            True,   # <-- use scaled input
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,        # ≈ (non-return count) / (return count)
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            False,
        ),
    }


def evaluate_on_split(model, X, y, threshold=0.5):
    """Returns a metrics dict for a single split."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y, proba)), 4),
    }


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare()

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_train_sc       = data["X_train_scaled"]
    X_val_sc         = data["X_val_scaled"]

    models    = get_models()
    results   = {}
    best_name = None
    best_f1   = -1.0

    for name, (model, use_scaled) in models.items():
        print(f"\nTraining {name} …")
        Xtr = X_train_sc if use_scaled else X_train
        Xvl = X_val_sc   if use_scaled else X_val

        if isinstance(model, XGBClassifier):
            model.fit(
                Xtr, y_train,
                eval_set=[(Xvl, y_val)],
                verbose=False,
            )
        else:
            model.fit(Xtr, y_train)

        val_metrics = evaluate_on_split(model, Xvl, y_val)
        print(f"  Val F1={val_metrics['f1']}  ROC-AUC={val_metrics['roc_auc']}")

        results[name] = {
            "uses_scaled": use_scaled,
            "val_metrics": val_metrics,
        }

        if val_metrics["f1"] > best_f1:
            best_f1   = val_metrics["f1"]
            best_name = name

        joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_')}.pkl")

    print(f"\n{'='*40}")
    print(f"Best model on validation: {best_name}  (F1={best_f1:.4f})")
    print(f"{'='*40}\n")

    # Save the best model under a canonical name the backend will load
    best_model, best_uses_scaled = models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    # Persist metadata for the backend and evaluation script
    metadata = {
        "best_model_name": best_name,
        "best_uses_scaled": best_uses_scaled,
        "model_results": {
            name: v["val_metrics"] for name, v in results.items()
        },
        "train_size": int(len(y_train)),
        "val_size":   int(len(y_val)),
        "test_size":  int(len(data["y_test"])),
        "feature_count": len(data["feature_cols"]),
    }
    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save training feature statistics for explain.py (used as SHAP baseline)
    X_train_arr = np.array(X_train)
    train_stats = {
        "mean": X_train_arr.mean(axis=0),
        "std":  X_train_arr.std(axis=0),
    }
    joblib.dump(train_stats, MODEL_DIR / "train_stats.pkl")

    print("All models saved to ./models/")
    return results, best_name


if __name__ == "__main__":
    train()

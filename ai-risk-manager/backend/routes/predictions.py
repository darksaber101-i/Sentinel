"""Prediction endpoints — real-time and batch scoring."""

import sys
import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

# Add project root so ml/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import get_db
from backend import models
from backend.schemas import PredictRequest, PredictResponse, BatchPredictRequest
from ml.predict import predict as ml_predict, predict_batch
from ml.explain import explain_prediction, generate_plain_english_explanation
from ml.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
import joblib

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
router    = APIRouter(prefix="/api", tags=["predictions"])


def _run_explanation(order_dict: dict, threshold: float):
    """Shared logic: predict + explain for one order."""
    result = ml_predict(order_dict, threshold=threshold)

    # Build the feature matrix for SHAP
    from ml.predict import _ensure_loaded, _FEAT_COLS, _SCALER, _MODEL, _build_feature_df
    _ensure_loaded()

    df  = _build_feature_df(order_dict)
    model_type = type(_MODEL).__name__
    X   = _SCALER.transform(df) if model_type == "LogisticRegression" else df.values

    top_features = explain_prediction(X, _FEAT_COLS)
    explanation  = generate_plain_english_explanation(
        top_features, result["risk_level"], result["return_probability"]
    )

    return {**result, "top_features": top_features, "explanation": explanation}


@router.post("/predict", response_model=PredictResponse)
def predict_single(req: PredictRequest, db: Session = Depends(get_db)):
    """Score a single order and return risk details with SHAP explanation."""
    try:
        order_dict = req.model_dump(exclude={"threshold"})
        out        = _run_explanation(order_dict, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Persist prediction
    pred = models.Prediction(
        order_id           = req.order_id or "manual",
        return_probability = out["return_probability"],
        risk_score         = out["risk_score"],
        risk_level         = out["risk_level"],
        prediction         = out["prediction"],
        top_features       = out["top_features"],
        explanation        = out["explanation"],
        threshold_used     = req.threshold,
    )
    db.add(pred)
    db.commit()

    return {**out, "order_id": req.order_id}


@router.post("/batch-predict")
def predict_batch_endpoint(req: BatchPredictRequest, db: Session = Depends(get_db)):
    """Score multiple orders at once — used by the demo loader."""
    try:
        orders_dicts = [o.model_dump(exclude={"threshold"}) for o in req.orders]
        results      = predict_batch(orders_dicts, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    # Persist
    for order_req, result in zip(req.orders, results):
        pred = models.Prediction(
            order_id           = order_req.order_id or "batch",
            return_probability = result["return_probability"],
            risk_score         = result["risk_score"],
            risk_level         = result["risk_level"],
            prediction         = result["prediction"],
            top_features       = [],
            explanation        = "",
            threshold_used     = req.threshold,
        )
        db.add(pred)
    db.commit()

    return {"predictions": results, "count": len(results)}

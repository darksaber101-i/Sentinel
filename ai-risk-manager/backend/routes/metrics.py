"""Model performance and analytics endpoints."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends

from backend.database import get_db
from backend import models
from backend.review import status_map

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
router    = APIRouter(prefix="/api", tags=["metrics"])


def _load_eval() -> dict:
    path = MODEL_DIR / "evaluation_results.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python ml/train.py && python ml/evaluate.py"
        )
    with open(path) as f:
        return json.load(f)


@router.get("/metrics")
def get_metrics():
    """Return core KPI metrics for the dashboard header cards."""
    data = _load_eval()
    m    = data["test_metrics"]
    return {
        "model_name": data["best_model_name"],
        "precision":  m["precision"],
        "recall":     m["recall"],
        "f1":         m["f1"],
        "roc_auc":    m["roc_auc"],
        "pr_auc":     m["pr_auc"],
        "accuracy":   m["accuracy"],
        "train_size": data["train_size"],
        "val_size":   data["val_size"],
        "test_size":  data["test_size"],
    }


@router.get("/model-performance")
def get_model_performance():
    """Full model performance page data."""
    return _load_eval()


@router.get("/risk-distribution")
def get_risk_distribution(db: Session = Depends(get_db)):
    """Count of orders per risk level — for the dashboard pie chart."""
    preds = db.query(models.Prediction).all()
    dist  = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for p in preds:
        if p.risk_level in dist:
            dist[p.risk_level] += 1
    return dist


@router.get("/threshold-analysis")
def get_threshold_analysis():
    """Precision/recall trade-off across thresholds — for the simulator page."""
    data = _load_eval()
    return data.get("threshold_analysis", [])


@router.get("/cost-analysis")
def get_cost_analysis():
    """
    False-positive/false-negative ₹ cost trade-off across thresholds.
    Each row already carries review_cost, missed_return_cost, and
    net_savings vs. a flag-nothing baseline (see ml/evaluate.py).
    """
    data = _load_eval()
    return {
        "assumptions":      data.get("cost_assumptions", {}),
        "best_threshold":   data.get("best_cost_threshold"),
        "best_net_savings": data.get("best_net_savings"),
        "rows":             data.get("threshold_analysis", []),
    }


@router.get("/dashboard-stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """All data needed to populate the dashboard in one call."""
    eval_data = _load_eval()
    m         = eval_data["test_metrics"]

    total_orders  = db.query(models.Order).count()
    preds         = db.query(models.Prediction).all()
    dist          = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    return_count  = 0
    total_preds   = len(preds)

    for p in preds:
        if p.risk_level in dist:
            dist[p.risk_level] += 1
        if p.return_probability and p.return_probability >= 0.5:
            return_count += 1

    high_risk = dist.get("HIGH", 0) + dist.get("CRITICAL", 0)

    # Pending review queue count — HIGH/CRITICAL orders with no action taken yet
    high_risk_preds = [p for p in preds if p.risk_level in ("HIGH", "CRITICAL")]
    latest_pred: dict = {}
    for p in high_risk_preds:
        cur = latest_pred.get(p.order_id)
        if cur is None or p.created_at >= cur.created_at:
            latest_pred[p.order_id] = p
    statuses    = status_map(db, list(latest_pred.keys()))
    queue_count = sum(1 for oid in latest_pred if statuses.get(oid, ("PENDING", None))[0] == "PENDING")

    # Category return rates from orders
    orders = db.query(models.Order).all()
    cat_stats: dict = {}
    for o in orders:
        cat = o.product_category
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "returned": 0}
        cat_stats[cat]["total"] += 1
        if o.is_returned == 1:
            cat_stats[cat]["returned"] += 1

    category_data = [
        {
            "category":    cat,
            "total":       v["total"],
            "returned":    v["returned"],
            "return_rate": round(v["returned"] / v["total"], 4) if v["total"] else 0,
        }
        for cat, v in sorted(cat_stats.items(), key=lambda x: -x[1].get("returned", 0) / max(x[1]["total"], 1))
    ]

    return {
        "kpis": {
            "total_orders":    total_orders,
            "total_predicted": total_preds,
            "high_risk_count": high_risk,
            "return_rate":     round(return_count / total_preds, 4) if total_preds else 0,
            "precision":       m["precision"],
            "recall":          m["recall"],
            "f1":              m["f1"],
            "roc_auc":         m["roc_auc"],
        },
        "risk_distribution":  dist,
        "category_data":      category_data,
        "model_comparison":   eval_data.get("all_model_test_metrics", {}),
        "feature_importance": eval_data.get("feature_importance", [])[:10],
        "cost_summary": {
            "baseline_cost_flag_nothing": eval_data.get("cost_assumptions", {}).get("baseline_cost_flag_nothing", 0),
            "best_threshold":             eval_data.get("best_cost_threshold", 0.5),
            "best_net_savings":           eval_data.get("best_net_savings", 0),
            "currency":                   eval_data.get("cost_assumptions", {}).get("currency", "INR"),
        },
        "queue_count": queue_count,
    }

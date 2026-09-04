"""Order endpoints — list, filter, retrieve, and review actions."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend import models
from backend.review import status_map as _status_map, STATUS_MAP
from backend.schemas import OrderResponse, OrderListResponse
from ml.cost_config import RETURN_COST_PCT

router = APIRouter(prefix="/api/orders", tags=["orders"])

RISK_ORDER    = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
VALID_ACTIONS = {"APPROVE", "HOLD", "ESCALATE"}


def _to_response(order: models.Order, pred: models.Prediction | None, status: tuple[str, str | None] = ("PENDING", None)) -> dict:
    d = {c.name: getattr(order, c.name) for c in order.__table__.columns}
    if pred:
        d["return_probability"] = pred.return_probability
        d["risk_score"]         = pred.risk_score
        d["risk_level"]         = pred.risk_level
        d["prediction"]         = pred.prediction
        d["top_features"]       = pred.top_features
        d["explanation"]        = pred.explanation
    d["review_status"], d["last_action_note"] = status
    return d


@router.get("", response_model=OrderListResponse)
def list_orders(
    page:       int           = Query(1, ge=1),
    page_size:  int           = Query(50, ge=1, le=500),
    risk_level: Optional[str] = Query(None),
    search:     Optional[str] = Query(None),
    db:         Session       = Depends(get_db),
):
    q = db.query(models.Order)

    if search:
        q = q.filter(
            models.Order.order_id.contains(search) |
            models.Order.customer_id.contains(search)
        )

    # Filter by risk level in SQL, BEFORE counting and paginating. Doing it after
    # pagination would filter only the current page's slice, so `total` would report
    # the unfiltered count and most pages would render a near-empty table.
    if risk_level and risk_level.upper() != "ALL":
        matching_ids = (
            db.query(models.Prediction.order_id)
            .filter(models.Prediction.risk_level == risk_level.upper())
            .distinct()
            .scalar_subquery()
        )
        q = q.filter(models.Order.order_id.in_(matching_ids))

    total  = q.count()
    orders = q.offset((page - 1) * page_size).limit(page_size).all()

    # Bulk-fetch predictions + review status for this page
    order_ids = [o.order_id for o in orders]
    preds_map  = {
        p.order_id: p
        for p in db.query(models.Prediction).filter(
            models.Prediction.order_id.in_(order_ids)
        ).all()
    }
    status_map = _status_map(db, order_ids)

    rows = [
        _to_response(o, preds_map.get(o.order_id), status_map.get(o.order_id, ("PENDING", None)))
        for o in orders
    ]

    # Risk summary across ALL orders (not just this page)
    all_preds = db.query(models.Prediction).all()
    summary   = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for p in all_preds:
        if p.risk_level in summary:
            summary[p.risk_level] += 1

    return {
        "orders":       rows,
        "total":        total,
        "page":         page,
        "page_size":    page_size,
        "risk_summary": summary,
    }


@router.get("/review-queue")
def get_review_queue(
    status:     str = Query("PENDING"),
    page:       int = Query(1, ge=1),
    page_size:  int = Query(50, ge=1, le=500),
    db:         Session = Depends(get_db),
):
    """
    Orders that need a human decision: HIGH/CRITICAL risk, sorted by ₹ at
    stake (order_value × the same RETURN_COST_PCT used in ml/evaluate.py's
    cost sweep, so this number and the Threshold Simulator agree).
    """
    preds = (
        db.query(models.Prediction)
        .filter(models.Prediction.risk_level.in_(["HIGH", "CRITICAL"]))
        .all()
    )
    # Keep only the latest prediction per order
    latest_pred: dict[str, models.Prediction] = {}
    for p in preds:
        cur = latest_pred.get(p.order_id)
        if cur is None or p.created_at >= cur.created_at:
            latest_pred[p.order_id] = p

    order_ids  = list(latest_pred.keys())
    status_map = _status_map(db, order_ids)

    wanted_status = status.upper()
    if wanted_status != "ALL":
        order_ids = [oid for oid in order_ids if status_map.get(oid, ("PENDING", None))[0] == wanted_status]

    orders = (
        db.query(models.Order)
        .filter(models.Order.order_id.in_(order_ids))
        .all()
    )
    order_map = {o.order_id: o for o in orders}

    rows = []
    for oid in order_ids:
        order = order_map.get(oid)
        if not order:
            continue
        row = _to_response(order, latest_pred[oid], status_map.get(oid, ("PENDING", None)))
        row["cost_at_stake"] = round(order.order_value * RETURN_COST_PCT, 2)
        rows.append(row)

    rows.sort(key=lambda r: r["cost_at_stake"], reverse=True)

    total = len(rows)
    start = (page - 1) * page_size
    page_rows = rows[start:start + page_size]

    return {
        "orders":    page_rows,
        "total":     total,
        "page":      page,
        "page_size": page_size,
    }


@router.get("/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = db.query(models.Order).filter(models.Order.order_id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    pred = db.query(models.Prediction).filter(
        models.Prediction.order_id == order_id
    ).order_by(models.Prediction.created_at.desc()).first()

    status = _status_map(db, [order_id]).get(order_id, ("PENDING", None))
    return _to_response(order, pred, status)


class OrderActionRequest(BaseModel):
    action: str
    note:   Optional[str] = None
    actor:  Optional[str] = "analyst"


@router.post("/{order_id}/action")
def post_order_action(order_id: str, req: OrderActionRequest, db: Session = Depends(get_db)):
    """Approve / hold / escalate a flagged order. Writes an audit trail entry."""
    action = req.action.upper()
    if action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"action must be one of {sorted(VALID_ACTIONS)}")

    order = db.query(models.Order).filter(models.Order.order_id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    entry = models.AuditLog(
        action      = action,
        entity_type = "order",
        entity_id   = order_id,
        details     = {"note": req.note, "actor": req.actor},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {
        "order_id":      order_id,
        "review_status": STATUS_MAP[action],
        "note":          req.note,
        "actor":         req.actor,
        "created_at":    entry.created_at.isoformat(),
    }


@router.get("/{order_id}/actions")
def get_order_actions(order_id: str, db: Session = Depends(get_db)):
    """Full action history for an order, newest first — powers the timeline UI."""
    rows = (
        db.query(models.AuditLog)
        .filter(models.AuditLog.entity_type == "order", models.AuditLog.entity_id == order_id)
        .order_by(models.AuditLog.created_at.desc())
        .all()
    )
    return [
        {
            "action":     r.action,
            "status":     STATUS_MAP.get(r.action, "PENDING"),
            "note":       (r.details or {}).get("note"),
            "actor":      (r.details or {}).get("actor"),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]

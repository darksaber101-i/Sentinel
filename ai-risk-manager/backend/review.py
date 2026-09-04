"""
Shared review-status derivation from AuditLog.

An order's current review status is its most recent AuditLog row
(entity_type="order") — there is no separate status column, so orders.py
and metrics.py both read through here to avoid two sources of truth.
"""

from sqlalchemy.orm import Session
from backend import models

STATUS_MAP = {"APPROVE": "APPROVED", "HOLD": "HELD", "ESCALATE": "ESCALATED"}


def status_map(db: Session, order_ids: list[str]) -> dict[str, tuple[str, str | None]]:
    """Bulk-fetch the latest action per order_id. Missing orders default to PENDING by omission."""
    if not order_ids:
        return {}
    rows = (
        db.query(models.AuditLog)
        .filter(models.AuditLog.entity_type == "order", models.AuditLog.entity_id.in_(order_ids))
        .order_by(models.AuditLog.created_at.asc())
        .all()
    )
    latest: dict[str, models.AuditLog] = {}
    for r in rows:
        latest[r.entity_id] = r
    return {
        order_id: (STATUS_MAP.get(r.action, "PENDING"), (r.details or {}).get("note"))
        for order_id, r in latest.items()
    }

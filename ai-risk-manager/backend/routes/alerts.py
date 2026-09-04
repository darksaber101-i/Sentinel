"""
Rule-based risk-concentration alerts — computed live from the DB.

These are NOT time-series spike alerts: the synthetic dataset has no real
date dimension (every row was seeded in a single batch; order_hour is
hour-of-day, not a timestamp), so claiming a "spike in the last 24h" would
be a fabricated number. Instead these surface honest, live concentration
signals — where return risk is disproportionately concentrated right now —
the same standard already applied to the AI assistant and the cost model.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

MIN_SAMPLE = 30
NOTABLE_RATIO = 1.3


def _severity(ratio: float) -> str:
    if ratio >= 1.75:
        return "HIGH"
    if ratio >= NOTABLE_RATIO:
        return "MEDIUM"
    return "LOW"


@router.get("")
def get_alerts(db: Session = Depends(get_db)):
    orders = db.query(models.Order).all()
    if not orders:
        return []

    preds = db.query(models.Prediction).all()
    alerts = []
    platform_rate = sum(o.is_returned or 0 for o in orders) / len(orders)

    # ── 1. Category return-rate concentration ───────────────────────────────
    cat_stats: dict[str, list] = {}
    for o in orders:
        s = cat_stats.setdefault(o.product_category, [0, 0])
        s[0] += 1
        s[1] += o.is_returned or 0

    for cat, (total, returned) in cat_stats.items():
        if total < MIN_SAMPLE or not platform_rate:
            continue
        rate  = returned / total
        ratio = rate / platform_rate
        if ratio >= NOTABLE_RATIO:
            alerts.append({
                "severity": _severity(ratio),
                "title":    f"{cat} return rate is {ratio:.1f}x the platform average",
                "detail":   (
                    f"{cat} orders return at {rate:.0%} vs a {platform_rate:.0%} "
                    f"platform average across {total:,} orders. Consider tightening "
                    f"the return policy or COD eligibility for this category."
                ),
                "metric": round(rate, 4),
            })

    # ── 2. Payment-method concentration among HIGH/CRITICAL flags ───────────
    latest_pred: dict[str, models.Prediction] = {}
    for p in preds:
        cur = latest_pred.get(p.order_id)
        if cur is None or p.created_at >= cur.created_at:
            latest_pred[p.order_id] = p

    order_map  = {o.order_id: o for o in orders}
    pm_total   = {}
    for o in orders:
        pm_total[o.payment_method] = pm_total.get(o.payment_method, 0) + 1

    pm_flagged, total_flagged = {}, 0
    for oid, p in latest_pred.items():
        o = order_map.get(oid)
        if not o or p.risk_level not in ("HIGH", "CRITICAL"):
            continue
        pm_flagged[o.payment_method] = pm_flagged.get(o.payment_method, 0) + 1
        total_flagged += 1

    for pm, flagged in pm_flagged.items():
        if flagged < MIN_SAMPLE or not total_flagged:
            continue
        share_of_flags  = flagged / total_flagged
        share_of_orders = pm_total.get(pm, 0) / len(orders)
        if not share_of_orders:
            continue
        ratio = share_of_flags / share_of_orders
        if ratio >= NOTABLE_RATIO:
            alerts.append({
                "severity": _severity(ratio),
                "title":    f"{pm} is over-represented in high-risk orders",
                "detail":   (
                    f"{pm} makes up {share_of_orders:.0%} of all orders but "
                    f"{share_of_flags:.0%} of HIGH/CRITICAL flags — {ratio:.1f}x "
                    f"its expected share."
                ),
                "metric": round(share_of_flags, 4),
            })

    # ── 3. New-customer cohort return rate ───────────────────────────────────
    new_total    = sum(1 for o in orders if o.is_new_customer)
    new_returned = sum((o.is_returned or 0) for o in orders if o.is_new_customer)
    old_total    = len(orders) - new_total
    old_returned = sum((o.is_returned or 0) for o in orders if not o.is_new_customer)

    if new_total >= MIN_SAMPLE and old_total >= MIN_SAMPLE and old_returned:
        new_rate = new_returned / new_total
        old_rate = old_returned / old_total
        ratio = new_rate / old_rate if old_rate else 1
        if ratio >= NOTABLE_RATIO:
            alerts.append({
                "severity": _severity(ratio),
                "title":    "New customers return at a notably higher rate",
                "detail":   (
                    f"New customers return {new_rate:.0%} of orders vs {old_rate:.0%} "
                    f"for returning customers — {ratio:.1f}x higher."
                ),
                "metric": round(new_rate, 4),
            })

    order_key = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    alerts.sort(key=lambda a: order_key[a["severity"]])
    return alerts

"""
Shared ₹ cost assumptions for a mid-size Indian e-commerce merchant.
Used by ml/evaluate.py (offline threshold sweep) AND backend/routes/orders.py
(the live "₹ at stake" figure on the Review Queue) so both numbers always
agree. Swap these for real finance numbers when you have them.

  REVIEW_COST_PER_FLAG        : ops cost to verify/intervene on one flagged order
  RETURN_COST_PCT             : reverse logistics + restocking + refund
                                 processing, as a % of order value, paid in
                                 full whenever a return is NOT caught
  INTERVENTION_EFFECTIVENESS  : fraction of that return cost avoided when a
                                 true positive is caught and acted on
                                 (address/COD verification, stricter return
                                 terms, etc.) — the rest is still incurred
                                 because intervention rarely stops 100%
"""

REVIEW_COST_PER_FLAG       = 40.0
RETURN_COST_PCT            = 0.20
INTERVENTION_EFFECTIVENESS = 0.55

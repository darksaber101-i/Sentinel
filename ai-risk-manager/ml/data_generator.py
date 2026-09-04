"""
Synthetic E-Commerce Order Dataset Generator
─────────────────────────────────────────────
WHY SYNTHETIC DATA?
  Real merchant data contains PII and is commercially sensitive.
  For a portfolio project we generate realistic data that mirrors
  patterns found in published e-commerce return studies.

DESIGN DECISIONS
  - 15,000 orders (large enough for meaningful ML, small enough for fast training)
  - ~28% overall return rate  (realistic but imbalanced → forces us to use proper metrics)
  - Feature correlations are realistic but NOISY (prevents a perfect model, which would
    be suspicious and un-learnable in real life)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_ORDERS = 15_000
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "synthetic_orders.csv"

# Category base return rates come from industry benchmarks:
#   Fashion ~35%, Electronics ~18%, Books ~8%, Home ~22%
CATEGORY_CONFIG = {
    "Fashion":        {"return_rate": 0.38, "avg_value": 2200, "std_value": 1800},
    "Electronics":    {"return_rate": 0.18, "avg_value": 8500, "std_value": 5000},
    "Books":          {"return_rate": 0.07, "avg_value": 450,  "std_value": 300},
    "Home & Kitchen": {"return_rate": 0.22, "avg_value": 3200, "std_value": 2000},
    "Sports":         {"return_rate": 0.16, "avg_value": 2800, "std_value": 1500},
    "Beauty":         {"return_rate": 0.21, "avg_value": 900,  "std_value": 600},
    "Toys":           {"return_rate": 0.19, "avg_value": 1200, "std_value": 800},
    "Jewelry":        {"return_rate": 0.28, "avg_value": 5500, "std_value": 4000},
}

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "COD", "Wallet"]
DEVICE_TYPES    = ["mobile", "desktop", "tablet"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n: int = N_ORDERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = list(CATEGORY_CONFIG.keys())
    cat_weights = [1.5, 1.0, 0.8, 1.2, 0.9, 1.1, 0.9, 0.6]  # sampling frequency

    # ── Core identifiers ───────────────────────────────────────────────────────
    order_ids    = [f"ORD-{10000 + i}" for i in range(n)]
    customer_ids = [f"CUST-{rng.integers(1000, 6000)}" for _ in range(n)]
    product_ids  = [f"PROD-{rng.integers(100, 500)}"   for _ in range(n)]

    # ── Product category & base rate ──────────────────────────────────────────
    chosen_cats   = rng.choice(categories, size=n, p=np.array(cat_weights) / sum(cat_weights))
    product_return_rate = np.array([CATEGORY_CONFIG[c]["return_rate"] for c in chosen_cats])

    # ── Order value (category-specific distribution) ──────────────────────────
    order_value = np.array([
        max(99, rng.normal(CATEGORY_CONFIG[c]["avg_value"], CATEGORY_CONFIG[c]["std_value"]))
        for c in chosen_cats
    ]).round(2)

    # ── Customer history ──────────────────────────────────────────────────────
    is_new_customer       = rng.random(n) < 0.20            # 20% new customers
    customer_tenure_days  = np.where(
        is_new_customer,
        rng.integers(1, 30, size=n),
        rng.integers(30, 1500, size=n)
    )
    previous_orders = np.where(
        is_new_customer,
        0,
        rng.integers(1, 50, size=n)
    )
    # Returns can't exceed previous orders
    previous_returns = np.array([
        rng.integers(0, max(1, int(po * 0.6) + 1))
        for po in previous_orders
    ])
    previous_return_rate = np.where(
        previous_orders > 0,
        previous_returns / np.where(previous_orders > 0, previous_orders, 1),
        0.0
    )
    is_first_order = (previous_orders == 0).astype(int)

    # ── Order features ────────────────────────────────────────────────────────
    quantity            = rng.integers(1, 6, size=n)
    discount_percentage = rng.choice(
        [0, 5, 10, 15, 20, 25, 30, 40, 50, 60],
        size=n,
        p=[0.15, 0.12, 0.15, 0.12, 0.12, 0.10, 0.10, 0.07, 0.05, 0.02]
    ).astype(float)

    # ── Payment & device ──────────────────────────────────────────────────────
    payment_method = rng.choice(
        PAYMENT_METHODS,
        size=n,
        p=[0.35, 0.20, 0.18, 0.08, 0.12, 0.07]
    )
    device_type = rng.choice(DEVICE_TYPES, size=n, p=[0.65, 0.28, 0.07])

    # ── Shipping & delivery ───────────────────────────────────────────────────
    shipping_distance_km = rng.integers(5, 2000, size=n).astype(float)
    delivery_days        = np.clip(
        rng.normal(4, 2, size=n) + shipping_distance_km / 500,
        1, 15
    ).round(0).astype(int)

    # ── Behavioral signals ────────────────────────────────────────────────────
    order_hour           = rng.integers(0, 24, size=n)
    days_since_last_order = np.where(
        is_new_customer | (previous_orders == 0),
        -1,
        rng.integers(1, 365, size=n)
    )
    previous_failed_payments = rng.choice([0, 1, 2, 3], size=n, p=[0.75, 0.15, 0.07, 0.03])
    previous_chargebacks     = rng.choice([0, 1, 2],    size=n, p=[0.88, 0.09, 0.03])
    support_tickets          = rng.integers(0, 8, size=n)

    # ── Compute return probability (the ground-truth signal) ──────────────────
    # Each driver has an empirically-motivated weight.
    # We use a log-odds (logit) formulation so the final sigmoid gives a
    # probability — the same math as logistic regression.
    logit = (
        -2.20                                                       # intercept (baseline ~10%; pushed to ~27% by features)
        + 3.00 * previous_return_rate                               # strongest driver — well-established in literature
        + 2.50 * product_return_rate                                # category signal
        + 1.00 * (discount_percentage / 100)                        # discount nudge
        + 0.70 * is_new_customer.astype(float)                      # new-customer uncertainty
        + 0.60 * (payment_method == "COD").astype(float)            # COD → easy rejection at door
        + 0.40 * (quantity / 5)                                     # more items → more chance
        + 0.35 * (delivery_days / 15)                               # late delivery → unmet expectations
        + 0.40 * (previous_chargebacks * 0.5)                       # chargeback history
        + 0.30 * (support_tickets / 7)                              # past issues signal dissatisfaction
        + 0.20 * np.clip((order_value - 5000) / 20000, 0, 1)        # high-value friction
        - 0.50 * np.clip(customer_tenure_days / 365, 0, 3)          # loyalty lowers return risk
    )

    prob = _sigmoid(logit)

    # Add realistic noise: 5% of records flip — keeps the problem learnable but not trivial
    noise_mask = rng.random(n) < 0.05
    prob = np.where(noise_mask, rng.random(n), prob)

    is_returned = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "order_id":                order_ids,
        "customer_id":             customer_ids,
        "product_id":              product_ids,
        "product_category":        chosen_cats,
        "order_value":             order_value,
        "quantity":                quantity,
        "discount_percentage":     discount_percentage,
        "customer_tenure_days":    customer_tenure_days,
        "previous_orders":         previous_orders,
        "previous_returns":        previous_returns,
        "previous_return_rate":    previous_return_rate.round(4),
        "previous_failed_payments":previous_failed_payments,
        "previous_chargebacks":    previous_chargebacks,
        "payment_method":          payment_method,
        "device_type":             device_type,
        "shipping_distance_km":    shipping_distance_km,
        "delivery_days":           delivery_days,
        "is_new_customer":         is_new_customer.astype(int),
        "is_first_order":          is_first_order,
        "order_hour":              order_hour,
        "days_since_last_order":   days_since_last_order,
        "product_return_rate":     product_return_rate.round(4),
        "support_tickets":         support_tickets,
        "is_returned":             is_returned,
    })

    return df


def main():
    print("Generating synthetic dataset …")
    df = generate_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return_rate = df["is_returned"].mean()
    print(f"  Orders     : {len(df):,}")
    print(f"  Return rate: {return_rate:.1%}")
    print(f"  Saved to   : {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    main()

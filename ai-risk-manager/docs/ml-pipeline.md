# ML Pipeline Documentation

## 1. Why This Problem?

E-commerce returns cost merchants 3–4× the original shipping cost. Predicting
returns **before fulfillment** lets a merchant decide whether to add friction
(e.g., require prepayment for COD orders) or flag for manual review — without
automatically rejecting customers.

This is a binary classification problem: given an order, predict whether it
will be returned (1) or kept (0).

---

## 2. Synthetic Data Generation

**Why synthetic?** Real merchant data is commercially sensitive and contains PII.
Synthetic data lets us demonstrate the full ML lifecycle without ethical or legal issues.

**How realistic is it?**
- Category return rates match industry benchmarks (Fashion ~38%, Electronics ~18%)
- Feature correlations are based on published e-commerce research
- 5% random noise prevents the model from achieving unrealistic accuracy

**Key signal drivers embedded in the data:**
| Feature | Effect |
|---------|--------|
| `previous_return_rate` | Strongest predictor — history predicts future |
| `product_return_rate` | Category-level baseline risk |
| `discount_percentage` | Higher discount → try-and-return behavior |
| `payment_method = COD` | Easy rejection at doorstep |
| `is_new_customer` | Unknown reliability |
| `customer_tenure_days` | Longer tenure → more loyal → less returns |

---

## 3. Feature Engineering

Raw data is not enough. We engineer features that capture domain intuition:

| Derived Feature | Formula | Intuition |
|----------------|---------|-----------|
| `value_per_item` | `order_value / quantity` | Expensive individual items → higher return scrutiny |
| `return_to_order_ratio` | `previous_returns / previous_orders` | Redundant with `previous_return_rate` but captures denominator |
| `is_late_night_order` | `hour ∈ [0,5]` | Impulse purchases → higher return risk |
| `is_high_discount` | `discount ≥ 40%` | Binary signal for aggressive discount |
| `is_repeat_returner` | `return_rate > 50%` | Binary flag for serial returners |
| `days_since_last_order_clean` | Replace -1 with median | New customers have -1; needs imputation |

---

## 4. Why Classification?

We're predicting a binary outcome (returned / not returned), not a continuous value.
Binary classification is the right tool. The model outputs a **probability** (0–1),
which we convert to a risk score (0–100) for human readability.

---

## 5. Why Multiple Models?

Different algorithms have different inductive biases:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Logistic Regression** | Fast, interpretable, great baseline | Assumes linear decision boundary |
| **Random Forest** | Captures non-linear interactions, robust | Slower, less interpretable |
| **Gradient Boosting** | State-of-the-art on tabular data | Needs careful tuning |

We compare all three and pick the best by F1 on the **validation set**.

---

## 6. Train / Validation / Test Split

```
Total: 15,000 orders
  Train: 10,500 (70%)   ← model learns from this
  Val:    2,250 (15%)   ← model selection and hyperparameter tuning
  Test:   2,250 (15%)   ← final honest evaluation (NEVER touched during training)
```

**Why stratify?** With 35% return rate, random splits might give 20% returns
in one split and 50% in another. Stratification ensures each split has the same
class ratio, making comparisons valid.

**Data leakage rule:** The test set is sealed until after the best model is chosen.
Any peek at test metrics during development would give falsely optimistic results.

---

## 7. Why Precision and Recall Matter

A simple accuracy metric is misleading with imbalanced data.

**Scenario:** If 65% of orders are not returned, a model that predicts "no return"
for *every* order achieves 65% accuracy — while catching zero actual returns.

| Metric | What It Answers |
|--------|----------------|
| **Precision** | Of orders I flagged as risky, what fraction actually returned? |
| **Recall** | Of all returns that occurred, what fraction did I catch? |
| **F1** | Harmonic mean — penalizes models that sacrifice one for the other |
| **PR-AUC** | Area under the precision-recall curve — single-number summary for imbalanced data |

**Business trade-off:**
- High recall → catch more returns → higher ops review cost
- High precision → fewer false alarms → miss some returns

The threshold simulator lets you tune this trade-off based on business costs.

---

## 8. How Threshold Selection Works

The model outputs `P(return) = 0.73`. We still need to choose a cutoff:

```
Threshold = 0.5  →  if P ≥ 0.5: flag as HIGH RISK
Threshold = 0.3  →  if P ≥ 0.3: flag as HIGH RISK (more sensitive)
Threshold = 0.7  →  if P ≥ 0.7: flag as HIGH RISK (more conservative)
```

Lower threshold → higher recall, lower precision  
Higher threshold → lower recall, higher precision  

The Threshold Simulator page lets you explore this trade-off on the actual test set.

---

## 9. Feature Contributions (XAI)

We use a local feature contribution method:

```
contribution_i = importance_i × |z_score_i|

where z_score_i = (feature_value - training_mean) / training_std
```

This answers: "For THIS specific order, which features deviated most from average
AND had the highest global importance?"

The method is conceptually equivalent to SHAP's first-order approximation and
is easier to explain in interviews without requiring numba/C extensions.

---

## 10. Limitations

1. **Synthetic data** — correlations are manually designed, not learned from real patterns
2. **No temporal features** — real systems would use day-of-week, seasonality, etc.
3. **Static model** — no online learning; the model doesn't update as new returns come in
4. **No customer segmentation** — a more sophisticated system would cluster customers
5. **Logistic Regression won best** — expected with linearly-generated data; XGBoost would win on real messy data
6. **Threshold is business-dependent** — 0.5 is a placeholder, not an optimized business decision

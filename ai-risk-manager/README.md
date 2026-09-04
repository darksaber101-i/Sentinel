# Sentinel

> **"Detect risk before it becomes loss."**

An end-to-end ML platform that predicts whether an e-commerce order will be
returned before fulfillment — with explainable predictions, a real-time
dashboard, and an AI assistant.

---

## Architecture

```
User
 ↓
Next.js Dashboard  (localhost:3000)
 ↓
FastAPI Backend    (localhost:8000)
 ↓  ↓
ML Model          SQLite / PostgreSQL
 ↓
Trained XGBoost / Random Forest / Logistic Regression
 ↓
Synthetic Orders Dataset (15,000 records)

AI Assistant
 ↓
FastAPI → Claude (Anthropic) or Rule-Based Fallback
 ↓
Live Application Data (no hallucinations)
```

---

## Features

| Feature | Description |
|---------|-------------|
| Risk Scoring | 0–100 score + LOW/MEDIUM/HIGH/CRITICAL level |
| 3 ML Models | Logistic Regression, Random Forest, XGBoost |
| Model Evaluation | Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix |
| Explainability | Per-order feature contributions (SHAP-style) |
| Threshold Simulator | See precision/recall trade-off across thresholds |
| AI Assistant | Grounded Q&A about model and order data |
| Orders Table | Searchable, filterable, paginated |
| Order Detail | Full risk breakdown with explanations |
| Real Metrics | Every number comes from the actual trained model |

---

## ML Pipeline

```
Synthetic Data (15,000 orders)
 ↓
Feature Engineering (22 raw + 6 derived features)
 ↓
Stratified Train/Val/Test Split (70/15/15)
 ↓
Train: Logistic Regression, Random Forest, XGBoost
 ↓
Select best by F1 on validation set
 ↓
Final evaluation on held-out test set (NEVER used during training)
 ↓
Save model + evaluation results
 ↓
FastAPI backend loads model at startup
```

---

## Real Model Results

> These values are computed from the actual trained model — not hardcoded.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| **Logistic Regression (selected)** | 0.477 | 0.698 | 0.567 | 0.704 | 0.548 |
| Random Forest | 0.487 | 0.667 | 0.563 | 0.687 | 0.518 |
| Gradient Boosting | 0.439 | 0.740 | 0.551 | 0.672 | 0.511 |

*Run `python -m ml.evaluate` to regenerate against your own data.*

### Cost-adjusted view

Precision/recall alone doesn't say whether the model saves money — a false positive
(reviewing a good order) and a false negative (missing a real return) cost very
different amounts. On the held-out test set, using flat assumptions of ₹40 per
manual review and 20% of order value per uncaught return (55% mitigated when
caught):

| | Value |
|---|---|
| Baseline loss (flag nothing) | ₹510,602 |
| Best threshold by F1 | 50% |
| Best threshold by ₹ savings | **15%** |
| Net savings at the ₹-optimal threshold | **₹190,996 (~37%)** |

The F1-optimal threshold and the cost-optimal threshold are not the same —
optimizing for a symmetric ML metric leaves real money on the table. See the
Threshold Simulator page (`/threshold-simulator`) for the interactive version,
and `ml/evaluate.py` for the cost model and its assumptions.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost, pandas, numpy |
| Backend | FastAPI, SQLAlchemy, Pydantic, uvicorn |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| AI Assistant | Anthropic Claude (with rule-based fallback) |

---

## Quick Start

### Option 1: Automated Setup

```bash
python setup.py
```

### Option 2: Manual

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
PYTHONPATH=. python -m ml.data_generator

# 3. Train models
PYTHONPATH=. python -m ml.train

# 4. Evaluate (test set)
PYTHONPATH=. python -m ml.evaluate

# 5. Start backend
cp .env.example .env
PYTHONPATH=. uvicorn backend.main:app --reload

# 6. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open: http://localhost:3000

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/orders` | GET | List all orders (paginated, filterable) |
| `/api/orders/{id}` | GET | Single order with risk details |
| `/api/predict` | POST | Score a new order |
| `/api/batch-predict` | POST | Score multiple orders |
| `/api/metrics` | GET | Model KPI metrics |
| `/api/model-performance` | GET | Full evaluation results |
| `/api/risk-distribution` | GET | Count by risk level |
| `/api/threshold-analysis` | GET | Precision/recall across thresholds |
| `/api/cost-analysis` | GET | ₹ cost/savings trade-off across thresholds |
| `/api/dashboard-stats` | GET | All dashboard data in one call |
| `/api/assistant` | POST | AI assistant Q&A |

Interactive docs: http://localhost:8000/docs

---

## Environment Variables

```env
DATABASE_URL=sqlite:///./ai_risk_manager.db
ANTHROPIC_API_KEY=your_key_here   # optional — falls back to rule-based answers
FRONTEND_URL=http://localhost:3000
```

---

## Dataset

15,000 synthetic e-commerce orders with:
- 8 product categories (Fashion, Electronics, Books, Home & Kitchen, Sports, Beauty, Toys, Jewelry)
- Realistic class imbalance (~35% return rate)
- 22 features capturing order, customer, product, and behavioral signals
- Deliberate noise (5%) to prevent perfect prediction

---

## Key Limitations

1. Synthetic data — not trained on real merchant transactions
2. No temporal features (seasonality, day-of-week effects)
3. Static model — doesn't update as new data arrives
4. Logistic Regression won on this linearly-generated data; XGBoost typically wins on real messy data

---

## Future Improvements

- [ ] Real merchant data integration
- [ ] Model retraining pipeline
- [ ] Drift detection monitoring
- [ ] Customer segmentation
- [ ] A/B testing framework for thresholds
- [ ] Slack/email alerts for CRITICAL risk orders

---

*Built as a portfolio project to demonstrate the complete ML engineering lifecycle.*

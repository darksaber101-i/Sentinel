<div align="center">

# Sentinel

### Risk decisioning where the metric is money, not F1

**Two production-shaped ML systems built around one argument:**
*the threshold that maximises a machine-learning metric and the threshold that*
*maximises money saved are not the same number — and the gap is measurable.*

<br>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14.1-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-337AB7?style=flat-square)
![Tests](https://img.shields.io/badge/tests-21%20passing-success?style=flat-square)

</div>

---

## The one-paragraph version

Most risk projects stop at *"here's the precision and recall."* But precision and recall are
**symmetric** — they treat a false positive and a false negative as equally interesting events.
In risk, they never are. Reviewing a good order costs ₹40; missing a fraudulent transaction costs
₹1.57M. This repository contains two systems that take that asymmetry seriously, and the
headline result is that **optimising a symmetric ML metric measurably loses money**:

| System | F1-optimal threshold | ₹-optimal threshold | Cost of choosing F1 |
|---|---|---|---|
| **Sentinel** — return risk | 0.50 | **0.15** | ₹190,996 of avoidable loss (37%) |
| **Fraud Detector** — transaction fraud | 0.989 | **0.090** | **₹110.5M** left on the table |

---

## Two systems, deliberately different

| | **Sentinel** `ai-risk-manager/` | **Fraud Detector** `fraud-detector/` |
|---|---|---|
| **Domain** | E-commerce return risk | Mobile-money transaction fraud |
| **Decision** | Hold / verify an order *before fulfilment* | Flag a transaction for review |
| **Data** | 15,000 synthetic orders (~35% return rate) | PaySim — 6.36M transactions, 8,213 fraud (0.13%) |
| **Shape** | Full product — ML + API + DB + dashboard | Deep analysis — 11 scripts + CLI + HTTP API |
| **Proves** | It ships | It's true |

Sentinel shows a model becoming a **product**: a routing decision with an audit trail and a human
in the loop. Fraud Detector shows a model being **interrogated**: leakage-audited, adversarially
stressed, calibrated, bounded with confidence intervals, and benchmarked against an unsupervised
baseline.

---

## Quick start

**Prerequisites** — Python 3.11+, Node 18+, ~2GB disk.

```bash
git clone <your-repo-url> sentinel && cd sentinel
```

### 1 · Backend

```bash
cd ai-risk-manager
pip install -r requirements.txt
cp .env.example .env          # Windows: Copy-Item .env.example .env

# Generate data → train → evaluate  (first run only, ~2 min)
PYTHONPATH=. python -m ml.data_generator
PYTHONPATH=. python -m ml.train
PYTHONPATH=. python -m ml.evaluate

PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
```

<details>
<summary><b>Windows PowerShell equivalent</b></summary>

```powershell
cd ai-risk-manager
pip install -r requirements.txt
Copy-Item .env.example .env
$env:PYTHONPATH = "."
py -3 -m ml.data_generator
py -3 -m ml.train
py -3 -m ml.evaluate
py -3 -m uvicorn backend.main:app --reload --port 8000
```
</details>

### 2 · Frontend

In a **second** terminal:

```bash
cd ai-risk-manager/frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| Dashboard | <http://localhost:3000> |
| API | <http://localhost:8000> |
| Interactive API docs | <http://localhost:8000/docs> |

> **Start the backend first.** The frontend reads `NEXT_PUBLIC_API_URL` (default
> `http://localhost:8000`), and CORS on the backend allows exactly the origin in `FRONTEND_URL`.
> If Next.js offers to use port 3001 because 3000 is busy, **say no** — a second dev server
> sharing the same `.next` directory corrupts both, and 3001 is not an allowed CORS origin.

### 3 · Fraud detector *(optional, separate)*

Needs the PaySim dataset — 471MB, too large for git, so it isn't committed.
Download `PS_20174392719_1491204439457_log.csv` from
[Kaggle: PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) into the repo root.

```bash
cd fraud-detector
python predict.py                      # interactive verifier
python -m uvicorn api:app --port 8010  # HTTP API, docs at /docs
```

---

## What's inside

### Sentinel — the product

<table>
<tr><td width="50%" valign="top">

**Seven UI surfaces**
- Control Center — KPIs, risk distribution, alerts
- Orders — searchable, filterable, paginated
- Order detail — full risk breakdown + contributions
- Review Queue — approve / hold / escalate
- Model Trust — confusion matrix, ROC, PR curves
- Policy Simulator — the ₹-vs-F1 threshold sweep
- AI Assistant — grounded Q&A

</td><td width="50%" valign="top">

**Fifteen API endpoints**
- `GET /api/orders` · `/api/orders/{id}`
- `GET /api/orders/review-queue`
- `GET /api/orders/{id}/actions`
- `POST /api/orders/{id}/action`
- `POST /api/predict` · `/api/batch-predict`
- `GET /api/dashboard-stats` · `/api/metrics`
- `GET /api/model-performance`
- `GET /api/threshold-analysis` · `/api/cost-analysis`
- `GET /api/risk-distribution` · `/api/alerts`
- `POST /api/assistant`

</td></tr>
</table>

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Next.js 14  ·  TypeScript · Tailwind · Recharts         │
│  dashboard · orders · review queue · simulator · assistant│
└───────────────────────────┬──────────────────────────────┘
                            │  HTTP  (CORS-scoped)
┌───────────────────────────▼──────────────────────────────┐
│  FastAPI  ·  Pydantic schemas · SQLAlchemy ORM            │
│  routes/  orders · predictions · metrics · alerts · assistant│
└──────┬─────────────────────────────────┬─────────────────┘
       │                                 │
┌──────▼───────────────┐      ┌──────────▼──────────────────┐
│ Pickled artefacts    │      │ SQLite (dev) / PostgreSQL   │
│ model · scaler ·     │      │ orders · predictions ·      │
│ metadata · eval JSON │      │ audit_logs                  │
└──────▲───────────────┘      └─────────────────────────────┘
       │
┌──────┴───────────────────────────────────────────────────┐
│  ml/  data_generator → preprocessing → train → evaluate  │
│       cost_config (shared ₹ assumptions) · explain        │
└──────────────────────────────────────────────────────────┘
```

---

## Results

> Every number below is regenerated by the pipeline — nothing is hardcoded in the UI.
> Reproduce with `python -m ml.evaluate` and `fraud-detector/*.py`.

### Sentinel · model selection

Stratified 70/15/15 split (10,500 / 2,250 / 2,250). Best model chosen by **validation** F1; the
test set is evaluated **once**.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| **Logistic Regression** *(selected)* | **0.477** | **0.698** | **0.567** | **0.704** |
| Random Forest | 0.472 | 0.636 | 0.542 | 0.676 |
| Gradient Boosting | 0.444 | 0.735 | 0.553 | 0.666 |

<sub>Held-out **test-set** figures. `models/training_metadata.json` stores the **validation**
numbers used for selection — different split, not a contradiction.</sub>

An AUC of 0.70 is deliberate: `ml/data_generator.py` flips 5% of labels so the problem stays
learnable but not trivial. A synthetic dataset a model can ace teaches you nothing.

### Sentinel · the cost model

Three named assumptions in `ml/cost_config.py`, imported by **both** the offline sweep and the
live dashboard so they cannot drift apart:

```python
REVIEW_COST_PER_FLAG       = 40.0   # ops cost per flagged order
RETURN_COST_PCT            = 0.20   # reverse logistics, as % of order value
INTERVENTION_EFFECTIVENESS = 0.55   # fraction of return cost avoided when caught
```

| | Value |
|---|---|
| Baseline loss (flag nothing) | ₹510,602 |
| Best threshold by F1 | 0.50 |
| **Best threshold by ₹ saved** | **0.15** |
| **Net savings at the ₹-optimal threshold** | **₹190,996 (~37%)** |

### Fraud Detector · why the weaker model ships

PaySim's generator drains the origin account to zero in **99.2%** of fraud rows. A model given
that feature scores a near-perfect PR-AUC — and is worthless.

| Model | Features | Test PR-AUC | Precision | Recall |
|---|---|---:|---:|---:|
| Full | includes origin-balance drain | 0.9999 | 1.000 | 0.999 |
| **Honest** *(shipped)* | excludes it | **0.9449** | **0.979** | **0.838** |

**The adversarial test that settles it** (`adversarial_test.py`) — re-simulating the 4,250 real
test-set frauds while leaving *X%* of the balance behind:

| Balance left behind | Full model recall | Honest model recall |
|---|---:|---:|
| 0% *(dataset's own behaviour)* | 97.5% | 99.3% |
| **10%** | **8.8%** | **99.3%** |
| 25% | 12.9% | 99.3% |
| 50% | 16.4% | 99.3% |

The full model collapses **89 points** from a trivial change in fraudster behaviour. It was never
detecting fraud — it was detecting *"was this account emptied."*

### Fraud Detector · cost at each threshold

Held-out **temporal** test set (final 20% of time steps — 1,248,736 transactions, 4,250 fraud).
False-negative cost is the transaction's real amount; false-positive cost is a ₹50 assumption.

| Threshold | Precision | Recall | Fraud ₹ missed | Review ₹ cost | **Total cost** |
|---|---:|---:|---:|---:|---:|
| 0.500 — naive default | 0.392 | 0.976 | ₹6,988,624 | ₹322,200 | ₹7,310,824 |
| 0.989 — F1-optimal | 0.979 | 0.838 | ₹110,511,888 | ₹3,850 | ₹110,515,738 |
| **0.090 — cost-optimal** | 0.189 | **0.993** | ₹1,123,420 | ₹906,350 | **₹2,029,770** |

A missed fraud costs ~**31,000×** a false-positive review, so tanking precision to 18.9% to reach
99.3% recall is the *correct* decision. F1 has no concept of money.

---

## Beyond the baseline

Seven checks most projects on these datasets skip:

| # | Question | Script | Finding |
|---|---|---|---|
| 1 | Does it survive evolving fraud? | `adversarial_test.py` | Leaky model −89pp; shipped model unmoved |
| 2 | Why was this flagged? | `explain.py` | SHAP reasons on every prediction |
| 3 | Does 0.7 mean a 70% fraud rate? | `calibration.py` | Badly overconfident → isotonic fix, **Brier 0.00382 → 0.00058 (84.7%)** |
| 4 | One threshold for all fraud types? | `segment_thresholds.py` | TRANSFER nearly solved; **CASH_OUT is where effort belongs** |
| 5 | Does it decay without retraining? | `decay_over_time.py` | Recall flat ~98% across 4 weeks |
| 6 | How certain are these numbers? | `bootstrap_ci.py` | 1,000 resamples; recall 0.993 `[0.991, 0.996]` |
| 7 | What works with *no* labels? | `isolation_forest_baseline.py` | PR-AUC 0.035 → 0.125. **Still weak — reported as weak** |

<details>
<summary><b>The wrong fix, kept in the record</b></summary>

The Isolation Forest baseline scored PR-AUC 0.035 — barely above random. The reflexive fix was
feature scaling. It did nothing (0.035 → 0.035), because Isolation Forest splits *within each
feature's own range* and, unlike a distance-based method, isn't scale-sensitive — an intuition
imported from KNN that didn't apply. What actually helped: more trees (200 → 500),
`max_samples=0.5`, and `contamination='auto'` → **0.125**, a 3.6× improvement that is still weak.

The failed hypothesis is documented rather than deleted, because the wrong turn is the more
useful artefact.
</details>

---

## Repository layout

```
.
├── ai-risk-manager/              # Sentinel — the product
│   ├── backend/
│   │   ├── main.py               # app factory, CORS, DB init + seed
│   │   ├── config.py             # pydantic-settings, reads .env
│   │   ├── models.py             # Order · Prediction · AuditLog
│   │   ├── review.py             # status derived from AuditLog (single source of truth)
│   │   └── routes/               # orders · predictions · metrics · alerts · assistant
│   ├── ml/
│   │   ├── data_generator.py     # 15k synthetic orders, 5% label noise
│   │   ├── train.py              # 3 models, selection on validation F1
│   │   ├── evaluate.py           # test metrics + ₹ threshold sweep
│   │   └── cost_config.py        # shared ₹ assumptions
│   ├── frontend/src/
│   │   ├── app/(app)/            # dashboard · orders · review-queue · simulator · assistant
│   │   ├── components/           # RiskBadge · StatusBadge · ActionBar · ActionTimeline
│   │   └── lib/api.ts            # typed API client
│   ├── models/                   # pickled artefacts + evaluation JSON
│   └── tests/                    # 21 tests
│
├── fraud-detector/               # PaySim analysis
│   ├── train_ablation.py         # the honest model (shipped)
│   ├── adversarial_test.py       # the leak, quantified
│   ├── calibration.py · bootstrap_ci.py · segment_thresholds.py
│   ├── decay_over_time.py · isolation_forest_baseline.py
│   ├── predict.py · api.py       # CLI + HTTP scoring
│   └── models/                   # models + results JSON for every script
│
├── .mcp.json                     # MCP config — references ${STITCH_API_KEY}
├── .env.example                  # template for tooling secrets
└── start.ps1 / start.sh          # loads .env, then launches tooling
```

---

## Engineering notes

Three places the codebase deliberately keeps **one source of truth**:

**Review status is derived, not stored.** `backend/review.py` computes an order's status from its
most recent `AuditLog` row rather than a mutable `status` column. The shortcut loses history and
lets two modules disagree; deriving it costs a bulk query and buys full attributability of every
operator action.

**Cost assumptions live in one file.** `ml/cost_config.py` is imported by the offline sweep
(`evaluate.py`) *and* the live "₹ at stake" figure (`routes/orders.py`), which structurally
prevents the analysis and the dashboard from disagreeing.

**Metrics come from artefacts.** The UI reads `models/training_metadata.json` and
`evaluation_results.json` — the files the training run writes. Re-run training and the dashboard
moves.

Two things the code refuses to do:

- **`routes/alerts.py` won't fake a time dimension.** The obvious feature is *"risk spiked 40% in
  the last 24h"* — but the synthetic data has no real timestamps, so that number would be
  fabricated. It surfaces live risk-*concentration* signals instead.
- **The LLM never predicts.** `routes/assistant.py` scopes it to explanation only, grounded in
  live database values, with a deterministic rule-based fallback when no API key is present —
  and the response labels which one answered.

### Testing

```bash
cd ai-risk-manager && PYTHONPATH=. pytest tests/ -q     # 21 passed
```

---

## Configuration

`ai-risk-manager/.env` (copy from `.env.example`):

```env
DATABASE_URL=sqlite:///./ai_risk_manager.db
ANTHROPIC_API_KEY=          # optional — falls back to rule-based answers
FRONTEND_URL=http://localhost:3000
```

Root `.env` holds tooling secrets referenced by `.mcp.json` as `${VAR}`. Claude Code expands those
from the **system environment** and does not read `.env` itself, so use the launcher:

```powershell
Copy-Item .env.example .env    # fill in values
.\start.ps1                    # loads .env, then starts the tool
```

> **No secret is committed.** `.env`, `*.db`, `node_modules/`, `.next/`, logs, and the 471MB
> PaySim CSV are all gitignored.

---

## Known limitations

Stated plainly, because a risk project that hides its caveats has the wrong instincts:

1. **Both datasets are synthetic.** The cost framing is the contribution; the specific thresholds
   are illustrations of a method, not recommendations. Real merchant data is required before any
   of these numbers is trusted.
2. **The cost constants are placeholders.** ₹40/review, 20% of order value, 55% intervention
   effectiveness, ₹50/fraud-review — all labelled assumptions, none from a real finance function.
3. **Model selection optimises F1**, even though the project's thesis is that F1 is the wrong
   objective. Selecting the *model* on expected cost — not just the threshold — would make the
   argument end-to-end.
4. **Sentinel receives none of the fraud project's rigour** — no adversarial test, calibration
   check or confidence intervals, despite being the system with the deployment story.
5. **Logistic Regression won** because the generator produces largely linear relationships. On
   real, messier data a gradient-boosted model would be expected to win; selection is automatic,
   so nothing needs editing.
6. **Static models.** No retraining pipeline, no drift detection, no A/B framework for thresholds.
7. **No CI, no containerisation**, and the fraud scripts use absolute paths that need editing to
   run elsewhere.

---

## Roadmap

- [ ] Replace cost constants with real finance numbers
- [ ] Retrain on real merchant transactions
- [ ] Drift detection + scheduled retraining
- [ ] Select models on expected cost, not F1
- [ ] Port the adversarial / calibration / CI suite onto Sentinel
- [ ] Parameterise fraud-detector paths; add its test suite
- [ ] CI pipeline + Docker Compose
- [ ] Slack / email alerts on CRITICAL orders

---

<div align="center">
<sub>

Built to demonstrate the full ML engineering lifecycle — and the judgement to ship the model
that scored **worse**, on purpose, because it was the one that actually worked.

</sub>
</div>

# Sentinel + Fraud Detector — Project Explanation

**Framed against four criteria: Problem Taste · Build Quality · AI Judgement · Failure Recovery**

---

## What was built

Two risk-decisioning systems that share one thesis:

> **The threshold that maximises a machine-learning metric and the threshold that maximises money saved are not the same number — and the gap between them is measurable in rupees.**

| | **Sentinel** (`ai-risk-manager/`) | **Fraud Detector** (`fraud-detector/`) |
|---|---|---|
| Domain | E-commerce return risk | Mobile-money transaction fraud |
| Decision | Hold / verify an order **before fulfilment** | Flag a transaction for review |
| Data | 15,000 synthetic orders, ~35% return rate | PaySim — 6.36M transactions, 8,213 fraud (0.13%) |
| Shape | Full product: ML + API + database + dashboard | Deep analysis: 11 scripts + CLI + HTTP API |
| Purpose | Proves it ships | Proves it's true |

They are deliberately complementary. Sentinel demonstrates that a model can become a **product** — a routing decision with an audit trail and a human in the loop. Fraud Detector demonstrates that a model can be **interrogated** — leakage-checked, adversarially stressed, calibrated, bounded with confidence intervals, and compared against an unsupervised baseline.

Neither is interesting alone. Together they make the argument: *I can build the thing, and I can tell you when the thing is lying to me.*

---

# 1. Problem Taste

> *Did the project choose a problem worth solving, and frame it the way someone who understands the domain would frame it?*

## 1.1 The reframe that drives everything

The default framing of a fraud/returns project is: **"build a classifier, report precision and recall."** That framing is a trap, because precision and recall are symmetric — they weigh a false positive and a false negative as equally interesting events. In risk, they never are.

This project's framing is instead: **"choose an operating point under asymmetric cost."** The model is an input to that choice, not the deliverable.

The evidence that this isn't a retrofitted narrative is that the cost model is *load-bearing code*, not a paragraph in a README:

- `ai-risk-manager/ml/cost_config.py` — three named constants with documented meaning
- `ai-risk-manager/ml/evaluate.py:67` — `threshold_sweep()` computes precision/recall **and rupees** at every threshold from 0.10 to 0.90
- `ai-risk-manager/backend/routes/orders.py:12` — the live "₹ at stake" figure on the review queue imports `RETURN_COST_PCT` from that same module

## 1.2 What the reframe actually surfaced

**Sentinel** — on the held-out test set:

| | Value |
|---|---|
| Baseline loss (flag nothing) | ₹510,602 |
| F1-optimal threshold | 0.50 |
| **₹-optimal threshold** | **0.15** |
| Net savings at ₹-optimal | **₹190,996 (~37%)** |

**Fraud Detector** — the same analysis, where the asymmetry is extreme:

| Threshold | Precision | Recall | Fraud ₹ missed | Review ₹ cost | Total cost |
|---|---|---|---|---|---|
| 0.500 — naive default | 0.392 | 0.976 | ₹6,988,624 | ₹322,200 | ₹7,310,824 |
| 0.989 — F1-optimal | 0.979 | 0.838 | ₹110,511,888 | ₹3,850 | ₹110,515,738 |
| **0.090 — cost-optimal** | **0.189** | **0.993** | **₹1,123,420** | **₹906,350** | **₹2,029,770** |

The average fraudulent transaction in the test set is ~₹1.57M. A manual review costs ₹50. **A missed fraud costs roughly 31,000× a false positive.** Under that ratio it is *correct* to drive precision down to 18.9% in order to reach 99.3% recall — 18,127 unnecessary ₹50 reviews cost ₹906K, which is cheap against even one more missed multi-lakh fraud.

The F1-optimal threshold looks safer (97.9% precision) and leaves **₹110.5M** on the table — 54× more loss — because F1 has no concept of money.

## 1.3 Taste shown in what was *refused*

Good taste shows as much in omissions as in features.

**Account IDs excluded as features.** `nameOrig` / `nameDest` are high-cardinality identifiers. Including them would have inflated test metrics while generalising to zero new accounts. They were deliberately dropped and the choice is documented as a limitation.

**Alerts refused to fake a time dimension.** From `ai-risk-manager/backend/routes/alerts.py`:

> *"These are NOT time-series spike alerts: the synthetic dataset has no real date dimension (every row was seeded in a single batch; order_hour is hour-of-day, not a timestamp), so claiming a 'spike in the last 24h' would be a fabricated number."*

The obvious, demo-friendly feature was a 24-hour spike alert. It was rejected because the data cannot support it, and replaced with live risk-*concentration* alerts (min sample 30; medium at ≥1.3× platform rate, high at ≥1.75×) — which the data *can* support.

**Noise injected on purpose.** `ml/data_generator.py:144` flips 5% of labels: *"keeps the problem learnable but not trivial."* The easy move with synthetic data is to generate a dataset your model can ace. This does the opposite, deliberately capping achievable performance.

**PaySim's own rule kept as a baseline to beat, not as a feature.** `isFlaggedFraud` catches 16 of 8,213 frauds (0.2% recall). Using it as a feature would have been free signal; it was instead retained as the bar to clear.

## 1.4 Where taste is weakest

- **Both datasets are synthetic.** The cost framing is the strong idea, but it is demonstrated on invented data. The thresholds are illustrations of a method, not recommendations.
- **The cost constants are guesses.** ₹40/review, 20% of order value, 55% intervention effectiveness — all labelled as assumptions, none sourced from a real finance function. The framework is right; the inputs are placeholders.
- **Sentinel's problem is a notch less urgent than the fraud problem.** Returns matter, but the fraud side is where the asymmetry argument really bites. The two projects are not equally sharp.

---

# 2. Build Quality

> *Is this engineered, or is it a notebook with a UI bolted on?*

## 2.1 Architecture

```
Next.js 14 dashboard  (TypeScript · Tailwind · Recharts)
        ↓  HTTP
FastAPI backend  (Pydantic schemas · SQLAlchemy ORM)
        ↓                    ↓
Pickled model artefacts   SQLite (dev) / PostgreSQL (prod)
        ↑
ml/ pipeline  (generate → preprocess → train → evaluate → explain)
```

Seven frontend surfaces: dashboard, orders table, order detail, review queue, threshold simulator, model performance, AI assistant. Eleven API endpoints. Twenty-one tests across `test_api.py`, `test_data_generator.py`, `test_predict.py`.

## 2.2 Three decisions that indicate engineering judgement

### One source of truth for review status

`backend/review.py` derives an order's status from its most recent `AuditLog` row rather than storing a mutable `status` column:

> *"An order's current review status is its most recent AuditLog row — there is no separate status column, so orders.py and metrics.py both read through here to avoid two sources of truth."*

The shortcut is an `order.status` field updated in place. That shortcut loses history, and lets two modules disagree about the same order. Deriving it costs a bulk query and buys full attributability of every operator action — which in a risk system is not a nice-to-have.

### One source of truth for cost

`ml/cost_config.py` is imported by both the offline threshold sweep (`evaluate.py:33`) and the live dashboard figure (`orders.py:12`). This structurally prevents a very common failure where the analysis says one number and the UI says another. The docstring is explicit that this is the point.

### One source of truth for metrics

The dashboard reads `models/training_metadata.json` and `models/evaluation_results.json` — the files the training run writes. No metric is hardcoded in the frontend. Re-run training and the UI moves.

## 2.3 ML pipeline discipline

| Practice | Where | Why it matters |
|---|---|---|
| Stratified 70/15/15 split | `ml/train.py` | Preserves class balance across splits |
| Model selected on **validation**, evaluated once on **test** | `ml/train.py` → `ml/evaluate.py` | Test set never influences selection |
| **Temporal** split on PaySim (train ≤ step 594, test > 594) | `fraud-detector/train_ablation.py` | Fraud rate moves over time (0.08% train → 0.34% test); a random shuffle leaks the future backwards |
| Artefacts versioned with metadata | `models/training_metadata.json` | Train size, val size, test size, feature count, per-model results |
| Memory-conscious data loading | `dtype` maps + `usecols` throughout `fraud-detector/` | 6.36M rows loaded as `float32`/`int8` rather than `float64` |

## 2.4 Code as documentation

Every non-obvious module opens with a docstring that explains **why**, not what. `calibration.py`:

> *"XGBoost trained with scale_pos_weight (needed here for the 1:1225 class imbalance) is a common source of miscalibration: it inflates positive-class scores to compensate for the imbalance, which helps ranking (what PR-AUC/recall measure) but skews the actual probability values."*

`adversarial_test.py` states its own method and its own limits:

> *"This is NOT retraining — it's stress-testing already-trained, already-thresholded models against a threat model they were not specifically trained to catch."*

This is the difference between code that has been written and code that has been *reasoned about*.

## 2.5 Where build quality is weakest

- **Hardcoded absolute paths.** `fraud-detector/*.py` embeds `C:\Users\Ishan\Desktop\razorpay\...`. The scripts will not run on another machine without editing. Sentinel does this correctly with `Path(__file__).parent`; the fraud project does not.
- **Test coverage is thin and one-sided.** 21 tests on Sentinel; **no unit-test suite at all** on the fraud detector, which is the more analytically complex codebase. (`adversarial_test.py` is an analysis script, not a pytest suite — it verifies the model, not the code.)
- **No CI, no containerisation, not a git repository.** There is no automated check that the tests still pass or that the README's numbers still match the artefacts — which is precisely how the drift in §4.6 happened.
- **Log files and build artefacts committed alongside source** (`backend.log`, `frontend2_err.log`, `.next/`). Housekeeping.
- **A live API key is checked into `.mcp.json`.** See the note at the end of this document — this one should be fixed today.

---

# 3. AI Judgement

> *Does the project know what machine learning is for, where it breaks, and — separately — where an LLM belongs?*

This splits into two distinct questions, and the project answers both.

## 3.1 Judgement about the ML

### Knowing that a great score is a warning sign

The full-feature fraud model scores **PR-AUC 0.9999, precision 1.000, recall 0.999**. The correct reaction to that on a 0.13%-base-rate problem is suspicion, not celebration. `check_leakage.py` confirmed the cause: PaySim's generator drains the origin account to zero in **99.2%** of fraud rows. The model had learned *"was this account emptied"* — a proxy for the label, not a property of fraud.

### Proving the leak instead of asserting it

Suspecting leakage is cheap. `adversarial_test.py` measured it: take the 4,250 real test-set frauds, re-simulate them leaving X% of balance behind (recomputing `newbalanceOrig` and the error-balance features consistently), and re-score both models at their already-fixed thresholds.

| Balance left behind | Full model recall | Honest model recall |
|---|---|---|
| 0% (PaySim's own behaviour) | 97.5% | 99.3% |
| **10%** | **8.8%** | **99.3%** |
| 25% | 12.9% | 99.3% |
| 50% | 16.4% | 99.3% |
| 75% | 16.6% | 99.3% |

An **89-point collapse** from a 10% behavioural change. The honest model does not move at all, because it was never given the crutch. This single table converts "leakage is a concern" into "here is the number."

### Shipping the weaker model

`predict.py` loads `fraud_model_honest.pkl` (PR-AUC 0.945), not `fraud_model.pkl` (PR-AUC 0.9999). The better-scoring model is retained for reference and the README explicitly says it *"should not be quoted as 'the' result."*

### Distinguishing ranking quality from probability quality

Precision, recall and PR-AUC only measure rank-ordering. `calibration.py` asked the separate question: does a score of 0.80 mean an 80% fraud rate? It did not — those transactions were fraud **6.6%** of the time, a 74-point gap. Isotonic regression on the validation set cut the Brier score from 0.00382 to 0.00058 (**84.7% better**), with the top bin moving from predicted ~98% / actual 72.9% to predicted ~100% / actual 99.8%.

The judgement here is knowing *when it matters*: the calibrated model is specified for any downstream use that multiplies probability by amount to estimate expected loss.

### Bounding claims

`bootstrap_ci.py` resamples the test set 1,000 times:

| Metric | Point estimate | 95% CI |
|---|---|---|
| Precision | 0.189 | [0.184, 0.194] |
| Recall | 0.993 | [0.991, 0.996] |
| PR-AUC | 0.945 | [0.940, 0.950] |

Tight intervals — but stating them rather than assuming them is the point.

### Testing its own blind spot

Supervised learning assumes labels exist. In production, chargebacks take weeks and undetected fraud is never labelled at all. `isolation_forest_baseline.py` trains with **zero labels** to probe that gap — and reports the result as weak, because it is (PR-AUC 0.125 against the supervised model's 0.945). The README says the quiet part out loud: *"Presenting a strong unsupervised number here would have been the easy, dishonest move — it isn't strong, and that's worth saying plainly."*

## 3.2 Judgement about the LLM

The assistant's docstring (`backend/routes/assistant.py`) contains the sharpest single statement of scope in the project:

> *"**Why not use an LLM as the ML model?** LLMs are not classifiers. They can't learn from tabular data, don't output calibrated probabilities, and can't be evaluated with precision/recall. The ML model does the prediction; the LLM does the explanation."*

Three concrete design choices follow from that:

1. **Grounding by construction.** `_load_context()` assembles a factual block from live database queries and the evaluation JSON — model metrics, order counts, risk distribution, top features, and the specific order if one is in scope. The system prompt instructs: *"Answer questions using ONLY the data provided below. Do not invent numbers. If data is not available, say so honestly."*
2. **Deterministic fallback.** With no API key, `_rule_based_response()` answers from the same context string. The feature degrades rather than disappearing, and the response labels its own source as `"rule-based fallback (no API key configured)"`.
3. **Right-sized model.** Claude Haiku 4.5, 512 max tokens, for a bounded summarisation task over pre-fetched context. No agentic loop where a template would do.

## 3.3 Where AI judgement is weakest

- **The assistant docstring says "(XGBoost)"** while the selected model is Logistic Regression. Stale comment; small, but it's in the file that best demonstrates judgement.
- **Sentinel receives none of the fraud project's rigour.** No adversarial test, no calibration check, no confidence intervals, no leakage audit — despite Sentinel being the system with the dashboard and the deployment story. The rigour lives in the project that isn't the product.
- **Model selection is F1-based while the whole thesis is that F1 is the wrong objective.** `train.py` picks the best model by validation F1, and only afterwards does `evaluate.py` sweep for the ₹-optimal threshold. Selecting the *model* on expected cost would make the argument end-to-end rather than bolted on at the final step.

---

# 4. Failure Recovery

> *What happened when something went wrong — and is the failure still visible in the record?*

Six failures, five recovered. All are documented in the repository rather than quietly deleted.

## 4.1 The leak (recovered — the headline)

**Failure:** the first fraud model scored a near-perfect PR-AUC 0.9999 by exploiting a generator artefact.
**Diagnosis:** `check_leakage.py` confirmed 99.2% of fraud rows end with `newbalanceOrig == 0`.
**Fix:** `train_ablation.py` retrains without `newbalanceOrig` / `errorBalanceOrig`.
**Verification:** `adversarial_test.py` quantified the difference — 89-point collapse vs. zero movement.
**Recovery shape:** the *worse* model was shipped, and the better one was kept in the repo, labelled as not-the-result. Both models and both sets of numbers remain visible.

## 4.2 The wrong fix, kept in the record (recovered — the most honest one)

**Failure:** the Isolation Forest baseline scored PR-AUC **0.035**, barely above random.
**First attempt:** feature scaling — the reflexive fix. It did nothing: **0.035 → 0.035**.
**Diagnosis:** Isolation Forest splits within each feature's own min/max range. Unlike a distance-based method, it is not scale-sensitive. The intuition was imported from KNN and did not apply.
**Actual fix:** more trees (200 → 500), a larger per-tree sample (`max_samples=0.5`), and `contamination='auto'` instead of forcing the known base rate → **PR-AUC 0.125**, a 3.6× improvement.
**Recovery shape:** the failed hypothesis is written into the README *with the reasoning for why it failed*. The tidy version of this story would have shown only the working fix. This one shows the wrong turn, which is the more useful artefact.

## 4.3 Overconfident probabilities (recovered)

**Failure:** the fix for class imbalance (`scale_pos_weight` on a 1:1225 ratio) silently broke probability semantics — scores near 0.80 corresponded to a 6.6% actual fraud rate.
**Why it was nearly missed:** every headline metric was fine. Precision, recall and PR-AUC measure ranking, and the ranking was good.
**Fix:** isotonic regression fitted on the validation set → Brier 0.00382 → 0.00058.
**Recovery shape:** shipped as a *separate* artefact (`fraud_model_honest_calibrated.pkl`) with documented guidance on when to use which — rather than silently swapping the default.

## 4.4 A feature the data couldn't support (recovered by redesign)

**Failure:** time-series spike alerts ("risk up 40% in the last 24h") were not buildable — the synthetic data has no real timestamps.
**Fix:** rather than fabricating a time axis or dropping the feature, the alerts were redesigned around a question the data *can* answer: where is return risk disproportionately concentrated right now, gated at a minimum sample of 30.
**Recovery shape:** the reasoning is preserved verbatim in the module docstring, so the constraint is legible to the next reader.

## 4.5 An unexpected result, diagnosed rather than hidden (recovered)

**Failure:** Logistic Regression beat Random Forest and Gradient Boosting on Sentinel — the opposite of the expected outcome, and the kind of result that invites quiet re-running until the "right" model wins.
**Diagnosis:** the synthetic generator produces largely linear relationships, so the linear model matches the generative process.
**Recovery shape:** documented as a limitation — *"Logistic Regression won on this linearly-generated data; XGBoost typically wins on real messy data"* — and the selection logic was left automatic, so real data would change the answer without a code edit.

## 4.6 Documentation drift (NOT recovered — open)

**Failure:** two inconsistencies survive in the repo today (a third, suspected one turned out to be a false alarm — see below):

| # | Problem | Detail |
|---|---|---|
| 1 | `docs/interview-questions.md` names the wrong model | Both prepared pitches say *"a Gradient Boosting model"*; `training_metadata.json` and the README say Logistic Regression was selected |
| 2 | Stale comment in `assistant.py` | Docstring says the prediction is made by *"(XGBoost)"* |

**The false alarm, worth recording.** The README (P 0.477 / R 0.698 / F1 0.567) and `training_metadata.json` (0.473 / 0.668 / 0.554) appear to disagree about the same model. They do not. `training_metadata.json` stores **validation-set** metrics — the numbers model selection was made on — while the README and the live `/api/metrics` endpoint report **held-out test-set** metrics. `evaluation_results.json` contains both, under `val_metrics` and `test_metrics`, and they match their respective sources exactly. Two different splits, correctly separated. **Quote the test numbers (0.477 / 0.698 / 0.567); they are the honest, held-out ones.**

**Root cause:** the code has one source of truth for status, one for cost and one for metrics — but the *prose* has none. Every document was written by hand against a model that later changed.

**The honest note:** listing this here is itself the recovery step. A project that claims "every number comes from the real model" and then contradicts itself in its own prep notes has a credibility problem precisely proportional to how loudly it claims rigour. Fix #1 before presenting — rehearsing from that file means stating aloud a fact the repository disproves.

## 4.7 The pattern across all six

| Failure | Signal | Response | Left visible? |
|---|---|---|---|
| Data leakage | Score too good | Retrained without the feature; proved it adversarially | Both models kept |
| Weak unsupervised baseline | Score too bad | Wrong fix tried, diagnosed, right fix found | Wrong turn documented |
| Miscalibration | No signal in headline metrics | Isotonic recalibration | Separate artefact + guidance |
| Unbuildable feature | Data lacks a time axis | Redesigned around what the data supports | Reasoning in docstring |
| Unexpected model winner | Contradicted expectation | Diagnosed as a data property | Named as a limitation |
| Doc drift | Found on review | **Open** | Listed here |

The consistent behaviour: **the failure stays in the record**. The leaky model is still in `models/`. The failed scaling experiment is still in the README. The weak unsupervised number is quoted rather than buried. That is the difference between recovering from a failure and erasing one.

---

# Honest summary by criterion

| Criterion | Strongest evidence | Weakest point |
|---|---|---|
| **Problem taste** | Cost-asymmetric thresholding as load-bearing code — F1-optimal vs ₹-optimal differ, and the gap is ₹110.5M on the fraud side | Both datasets synthetic; all three cost constants are placeholders |
| **Build quality** | Three deliberate single-sources-of-truth (status, cost, metrics); temporal split; 21 tests | Hardcoded absolute paths; zero tests on the fraud project; no CI; API key committed |
| **AI judgement** | Treated PR-AUC 0.9999 as a red flag, proved the leak, shipped the weaker model; LLM scoped to explanation only, grounded and fallback-backed | Sentinel gets none of that rigour; model selection still optimises F1 |
| **Failure recovery** | The Isolation Forest wrong-turn documented *with* the reasoning; the leaky model kept in-repo and labelled | Documentation drift, currently unresolved |

**The single sentence:** this is not a project about a model. It is a project about knowing what a model's number is worth — and the strongest evidence for that is that it shipped the model with the worse score, on purpose, and wrote down why.

---

> ⚠️ **Before this repository goes anywhere public:** `.mcp.json` contains a live Google API key in plaintext (`X-Goog-Api-Key`). Rotate that key and move it to an environment variable — `.env.example` already establishes the pattern the rest of the project follows. `ai-risk-manager/.env` should also be confirmed as excluded before any push.

---

*Evidence for every claim above is drawn from the repository as it stands: `ai-risk-manager/` and `fraud-detector/`. Model metrics quoted throughout are held-out **test-set** figures, verified against the live `/api/metrics` endpoint.*

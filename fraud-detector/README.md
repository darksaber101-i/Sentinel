# Fraud Detector (PaySim)

A transaction fraud **detector/verifier** trained on the PaySim mobile-money
dataset (`PS_20174392719_1491204439457_log.csv`, 6.36M transactions, 8,213
fraud cases — 0.13%).

**Defense-only, by construction:** this is a classifier that scores a
transaction and prints a probability + flag/no-flag verdict. It has no
network access, cannot move money, execute transactions, or interact with
any live system. There is nothing here that could be repurposed to commit
or evade fraud — it only ever runs in the direction of "is this suspicious."

## Honest metrics — read this before trusting any number

Two models were trained. The headline one is intentionally the *weaker*
of the two, because the stronger one is inflated by a data artifact.

### Why there are two models

PaySim's synthetic fraud generator **always fully drains the origin
account** when it simulates a fraud row (99.2% of fraud rows end with
`newbalanceOrig == 0`, confirmed in `check_leakage.py`). A model trained
with `newbalanceOrig`/`errorBalanceOrig` picks up on this and reaches
**precision=0.999, recall=1.000, PR-AUC≈1.0** on the held-out test set —
which is not a real fraud-detection result, it's the model learning
"was this account fully drained," a proxy for the label that a real
fraudster isn't guaranteed to leave behind.

| Model | Features | Test PR-AUC | Precision @ F1-optimal | Recall @ F1-optimal |
|---|---|---|---|---|
| Full (`fraud_model.pkl`) | includes origin-balance-drain signal | 0.9999 | 1.000 | 0.999 |
| **Honest (`fraud_model_honest.pkl`)** — **used by `predict.py`** | excludes it | **0.9449** | **0.979** | **0.838** |

The honest model is what `predict.py` uses. The full model is kept for
reference (`models/fraud_model.pkl`, `models/evaluation_results.json`) but
should not be quoted as "the" result.

### Held-out test set

**Temporal split**, not random: train on the earliest 70% of time steps,
validate on the next 10%, test on the final 20% (step > 594 of 743) —
1,248,736 transactions, 4,250 of them fraud, **never seen during training
or model/threshold selection**. A random shuffle would leak future fraud
patterns into training on this dataset (fraud rate varies over time:
0.08% in train, 0.34% in test) and overstate performance.

### Results on the held-out test set (honest model)

| Threshold | Precision | Recall | F1 | Meaning |
|---|---|---|---|---|
| 0.5 (default) | 0.392 | 0.976 | 0.559 | naive threshold — high recall, weak precision |
| **0.989 (F1-optimal)** | **0.979** | **0.838** | **0.903** | best symmetric trade-off |
| **0.090 (cost-optimal)** | **0.189** | **0.993** | 0.316 | minimizes ₹ lost, see below |

### False-positive cost — the number that actually matters

- **False positive cost** (assumption, labeled as such): ₹50 flat per
  transaction manually reviewed. This is not in the data — it's a
  stand-in for real ops cost and should be replaced with an actual number
  before using this for a real decision.
- **False negative cost** (exact, not assumed): the transaction's real
  `amount` — money actually lost to fraud that slipped through.
- Average fraud transaction in the test set: ~₹1.57M. Average review cost:
  ₹50. **A missed fraud costs ~31,000x more than a false-positive review.**

That asymmetry is why the cost-optimal threshold (0.090) looks nothing
like the F1-optimal one (0.989): it deliberately tanks precision to 18.9%
to push recall to 99.3%, because 6,444 unnecessary ₹50 reviews
(₹322K) is cheap next to even one more missed multi-lakh fraud. Optimizing
for F1 instead — a metric that doesn't know about money — would leave
recall at 83.8%, i.e. miss ~16% of fraud, to keep precision "high."

| Threshold | TP | FP | FN | Fraud ₹ missed | Review ₹ cost | Total cost |
|---|---|---|---|---|---|---|
| 0.5 (default) | 4,148 | 6,444 | 102 | ₹6,988,624 | ₹322,200 | ₹7,310,824 |
| 0.989 (F1-optimal) | 3,562 | 77 | 688 | ₹110,511,888 | ₹3,850 | ₹110,515,738 |
| **0.090 (cost-optimal)** | **4,222** | **18,127** | **28** | **₹1,123,420** | **₹906,350** | **₹2,029,770** |
| — (flag nothing) | 0 | 0 | 4,250 | ₹6,682,882,048 | ₹0 | ₹6,682,882,048 |

Net savings at the cost-optimal threshold vs. flagging nothing: **₹6,680,852,278 (99.97%)**.

The F1-optimal threshold looks "safer" (98% precision) but leaves **₹110.5M**
of fraud on the table — 54x more loss than the cost-optimal threshold,
because it misses 688 fraud cases (16% of all fraud) chasing a
precision/recall balance that has no concept of money. This is the concrete
version of "optimizing a symmetric ML metric leaves real money on the
table": F1 says 0.903 is worse than nothing to write home about next to
0.989's precision, but in ₹ terms the cost-optimal threshold is dramatically
better despite an F1 of only 0.317.

*(The 99.97% savings figure is technically correct but is driven by a small
number of very large fraud transactions in the baseline sum — don't read it
as "this model prevents fraud losses in general." The absolute ₹2.03M
residual cost at the cost-optimal threshold is the more honest number to
anchor on.)*

## Beyond the baseline: five things most PaySim projects skip

### 1. Adversarial robustness — does it survive fraud that evolves?

The whole "honest model" story above raises an obvious follow-up: if the
full model is bad because it relies on the full-drain artifact, does the
*honest* model actually hold up if a fraudster stops fully draining the
account? Tested it directly (`adversarial_test.py`) by re-simulating the
4,250 real test-set fraud cases at different levels of balance left behind:

| Balance left behind | Full model recall | Honest model recall |
|---|---|---|
| 0% (PaySim's original behavior) | 97.5% | 99.3% |
| 10% | **8.8%** | 99.3% |
| 25% | 12.9% | 99.3% |
| 50% | 16.4% | 99.3% |
| 75% | 16.6% | 99.3% |

The full model **collapses 89 percentage points** the moment a fraudster
leaves even 10% of the balance behind — it was never detecting fraud, it
was detecting "was this account drained to exactly zero." The honest model
doesn't move at all, because it was never given that crutch to lean on.
This is the single strongest piece of evidence in this project for why the
leakage caveat isn't pedantic — it's the difference between a detector
that works and one that only *looks* like it works.

### 2. Per-prediction explainability

`predict.py` and `api.py` now return SHAP-based reasons alongside every
verdict, e.g.:
```
Why (top contributing factors):
  is a CASH_OUT = 0  ->  decreased fraud score by 4.597
  transaction amount = 900000  ->  increased fraud score by 3.621
  hour of day = 3  ->  increased fraud score by 2.836
```
A bare probability with no reasoning is the single most common complaint
against ML fraud systems in practice ("why was my account flagged").

### 3. Calibration — does "70% probability" mean 70% fraud rate?

Precision/recall only care about rank-ordering, not whether the probability
value itself is trustworthy. Checked with `calibration.py`: the raw model
is badly **overconfident** in the middle of its range — e.g. transactions
scored ~0.80 were actually fraud only 6.6% of the time (a 74-point gap),
and even its most confident bin (0.9–1.0) overstates: predicted ~98%,
actual 72.9%. This is a known side effect of `scale_pos_weight` (needed to
handle the 1:1225 class imbalance) — it helps ranking, distorts the
probability's face value. Isotonic recalibration on the validation set
fixed this: Brier score improved from 0.00382 to 0.00058 (**84.7% better**),
and the top bin now reads predicted≈100%, actual≈99.8%. Saved as
`models/fraud_model_honest_calibrated.pkl` — use this one if anything
downstream needs the probability value itself to be meaningful (e.g.
multiplying it by transaction amount to estimate expected loss).

### 4. Per-segment thresholds

One global threshold (0.090) is not necessarily optimal for both fraud
types. Checked with `segment_thresholds.py`: TRANSFER fraud turns out to be
nearly perfectly separable — its own optimal threshold (0.850) reduces its
cost to essentially ₹100 — while CASH_OUT is where nearly all the remaining
cost lives regardless of threshold. Splitting thresholds by type saves a
real but modest ₹7,500 (0.4%) over the single global threshold. Small
improvement, but it surfaces a genuinely useful operational insight:
**TRANSFER fraud is basically solved; CASH_OUT is where investigation effort
should go.**

### 5. Does it decay over time without retraining?

Trained once on week 1 only, then scored every subsequent week without any
retraining, using a threshold fixed once and never re-tuned
(`decay_over_time.py`):

| Week | PR-AUC | Recall |
|---|---|---|
| 2 | 0.862 | 98.2% |
| 3 | 0.909 | 98.4% |
| 4 | 0.947 | 98.5% |
| 5 | 0.954 | 97.9% |

No decay — performance actually *improved*, because fraud becomes
relatively more concentrated later in this dataset (recall stayed
essentially flat at ~98% throughout, which is the more meaningful number
here). **Caveat:** this only tests drift in *volume/rate*, using the same
kind of fraud each time — it says nothing about fraud *tactics* evolving,
which is exactly what test #1 (adversarial robustness) is for. Decay
analysis and adversarial testing answer different questions; neither
substitutes for the other.

### 6. Confidence intervals on the headline numbers

A point estimate like "precision=0.189" implies more certainty than 4,250
fraud cases support. `bootstrap_ci.py` resamples the test set 1,000 times
for 95% confidence intervals at the cost-optimal threshold (0.090):

| Metric | Point estimate | 95% CI |
|---|---|---|
| Precision | 0.189 | [0.184, 0.194] |
| Recall | 0.993 | [0.991, 0.996] |
| F1 | 0.318 | [0.311, 0.325] |
| PR-AUC | 0.945 | [0.940, 0.950] |

Tight intervals (±0.5pp on recall, ±0.5pp on precision) — the fraud count
here (4,250) is large enough that these numbers aren't noise, which is
itself worth stating rather than assuming.

### 7. Unsupervised baseline (Isolation Forest)

Supervised learning assumes labels exist. In production, fraud labels are
often delayed (a chargeback takes weeks) or simply missing for undetected
fraud. Trained an Isolation Forest on the same data with **zero fraud
labels** (`isolation_forest_baseline.py`) as a check on that blind spot.

Reported honestly rather than talked up: it's weak. First attempt (default
settings, contamination set to the known base rate) scored **PR-AUC
0.035** — barely above random. Tried scaling the features, expecting that
to fix it — it didn't (0.035 → 0.035, negligible), which makes sense in
hindsight: Isolation Forest splits within each feature's own min/max range,
so unlike KNN it isn't distance-based and isn't scale-sensitive. What
actually helped was more trees (200→500), a larger per-tree sample
(`max_samples=0.5`), and `contamination='auto'` instead of forcing the
known rate: **PR-AUC 0.035 → 0.125** (3.6x), precision 0.162 / recall
0.316 at its own cutoff.

Still nowhere near the supervised model's 0.945 PR-AUC — expected, since
it never sees a single label. The honest conclusion: on this dataset,
label-free anomaly detection is a weak primary detector but could serve as
a secondary signal for exactly the blind spot supervised learning has
(catching a fraud typology no labeled example ever showed the model).
Presenting a strong unsupervised number here would have been the easy,
dishonest move — it isn't strong, and that's worth saying plainly.

### 8. Live API

`api.py` (FastAPI) wraps the same scoring logic as `predict.py` behind
`POST /score`, returning probability + verdicts at all three thresholds +
SHAP explanation — same defense-only scope, just callable over HTTP instead
of the CLI. Interactive docs at `/docs` once running.

## Files

| File | Purpose |
|---|---|
| `explore.py` | Data sanity checks (fraud rate, type breakdown, temporal split balance) |
| `check_leakage.py` | Confirms the origin-balance-drain artifact and feature importances |
| `train.py` | Trains the full-feature model (kept for reference, not recommended) |
| `train_ablation.py` | Trains the honest model used by `predict.py` |
| `adversarial_test.py` | Stress-tests both models against fraud that doesn't fully drain the account |
| `explain.py` | SHAP-based per-prediction explanations |
| `calibration.py` | Checks + fixes probability calibration (isotonic regression) |
| `segment_thresholds.py` | Per-transaction-type cost-optimal thresholds |
| `decay_over_time.py` | Week-by-week performance with no retraining, to check for drift |
| `bootstrap_ci.py` | 95% confidence intervals on the headline metrics |
| `isolation_forest_baseline.py` | Unsupervised (no-labels) anomaly-detection comparison |
| `predict.py` | **Interactive/CLI verifier — input your own transaction, get a probability + verdict + reasons** |
| `api.py` | FastAPI wrapper exposing the same scoring as an HTTP endpoint |
| `models/` | Saved models, scaler, feature lists, evaluation JSON for all of the above |

## Usage

```bash
# Interactive CLI
py -3 predict.py

# Non-interactive CLI
py -3 predict.py --type TRANSFER --amount 900000 \
    --old-balance-orig 900000 --old-balance-dest 0 --new-balance-dest 900000 --hour 3

# HTTP API
py -3 -m uvicorn api:app --port 8010
# then: POST http://127.0.0.1:8010/score  (see /docs for interactive schema)
```

## Known limitations

1. Synthetic data (PaySim) — a real deployment needs validation on real
   transaction data before the specific thresholds/costs are trusted.
2. `nameOrig`/`nameDest` (account IDs) were deliberately excluded as
   features — they're high-cardinality identifiers that would not
   generalize to new accounts and would make the evaluation dishonest.
3. The ₹50 false-positive cost is a placeholder assumption, not a real
   ops number — swap it in `train_ablation.py`/`predict.py` for your
   actual review cost.
4. `isFlaggedFraud`, PaySim's own built-in naive rule, catches only 16 of
   8,213 frauds (0.2% recall) — included in the data as a baseline to beat,
   not used as a feature.

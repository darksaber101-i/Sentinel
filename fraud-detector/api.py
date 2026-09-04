"""
Minimal FastAPI wrapper around the fraud detector.

Defense-only: this service only scores a transaction description POSTed to
it and returns a probability + verdict + explanation. It has no endpoint
that executes, holds, or modifies a transaction, no database, no ability
to reach any payment rail. It is read-in, score-out.

Run:
  py -3 -m uvicorn api:app --reload --port 8010

Then open http://127.0.0.1:8010/docs for interactive API docs, or:
  curl -X POST http://127.0.0.1:8010/score -H "Content-Type: application/json" -d "{\"type\":\"TRANSFER\",\"amount\":900000,\"old_balance_orig\":900000,\"old_balance_dest\":0,\"new_balance_dest\":900000,\"hour\":3}"
"""
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import load_model, score, THRESHOLDS
from explain import explain_row

app = FastAPI(
    title="Fraud Detector API",
    description="Defense-only transaction fraud scoring. Scores only -- no execution capability.",
    version="1.0.0",
)

_model, _features = None, None


@app.on_event("startup")
def _startup():
    global _model, _features
    _model, _features = load_model()


class TransactionIn(BaseModel):
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(gt=0)
    old_balance_orig: float = Field(ge=0, description="Sender's balance before this transaction")
    old_balance_dest: float = Field(ge=0, default=0.0, description="Recipient's balance before (0 if merchant/unknown)")
    new_balance_dest: float = Field(ge=0, default=0.0, description="Recipient's balance after (0 if merchant/unknown)")
    hour: int = Field(ge=0, le=23, default=12)


class ScoreOut(BaseModel):
    fraud_probability: float
    verdict_f1_threshold: bool
    verdict_cost_threshold: bool
    verdict_default_threshold: bool
    recommended_verdict: bool
    top_factors: list[str]
    note: str


@app.get("/")
def root():
    return {
        "service": "fraud-detector",
        "scope": "defense-only: scores transactions, never executes them",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/score", response_model=ScoreOut)
def score_transaction(txn: TransactionIn):
    prob, X = score(_model, _features, txn.type, txn.amount, txn.old_balance_orig,
                     txn.old_balance_dest, txn.new_balance_dest, txn.hour)
    factors = explain_row(X)
    return ScoreOut(
        fraud_probability=prob,
        verdict_f1_threshold=prob >= THRESHOLDS["f1"],
        verdict_cost_threshold=prob >= THRESHOLDS["cost"],
        verdict_default_threshold=prob >= THRESHOLDS["default"],
        recommended_verdict=prob >= THRESHOLDS["cost"],
        top_factors=factors,
        note=("recommended_verdict uses the cost-optimal threshold (0.090): on the held-out "
              "test set, missed fraud costs ~31,000x more on average than a false-positive "
              "review, so it deliberately favors recall over precision."),
    )

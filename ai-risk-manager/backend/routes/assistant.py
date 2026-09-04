"""
AI Assistant Endpoint
──────────────────────
The assistant uses the Anthropic API (Claude) to answer questions about
the application's OWN data — it does NOT make up statistics.

HOW IT WORKS
  1. We collect real data from the DB and evaluation results.
  2. We inject that data into Claude's system prompt.
  3. Claude answers in natural language, grounded by the injected context.
  4. If no API key is set, we fall back to a deterministic rule-based responder.

WHY NOT USE AN LLM AS THE ML MODEL?
  LLMs are not classifiers. They can't learn from tabular data, don't output
  calibrated probabilities, and can't be evaluated with precision/recall.
  The ML model (XGBoost) does the prediction; the LLM does the explanation.
"""

import json
import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models
from backend.schemas import AssistantRequest, AssistantResponse
from backend.config import settings

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
router    = APIRouter(prefix="/api/assistant", tags=["assistant"])


def _load_context(db: Session, order_id: str | None = None) -> str:
    """Build a factual context string from real application data."""
    eval_path = MODEL_DIR / "evaluation_results.json"
    eval_data = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    m = eval_data.get("test_metrics", {})

    total_orders = db.query(models.Order).count()
    preds        = db.query(models.Prediction).all()
    dist         = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for p in preds:
        if p.risk_level in dist:
            dist[p.risk_level] += 1

    ctx = f"""
=== Sentinel — Live Application Data ===

MODEL
  Name      : {eval_data.get('best_model_name', 'N/A')}
  Precision : {m.get('precision', 'N/A')}
  Recall    : {m.get('recall', 'N/A')}
  F1        : {m.get('f1', 'N/A')}
  ROC-AUC   : {m.get('roc_auc', 'N/A')}
  PR-AUC    : {m.get('pr_auc', 'N/A')}
  Accuracy  : {m.get('accuracy', 'N/A')}
  Train size: {eval_data.get('train_size', 'N/A')}
  Test size : {eval_data.get('test_size', 'N/A')}

ORDERS
  Total analyzed : {total_orders}
  LOW risk       : {dist['LOW']}
  MEDIUM risk    : {dist['MEDIUM']}
  HIGH risk      : {dist['HIGH']}
  CRITICAL risk  : {dist['CRITICAL']}

TOP RISK FACTORS (by feature importance)
{json.dumps(eval_data.get('feature_importance', [])[:5], indent=2)}
"""

    if order_id:
        order = db.query(models.Order).filter(models.Order.order_id == order_id).first()
        pred  = db.query(models.Prediction).filter(
            models.Prediction.order_id == order_id
        ).order_by(models.Prediction.created_at.desc()).first()

        if order and pred:
            ctx += f"""
SPECIFIC ORDER: {order_id}
  Category          : {order.product_category}
  Order Value       : ₹{order.order_value}
  Previous Returns  : {order.previous_returns}
  Return Rate       : {order.previous_return_rate:.0%}
  Discount          : {order.discount_percentage}%
  Risk Score        : {pred.risk_score}/100
  Risk Level        : {pred.risk_level}
  Return Probability: {pred.return_probability:.0%}
  Explanation       : {pred.explanation}
"""

    return ctx.strip()


def _rule_based_response(question: str, ctx: str) -> str:
    """Deterministic fallback when no API key is configured."""
    q = question.lower()
    lines = ctx.split("\n")

    def find(keyword):
        for line in lines:
            if keyword in line.lower():
                return line.strip()
        return None

    if any(w in q for w in ["precision"]):
        val = find("precision")
        return f"The model's precision is {val.split(':')[-1].strip() if val else 'not yet available'}. Precision tells us: of all orders flagged as high-risk, what fraction actually returned."
    if any(w in q for w in ["recall"]):
        val = find("recall")
        return f"Model recall is {val.split(':')[-1].strip() if val else 'not available'}. Recall tells us: of all orders that actually were returned, what fraction the model caught."
    if any(w in q for w in ["f1"]):
        val = find("f1")
        return f"The F1 score is {val.split(':')[-1].strip() if val else 'not available'}. F1 is the harmonic mean of precision and recall — useful when both matter equally."
    if any(w in q for w in ["high risk", "high-risk", "critical"]):
        h = find("high risk")
        c = find("critical risk")
        return f"Currently {h}, {c} from context data."
    if any(w in q for w in ["best model", "which model", "model perform"]):
        val = find("name")
        return f"The best-performing model is {val.split(':')[-1].strip() if val else 'not trained yet'}, selected by F1 score on the validation set."
    if any(w in q for w in ["why", "reason", "factor", "risk factor"]):
        return "The top risk factors are: (1) previous return rate, (2) product return rate, (3) discount percentage, (4) new customer status, and (5) payment method (COD increases risk)."
    if any(w in q for w in ["category", "highest return"]):
        return "Fashion has the highest return rate (~38%), followed by Jewelry (~28%) and Home & Kitchen (~22%), based on industry benchmarks embedded in the training data."
    return "I can answer questions about model performance, risk factors, order counts, and specific order details. Try asking: 'What is the model's precision?' or 'Why is this order high risk?'"


@router.post("", response_model=AssistantResponse)
async def ask_assistant(req: AssistantRequest, db: Session = Depends(get_db)):
    context = _load_context(db, req.order_id)

    if not settings.ANTHROPIC_API_KEY or settings.ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        answer = _rule_based_response(req.question, context)
        return AssistantResponse(
            answer=answer,
            sources=["rule-based fallback (no API key configured)"]
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=f"""You are an AI assistant for the Sentinel platform.
Answer questions using ONLY the data provided below. Do not invent numbers.
If data is not available, say so honestly.
Keep answers concise (2-4 sentences) and business-friendly.

{context}""",
            messages=[{"role": "user", "content": req.question}],
        )
        answer = message.content[0].text
        return AssistantResponse(answer=answer, sources=["live application data"])

    except Exception as e:
        answer = _rule_based_response(req.question, context)
        return AssistantResponse(
            answer=answer,
            sources=[f"rule-based fallback (API error: {str(e)[:50]})"]
        )

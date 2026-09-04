"""Tests for prediction and risk scoring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.predict import probability_to_risk, predict


def test_risk_levels():
    assert probability_to_risk(0.10)["risk_level"] == "LOW"
    assert probability_to_risk(0.40)["risk_level"] == "MEDIUM"
    assert probability_to_risk(0.70)["risk_level"] == "HIGH"
    assert probability_to_risk(0.85)["risk_level"] == "CRITICAL"


def test_risk_score_range():
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = probability_to_risk(p)
        assert 0 <= r["risk_score"] <= 100


def test_prediction_label():
    low = probability_to_risk(0.10)
    assert "Unlikely" in low["prediction"]
    high = probability_to_risk(0.85)
    assert "Return" in high["prediction"]


SAMPLE_ORDER = {
    "order_value": 8499, "quantity": 2, "discount_percentage": 25,
    "customer_tenure_days": 120, "previous_orders": 8, "previous_returns": 4,
    "previous_return_rate": 0.38, "previous_failed_payments": 0,
    "previous_chargebacks": 0, "payment_method": "COD", "device_type": "mobile",
    "shipping_distance_km": 450, "delivery_days": 5, "is_new_customer": 0,
    "is_first_order": 0, "order_hour": 14, "days_since_last_order": 30,
    "product_return_rate": 0.31, "support_tickets": 2, "product_category": "Fashion",
}


def test_predict_output_keys():
    result = predict(SAMPLE_ORDER)
    assert "return_probability" in result
    assert "risk_score" in result
    assert "risk_level" in result
    assert "prediction" in result


def test_predict_probability_range():
    result = predict(SAMPLE_ORDER)
    assert 0.0 <= result["return_probability"] <= 1.0


def test_high_risk_order_flagged():
    """A customer with 80% historical return rate should be flagged as high risk."""
    risky = dict(SAMPLE_ORDER)
    risky["previous_return_rate"] = 0.80
    risky["previous_returns"]     = 16
    result = predict(risky)
    assert result["risk_level"] in ("HIGH", "CRITICAL"), \
        f"Expected HIGH/CRITICAL, got {result['risk_level']}"


def test_low_risk_order_not_flagged():
    """Loyal customer with zero returns should be LOW risk."""
    safe = dict(SAMPLE_ORDER)
    safe["previous_return_rate"] = 0.0
    safe["previous_returns"]     = 0
    safe["customer_tenure_days"] = 1000
    safe["payment_method"]       = "UPI"
    safe["product_category"]     = "Books"
    safe["product_return_rate"]  = 0.07
    safe["is_new_customer"]      = 0
    result = predict(safe)
    assert result["risk_level"] in ("LOW", "MEDIUM"), \
        f"Expected LOW/MEDIUM, got {result['risk_level']}"

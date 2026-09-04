"""API endpoint integration tests (requires backend running on port 8000)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import os; os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent.parent))
    from backend.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "precision" in data
    assert "recall" in data
    assert 0 <= data["precision"] <= 1
    assert 0 <= data["recall"] <= 1


def test_predict_endpoint(client):
    payload = {
        "product_category": "Fashion",
        "order_value": 5000, "quantity": 2, "discount_percentage": 30,
        "customer_tenure_days": 100, "previous_orders": 5, "previous_returns": 3,
        "previous_return_rate": 0.6, "previous_failed_payments": 0,
        "previous_chargebacks": 0, "payment_method": "COD", "device_type": "mobile",
        "shipping_distance_km": 300, "delivery_days": 4, "is_new_customer": False,
        "is_first_order": False, "order_hour": 14, "days_since_last_order": 20,
        "product_return_rate": 0.38, "support_tickets": 1,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "return_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_risk_distribution(client):
    r = client.get("/api/risk-distribution")
    assert r.status_code == 200
    data = r.json()
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        assert level in data


def test_threshold_analysis(client):
    r = client.get("/api/threshold-analysis")
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 5
    assert "threshold" in data[0]
    assert "precision" in data[0]
    assert "recall" in data[0]


def test_cost_analysis(client):
    r = client.get("/api/cost-analysis")
    assert r.status_code == 200
    data = r.json()
    assert "assumptions" in data
    assert "best_threshold" in data
    assert len(data["rows"]) > 5


def test_alerts(client):
    r = client.get("/api/alerts")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    for a in data:
        assert a["severity"] in ("LOW", "MEDIUM", "HIGH")
        assert "title" in a and "detail" in a


def test_review_queue(client):
    r = client.get("/api/orders/review-queue")
    assert r.status_code == 200
    data = r.json()
    assert "orders" in data and "total" in data
    if data["orders"]:
        row = data["orders"][0]
        assert row["risk_level"] in ("HIGH", "CRITICAL")
        assert "cost_at_stake" in row
        assert row["review_status"] == "PENDING"


def test_order_action_flow(client):
    # Grab a real order_id from the queue to act on
    queue = client.get("/api/orders/review-queue").json()
    assert queue["orders"], "expected at least one HIGH/CRITICAL order to act on"
    order_id = queue["orders"][0]["order_id"]

    r = client.post(f"/api/orders/{order_id}/action", json={"action": "hold", "note": "verifying address"})
    assert r.status_code == 200
    body = r.json()
    assert body["review_status"] == "HELD"

    order = client.get(f"/api/orders/{order_id}").json()
    assert order["review_status"] == "HELD"
    assert order["last_action_note"] == "verifying address"

    history = client.get(f"/api/orders/{order_id}/actions").json()
    assert len(history) >= 1
    assert history[0]["status"] == "HELD"

    # It should have dropped out of the PENDING queue
    queue_after = client.get("/api/orders/review-queue").json()
    assert order_id not in [o["order_id"] for o in queue_after["orders"]]

    # Invalid action should 400
    bad = client.post(f"/api/orders/{order_id}/action", json={"action": "reject"})
    assert bad.status_code == 400

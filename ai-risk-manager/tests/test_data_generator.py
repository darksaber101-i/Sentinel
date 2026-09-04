"""Tests for synthetic data generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_generator import generate_dataset


def test_row_count():
    df = generate_dataset(n=500, seed=1)
    assert len(df) == 500


def test_columns_present():
    df = generate_dataset(n=100, seed=2)
    required = [
        "order_id", "customer_id", "product_id", "product_category",
        "order_value", "quantity", "discount_percentage",
        "previous_return_rate", "payment_method", "is_returned",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_return_rate_realistic():
    df = generate_dataset(n=5000, seed=3)
    rate = df["is_returned"].mean()
    # Expect 15–50% return rate (realistic range)
    assert 0.15 <= rate <= 0.50, f"Unrealistic return rate: {rate:.1%}"


def test_no_negative_values():
    df = generate_dataset(n=1000, seed=4)
    assert (df["order_value"] > 0).all()
    assert (df["quantity"] > 0).all()
    assert (df["previous_returns"] >= 0).all()


def test_return_rate_correlation():
    """Orders with high previous_return_rate should return more often."""
    df = generate_dataset(n=3000, seed=5)
    high = df[df["previous_return_rate"] > 0.5]["is_returned"].mean()
    low  = df[df["previous_return_rate"] < 0.1]["is_returned"].mean()
    assert high > low, "High return-rate customers should return more"

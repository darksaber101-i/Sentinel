"""Pydantic schemas — request/response shapes for the API."""

from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


# ── Order ─────────────────────────────────────────────────────────────────────

class OrderBase(BaseModel):
    order_id:                  str
    customer_id:               str
    product_id:                str
    product_category:          str
    order_value:               float
    quantity:                  int
    discount_percentage:       float
    customer_tenure_days:      int
    previous_orders:           int
    previous_returns:          int
    previous_return_rate:      float
    previous_failed_payments:  int
    previous_chargebacks:      int
    payment_method:            str
    device_type:               str
    shipping_distance_km:      float
    delivery_days:             int
    is_new_customer:           bool
    is_first_order:            bool
    order_hour:                int
    days_since_last_order:     int
    product_return_rate:       float
    support_tickets:           int
    is_returned:               Optional[int] = None


class OrderResponse(OrderBase):
    return_probability: Optional[float] = None
    risk_score:         Optional[int]   = None
    risk_level:         Optional[str]   = None
    prediction:         Optional[str]   = None
    top_features:       Optional[List]  = None
    explanation:        Optional[str]   = None
    review_status:      str             = "PENDING"
    last_action_note:   Optional[str]   = None

    model_config = {"from_attributes": True}


class OrderListResponse(BaseModel):
    orders:      List[OrderResponse]
    total:       int
    page:        int
    page_size:   int
    risk_summary: dict


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    order_id:                  Optional[str] = None
    customer_id:               Optional[str] = None
    product_id:                Optional[str] = None
    product_category:          str
    order_value:               float
    quantity:                  int
    discount_percentage:       float
    customer_tenure_days:      int
    previous_orders:           int
    previous_returns:          int
    previous_return_rate:      float
    previous_failed_payments:  int = 0
    previous_chargebacks:      int = 0
    payment_method:            str
    device_type:               str
    shipping_distance_km:      float
    delivery_days:             int
    is_new_customer:           bool
    is_first_order:            bool
    order_hour:                int
    days_since_last_order:     int
    product_return_rate:       float
    support_tickets:           int
    threshold:                 float = 0.5


class PredictResponse(BaseModel):
    order_id:           Optional[str]
    return_probability: float
    risk_score:         int
    risk_level:         str
    prediction:         str
    top_features:       List[dict]
    explanation:        str


class BatchPredictRequest(BaseModel):
    orders:    List[PredictRequest]
    threshold: float = 0.5


# ── Metrics ───────────────────────────────────────────────────────────────────

class ModelPerformanceResponse(BaseModel):
    best_model_name: str
    train_size:      int
    val_size:        int
    test_size:       int
    test_metrics:    dict
    val_metrics:     dict
    all_model_test_metrics: dict
    threshold_analysis: List[dict]
    feature_importance: List[dict]


# ── AI Assistant ──────────────────────────────────────────────────────────────

class AssistantRequest(BaseModel):
    question:  str
    order_id:  Optional[str] = None


class AssistantResponse(BaseModel):
    answer:    str
    sources:   List[str] = []

"""SQLAlchemy ORM models — defines database tables."""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, JSON, Text
from backend.database import Base


class Order(Base):
    __tablename__ = "orders"

    id                       = Column(Integer, primary_key=True, index=True)
    order_id                 = Column(String, unique=True, index=True)
    customer_id              = Column(String, index=True)
    product_id               = Column(String)
    product_category         = Column(String)
    order_value              = Column(Float)
    quantity                 = Column(Integer)
    discount_percentage      = Column(Float)
    customer_tenure_days     = Column(Integer)
    previous_orders          = Column(Integer)
    previous_returns         = Column(Integer)
    previous_return_rate     = Column(Float)
    previous_failed_payments = Column(Integer)
    previous_chargebacks     = Column(Integer)
    payment_method           = Column(String)
    device_type              = Column(String)
    shipping_distance_km     = Column(Float)
    delivery_days            = Column(Integer)
    is_new_customer          = Column(Boolean)
    is_first_order           = Column(Boolean)
    order_hour               = Column(Integer)
    days_since_last_order    = Column(Integer)
    product_return_rate      = Column(Float)
    support_tickets          = Column(Integer)
    is_returned              = Column(Integer, nullable=True)   # actual outcome
    created_at               = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, index=True)
    order_id            = Column(String, index=True)
    return_probability  = Column(Float)
    risk_score          = Column(Integer)
    risk_level          = Column(String)
    prediction          = Column(String)
    top_features        = Column(JSON)    # list of {feature, shap_value, direction}
    explanation         = Column(Text)
    threshold_used      = Column(Float, default=0.5)
    created_at          = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id          = Column(Integer, primary_key=True, index=True)
    model_name  = Column(String)
    precision   = Column(Float)
    recall      = Column(Float)
    f1          = Column(Float)
    roc_auc     = Column(Float)
    pr_auc      = Column(Float)
    accuracy    = Column(Float)
    train_size  = Column(Integer)
    val_size    = Column(Integer)
    test_size   = Column(Integer)
    is_best     = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, index=True)
    action      = Column(String)
    entity_type = Column(String)
    entity_id   = Column(String)
    details     = Column(JSON)
    created_at  = Column(DateTime, default=datetime.utcnow)

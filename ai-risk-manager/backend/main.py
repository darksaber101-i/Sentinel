"""
Sentinel — FastAPI Backend
───────────────────────────────────
Entry point. Registers all routers, sets up CORS, initialises the DB,
and seeds it with the synthetic dataset on first run (demo mode).
"""

import sys
import json
import csv
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Ensure the project root is on the path so `ml` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config   import settings
from backend.database import Base, engine, SessionLocal
from backend import models
from backend.routes import orders, predictions, metrics, assistant, alerts

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH  = Path(__file__).parent.parent / "data" / "synthetic_orders.csv"
MODEL_DIR  = Path(__file__).parent.parent / "models"


def _seed_database(db: Session):
    """
    Load synthetic_orders.csv into the DB and run predictions for all orders.
    Called once at startup when the orders table is empty.
    """
    if not DATA_PATH.exists():
        log.warning("synthetic_orders.csv not found. Run: python ml/data_generator.py")
        return

    log.info("Seeding database with synthetic orders …")

    from ml.predict import predict as ml_predict

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch_orders  = []
        batch_preds   = []

        for i, row in enumerate(reader):
            order = models.Order(
                order_id                 = row["order_id"],
                customer_id              = row["customer_id"],
                product_id               = row["product_id"],
                product_category         = row["product_category"],
                order_value              = float(row["order_value"]),
                quantity                 = int(row["quantity"]),
                discount_percentage      = float(row["discount_percentage"]),
                customer_tenure_days     = int(row["customer_tenure_days"]),
                previous_orders          = int(row["previous_orders"]),
                previous_returns         = int(row["previous_returns"]),
                previous_return_rate     = float(row["previous_return_rate"]),
                previous_failed_payments = int(row["previous_failed_payments"]),
                previous_chargebacks     = int(row["previous_chargebacks"]),
                payment_method           = row["payment_method"],
                device_type              = row["device_type"],
                shipping_distance_km     = float(row["shipping_distance_km"]),
                delivery_days            = int(row["delivery_days"]),
                is_new_customer          = row["is_new_customer"] == "1",
                is_first_order           = row["is_first_order"] == "1",
                order_hour               = int(row["order_hour"]),
                days_since_last_order    = int(row["days_since_last_order"]),
                product_return_rate      = float(row["product_return_rate"]),
                support_tickets          = int(row["support_tickets"]),
                is_returned              = int(row["is_returned"]),
            )
            batch_orders.append(order)

            try:
                result = ml_predict(row)
                pred   = models.Prediction(
                    order_id           = row["order_id"],
                    return_probability = result["return_probability"],
                    risk_score         = result["risk_score"],
                    risk_level         = result["risk_level"],
                    prediction         = result["prediction"],
                    top_features       = [],
                    explanation        = f"Risk level: {result['risk_level']} ({result['risk_score']}/100)",
                    threshold_used     = 0.5,
                )
                batch_preds.append(pred)
            except Exception as e:
                log.warning(f"Prediction failed for {row['order_id']}: {e}")

            # Commit in batches of 500 for speed
            if (i + 1) % 500 == 0:
                db.add_all(batch_orders)
                db.add_all(batch_preds)
                db.commit()
                batch_orders = []
                batch_preds  = []
                log.info(f"  Seeded {i + 1} orders …")

        # Final batch
        if batch_orders:
            db.add_all(batch_orders)
            db.add_all(batch_preds)
            db.commit()

    log.info(f"Database seeded with {db.query(models.Order).count():,} orders.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.query(models.Order).count() == 0:
            _seed_database(db)
        else:
            log.info(f"Database already has {db.query(models.Order).count():,} orders.")
    finally:
        db.close()

    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    log.info("Shutting down.")


app = FastAPI(
    title="Sentinel API",
    description="E-commerce order return risk prediction platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(orders.router)
app.include_router(predictions.router)
app.include_router(metrics.router)
app.include_router(assistant.router)
app.include_router(alerts.router)


@app.get("/")
def root():
    return {"message": "Sentinel API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}

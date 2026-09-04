"""
Database Setup — SQLAlchemy with SQLite default / PostgreSQL optional
─────────────────────────────────────────────────────────────────────
WHY SQLALCHEMY?
  It provides a Pythonic ORM that works identically with SQLite (dev)
  and PostgreSQL (production) — just swap the DATABASE_URL env var.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    # SQLite needs this flag to allow multi-thread access (FastAPI is async)
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

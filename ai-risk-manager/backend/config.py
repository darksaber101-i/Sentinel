from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./ai_risk_manager.db"
    ANTHROPIC_API_KEY: str = ""
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    FRONTEND_URL: str = "http://localhost:3000"

    model_config = {"env_file": str(Path(__file__).parent.parent / ".env"), "extra": "ignore"}


settings = Settings()

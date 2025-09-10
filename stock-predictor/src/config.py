from __future__ import annotations
import os, yaml
from pydantic import BaseModel

class Settings(BaseModel):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stocks.db")
    ALPHA_VANTAGE_API_KEY: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")
    MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock-predictor")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev")
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")

def load_hparams(path: str = "configs/config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"defaults": {}}

settings = Settings()
hparams = load_hparams()

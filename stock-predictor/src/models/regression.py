from __future__ import annotations
import os, joblib, numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from src.utils import mape, directional_accuracy

def train_regression(X_flat: np.ndarray, y: np.ndarray):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=5.0))
    ])
    pipe.fit(X_flat, y)
    y_pred = pipe.predict(X_flat)
    rmse = mean_squared_error(y, y_pred, squared=False)
    return pipe, {"rmse": rmse, "mape": mape(y, y_pred), "directional_accuracy": directional_accuracy(y, y_pred)}

def save_regression(model, symbol: str):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = f"artifacts/regression/{symbol}"
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f"ridge-{ts}.joblib")
    joblib.dump(model, fname)
    return fname

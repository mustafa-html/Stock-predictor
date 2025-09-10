from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from src.utils import mape, directional_accuracy

def rolling_backtest(model_fit_fn, predict_fn, X, y, idx, train_frac=0.8, step=5):
    n = len(X)
    split = int(n * train_frac)
    metrics = []
    preds = []

    start = split
    for t in range(start, n, step):
        X_train, y_train = X[:t], y[:t]
        X_test, y_test = X[t:t+step], y[t:t+step]
        idx_test = idx[t:t+step]

        if len(X_test) == 0:
            break

        model = model_fit_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)

        fold = pd.DataFrame({
            "dt": idx_test,
            "y_true": y_test,
            "y_pred": y_pred
        })
        preds.append(fold)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        metrics.append({
            "rmse": rmse,
            "mape": mape(y_test, y_pred),
            "directional_accuracy": directional_accuracy(y_test, y_pred)
        })

    preds_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["dt","y_true","y_pred"])
    metrics_df = pd.DataFrame(metrics)
    return preds_df, metrics_df

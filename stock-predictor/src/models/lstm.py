from __future__ import annotations
import os, numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
from src.utils import mape, directional_accuracy

def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

def train_lstm(X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
    model = build_lstm(X.shape[1:])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X, verbose=0).ravel()
    rmse = mean_squared_error(y, y_pred, squared=False)
    return model, {"rmse": rmse, "mape": mape(y, y_pred), "directional_accuracy": directional_accuracy(y, y_pred)}

def save_lstm(model, symbol: str):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = f"artifacts/lstm/{symbol}"
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f"lstm-{ts}.keras")
    model.save(fname)
    return fname

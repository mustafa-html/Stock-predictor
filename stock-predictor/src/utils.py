from __future__ import annotations
import numpy as np

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nz = y_true != 0
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if np.any(nz) else float("nan")

def directional_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    td = np.sign(np.diff(y_true))
    pd = np.sign(np.diff(y_pred))
    m = min(len(td), len(pd))
    if m == 0:
        return float("nan")
    return float(np.mean(td[:m] == pd[:m]) * 100)

from __future__ import annotations
import numpy as np, pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    gain = up.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = down.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss
    r = 100 - (100 / (1 + rs))
    return r

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy().sort_values("dt")
    df["return_1d"] = df["close"].pct_change()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma30"] = df["close"].rolling(30).mean()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["rsi14"] = rsi(df["close"], 14)
    macd_line, signal_line = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    return df

def to_supervised(df: pd.DataFrame, lookback: int = 60, horizon: int = 1):
    feats = ["close","return_1d","sma10","sma30","ema10","rsi14","macd","macd_signal","volume"]
    df = df.dropna().reset_index(drop=True)
    X, y, idx = [], [], []
    for i in range(lookback, len(df) - horizon + 1):
        X.append(df.loc[i-lookback:i-1, feats].values)
        y.append(df.loc[i + horizon - 1, "close"])
        idx.append(df.loc[i + horizon - 1, "dt"])
    return np.array(X), np.array(y), idx, feats

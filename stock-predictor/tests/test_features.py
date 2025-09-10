import pandas as pd
from src.features.technical import build_features

def test_build_features():
    df = pd.DataFrame({
        "dt": pd.date_range("2023-01-01", periods=100, freq="D"),
        "open": range(100,200),
        "high": range(101,201),
        "low": range(99,199),
        "close": range(100,200),
        "volume": [1_000_000]*100
    })
    out = build_features(df)
    assert {"rsi14","sma10","sma30","ema10","macd","macd_signal","return_1d"} <= set(out.columns)

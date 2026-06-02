from __future__ import annotations
import argparse
import pandas as pd
from sqlalchemy import text
from src.storage.db import get_engine
from src.config import settings

def fetch_yfinance(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df.reset_index(inplace=True)
    df.rename(columns={
        "Date": "dt", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    }, inplace=True)
    return df

def fetch_alpha_vantage(symbol: str, interval: str = "Daily") -> pd.DataFrame:
    from alpha_vantage.timeseries import TimeSeries
    key = settings.ALPHA_VANTAGE_API_KEY
    if not key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set")
    ts = TimeSeries(key=key, output_format="pandas")
    if interval.lower() == "daily":
        data, _ = ts.get_daily(symbol=symbol, outputsize="full")
        data.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close", "5. volume": "volume"
        }, inplace=True)
        data["dt"] = data.index
        data.reset_index(drop=True, inplace=True)
        return data[["dt","open","high","low","close","volume"]].sort_values("dt")
    else:
        raise NotImplementedError("Only Daily interval shown here for brevity.")

def upsert_prices(df: pd.DataFrame, symbol: str):
    engine = get_engine()
    df = df.copy().sort_values("dt")
    df["symbol"] = symbol.upper()
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            sql = "INSERT OR IGNORE INTO prices (symbol, dt, open, high, low, close, volume) VALUES (:symbol, :dt, :open, :high, :low, :close, :volume)"
        else:
            sql = "INSERT INTO prices (symbol, dt, open, high, low, close, volume) VALUES (:symbol, :dt, :open, :high, :low, :close, :volume) ON CONFLICT (symbol, dt) DO NOTHING"
        for _, row in df.iterrows():
            conn.execute(text(sql), dict(
                symbol=row["symbol"],
                dt=pd.to_datetime(row["dt"]).to_pydatetime(),
                open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                close=float(row["close"]), volume=float(row.get("volume", 0.0)),
            ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--source", type=str, default="yfinance", choices=["yfinance","alphavantage"])
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    args = parser.parse_args()
    #what is this

    if args.source == "yfinance":
        df = fetch_yfinance(args.symbol, period=args.period, interval=args.interval)
    else:
        df = fetch_alpha_vantage(args.symbol, interval="Daily")
    upsert_prices(df, args.symbol)
    print(f"Ingested {len(df)} rows for {args.symbol}.")

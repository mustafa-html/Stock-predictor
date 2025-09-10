from __future__ import annotations
import argparse, numpy as np, pandas as pd
from sqlalchemy import text
from src.storage.db import get_engine
from src.features.technical import build_features, to_supervised
from src.models.regression import train_regression, save_regression
from src.models.lstm import train_lstm, save_lstm
from src.evaluation.backtest import rolling_backtest
from src.config import settings, hparams

def load_prices(symbol: str) -> pd.DataFrame:
    engine = get_engine()
    q = "SELECT dt, open, high, low, close, volume FROM prices WHERE symbol = :symbol ORDER BY dt"
    with engine.begin() as conn:
        df = pd.read_sql(text(q), conn, params={"symbol": symbol.upper()})
    return df

def save_features(df_feat: pd.DataFrame, symbol: str):
    engine = get_engine()
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            sql = ("INSERT OR IGNORE INTO features (symbol, dt, rsi14, sma10, sma30, ema10, macd, macd_signal, return_1d) "
                   "VALUES (:symbol, :dt, :rsi14, :sma10, :sma30, :ema10, :macd, :macd_signal, :return_1d)")
        else:
            sql = ("INSERT INTO features (symbol, dt, rsi14, sma10, sma30, ema10, macd, macd_signal, return_1d) "
                   "VALUES (:symbol, :dt, :rsi14, :sma10, :sma30, :ema10, :macd, :macd_signal, :return_1d) "
                   "ON CONFLICT (symbol, dt) DO NOTHING")
        for _, row in df_feat.dropna().iterrows():
            conn.execute(text(sql), dict(
                symbol=symbol.upper(),
                dt=pd.to_datetime(row["dt"]).to_pydatetime(),
                rsi14=float(row["rsi14"]), sma10=float(row["sma10"]), sma30=float(row["sma30"]),
                ema10=float(row["ema10"]), macd=float(row["macd"]), macd_signal=float(row["macd_signal"]),
                return_1d=float(row["return_1d"]),
            ))

def save_predictions(preds: pd.DataFrame, symbol: str, model_name: str):
    engine = get_engine()
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            sql = ("INSERT OR REPLACE INTO predictions (symbol, dt, model, y_pred, y_true) "
                   "VALUES (:symbol, :dt, :model, :y_pred, :y_true)")
        else:
            sql = ("INSERT INTO predictions (symbol, dt, model, y_pred, y_true) "
                   "VALUES (:symbol, :dt, :model, :y_pred, :y_true) "
                   "ON CONFLICT (symbol, dt, model) DO UPDATE SET y_pred=EXCLUDED.y_pred, y_true=EXCLUDED.y_true")
        for _, row in preds.iterrows():
            conn.execute(text(sql), dict(
                symbol=symbol.upper(),
                dt=pd.to_datetime(row["dt"]).to_pydatetime(),
                model=model_name, y_pred=float(row["y_pred"]), y_true=float(row["y_true"])
            ))

def save_metrics(metrics_df: pd.DataFrame, symbol: str, model_name: str):
    engine = get_engine()
    with engine.begin() as conn:
        sql = ("INSERT INTO metrics (run_id, symbol, model, rmse, mape, directional_accuracy) "
               "VALUES (:run_id, :symbol, :model, :rmse, :mape, :directional_accuracy)")
        for _, row in metrics_df.iterrows():
            conn.execute(text(sql), dict(
                run_id=None, symbol=symbol.upper(), model=model_name,
                rmse=float(row["rmse"]), mape=float(row["mape"]), directional_accuracy=float(row["directional_accuracy"])
            ))

def main(symbol: str, epochs: int):
    prices = load_prices(symbol)
    if prices.empty:
        raise RuntimeError(f"No prices for {symbol}. Run ingestion first.")
    feats = build_features(prices)
    save_features(feats[["dt","rsi14","sma10","sma30","ema10","macd","macd_signal","return_1d"]].copy().assign(symbol=symbol), symbol)

    lookback = hparams["defaults"].get("lookback_window", 60)
    horizon = hparams["defaults"].get("forecast_horizon", 1)

    X, y, idx, feat_names = to_supervised(feats, lookback=lookback, horizon=horizon)

    # Regression baseline
    X_flat = X.reshape(X.shape[0], -1)
    reg_model, reg_train_metrics = train_regression(X_flat, y)

    def fit_reg(Xtr, ytr):
        from src.models.regression import train_regression
        m, _ = train_regression(Xtr.reshape(Xtr.shape[0], -1), ytr)
        return m
    def predict_reg(m, Xte):
        return m.predict(Xte.reshape(Xte.shape[0], -1))

    reg_preds, reg_metrics = rolling_backtest(fit_reg, predict_reg, X, y, idx)
    save_predictions(reg_preds, symbol, "ridge_regression")
    save_metrics(reg_metrics, symbol, "ridge_regression")
    reg_path = save_regression(reg_model, symbol)
    print(f"[REG] Train metrics: {reg_train_metrics} | saved: {reg_path}")

    # LSTM deep model
    lstm_model, lstm_train_metrics = train_lstm(X, y, epochs=epochs)
    def fit_lstm(Xtr, ytr):
        from src.models.lstm import train_lstm
        m, _ = train_lstm(Xtr, ytr, epochs=5)
        return m
    def predict_lstm(m, Xte):
        return m.predict(Xte, verbose=0).ravel()

    lstm_preds, lstm_metrics = rolling_backtest(fit_lstm, predict_lstm, X, y, idx)
    save_predictions(lstm_preds, symbol, "lstm")
    save_metrics(lstm_metrics, symbol, "lstm")
    lstm_path = save_lstm(lstm_model, symbol)
    print(f"[LSTM] Train metrics: {lstm_train_metrics} | saved: {lstm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.symbol, args.epochs)

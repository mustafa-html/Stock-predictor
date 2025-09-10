CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    dt TIMESTAMP NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    UNIQUE(symbol, dt)
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    dt TIMESTAMP NOT NULL,
    rsi14 REAL, sma10 REAL, sma30 REAL, ema10 REAL, macd REAL, macd_signal REAL, return_1d REAL,
    UNIQUE(symbol, dt)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    dt TIMESTAMP NOT NULL,
    model TEXT NOT NULL,
    y_pred REAL,
    y_true REAL,
    UNIQUE(symbol, dt, model)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    symbol TEXT NOT NULL,
    model TEXT NOT NULL,
    rmse REAL, mape REAL, directional_accuracy REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

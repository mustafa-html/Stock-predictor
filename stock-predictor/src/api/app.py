from __future__ import annotations
from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
from sqlalchemy import text
from src.storage.db import get_engine

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stock Predictor</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
    .row { display: flex; gap: 2rem; }
    .card { border: 1px solid #eee; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .metrics td { padding: .25rem .5rem; }
    input, select, button { padding: .5rem .8rem; border-radius: 8px; border: 1px solid #ccc; }
    .topbar { display:flex; align-items:center; gap:1rem; margin-bottom:1rem; }
  </style>
</head>
<body>
  <div class="topbar">
    <form method="get" action="/symbol">
      <label>Symbol:</label>
      <input type="text" name="symbol" value="{{ symbol }}" placeholder="AAPL" />
      <button type="submit">Load</button>
    </form>
    <form method="post" action="/train/{{ symbol }}">
      <button type="submit">Train (quick)</button>
    </form>
  </div>

  <div class="row">
    <div class="card" style="flex:2">
      <h3>Actual vs Predicted — {{ symbol }}</h3>
      <div id="chart"></div>
    </div>
    <div class="card" style="flex:1">
      <h3>Latest Metrics</h3>
      <table class="metrics">
        <tr><th>Model</th><th>RMSE</th><th>MAPE</th><th>DirAcc%</th></tr>
        {% for row in metrics %}
        <tr><td>{{ row.model }}</td><td>{{ "%.4f"|format(row.rmse or 0) }}</td><td>{{ "%.2f"|format(row.mape or 0) }}</td><td>{{ "%.2f"|format(row.directional_accuracy or 0) }}</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>

  <script>
    const data = {{ plot_data | safe }};
    const layout = { title: "Close vs Predictions", xaxis: {title: "Date"}, yaxis: {title: "Price"} };
    Plotly.newPlot('chart', data, layout);
  </script>
</body>
</html>
"""

def load_metrics(symbol: str):
    engine = get_engine()
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT model, AVG(rmse) rmse, AVG(mape) mape, AVG(directional_accuracy) directional_accuracy FROM metrics WHERE symbol=:s GROUP BY model"), conn, params={"s": symbol.upper()})
    return df

def load_series(symbol: str):
    engine = get_engine()
    with engine.begin() as conn:
        prices = pd.read_sql(text("SELECT dt, close FROM prices WHERE symbol=:s ORDER BY dt"), conn, params={"s": symbol.upper()})
        preds = pd.read_sql(text("SELECT dt, model, y_pred FROM predictions WHERE symbol=:s ORDER BY dt"), conn, params={"s": symbol.upper()})
    return prices, preds

@app.route("/")
def home():
    return redirect(url_for("symbol_view", symbol="AAPL"))

@app.route("/health")
def health():
    return {"status":"ok"}

@app.route("/symbol")
def symbol_query():
    symbol = request.args.get("symbol","AAPL")
    return redirect(url_for("symbol_view", symbol=symbol))

@app.route("/symbol/<symbol>")
def symbol_view(symbol):
    prices, preds = load_series(symbol)
    if prices.empty:
        return f"⚠️ No data for {symbol}. Ingest first.", 404
    data = [
        {"x": prices["dt"].astype(str).tolist(), "y": prices["close"].tolist(), "name":"Actual", "mode":"lines"}
    ]
    for model in preds["model"].unique():
        sub = preds[preds["model"] == model]
        data.append({"x": sub["dt"].astype(str).tolist(), "y": sub["y_pred"].tolist(), "name": model, "mode":"lines"})
    metrics = load_metrics(symbol).to_dict(orient="records")
    return render_template_string(PAGE, symbol=symbol.upper(), plot_data=data, metrics=metrics)

@app.post("/train/<symbol>")
def train(symbol):
    import subprocess, sys
    try:
        subprocess.run([sys.executable, "-m", "src.training.pipeline", "--symbol", symbol, "--epochs", "5"], check=True)
    except Exception:
        pass
    return redirect(url_for("symbol_view", symbol=symbol))

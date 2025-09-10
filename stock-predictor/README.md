# Stock Price Prediction Tool — Python, TensorFlow, Flask, SQL, AWS

Production-style project for forecasting stock prices using **historical market data** with both a **baseline regression model** and a **TensorFlow LSTM**. It demonstrates:

- **Models:** Ridge Regression (baseline) and LSTM for sequence forecasting.
- **Pipelines:** Ingestion from Yahoo Finance / Alpha Vantage, preprocessing, and **technical feature engineering**.
- **Backtesting:** Rolling-origin evaluation (walk-forward) with RMSE, MAPE, Directional Accuracy.
- **Dashboard:** Flask + Plotly web UI for interactive trends, predictions, and metrics.
- **Storage:** SQL (PostgreSQL recommended; SQLite fallback) for prices, features, predictions, metrics.
- **Cloud:** Docker image ready for AWS App Runner / ECS; optional MLflow tracking.

---

## Quickstart (Local)

**1) Setup**
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then edit if needed
```

**2) (Optional) Start Postgres and MLflow locally**
```bash
docker compose up -d
```
- `db`: Postgres at `localhost:5432` (user: `ml`, pass: `ml`, db: `stocks`)
- `mlflow`: http://localhost:5001

**3) Initialize DB + Ingest AAPL**
```bash
python -m src.storage.db --init
python -m src.data.ingest --symbol AAPL --source yfinance --period "5y" --interval "1d"
```

**4) Train and evaluate**
```bash
python -m src.training.pipeline --symbol AAPL --epochs 10
```

**5) Run the dashboard**
```bash
FLASK_APP=src.api.app:app flask run --port 8000
# Visit http://127.0.0.1:8000
```

---

## Project Structure

```
stock-predictor/
├─ README.md
├─ requirements.txt
├─ docker-compose.yml
├─ Dockerfile
├─ Makefile
├─ .gitignore
├─ .env.example
├─ configs/
│  └─ config.yaml
├─ sql/
│  ├─ create_tables_sqlite.sql
│  └─ create_tables_postgres.sql
├─ src/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ config.py
│  ├─ storage/
│  │  ├─ __init__.py
│  │  └─ db.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ ingest.py
│  ├─ features/
│  │  ├─ __init__.py
│  │  └─ technical.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ regression.py
│  │  └─ lstm.py
│  ├─ evaluation/
│  │  ├─ __init__.py
│  │  └─ backtest.py
│  ├─ training/
│  │  ├─ __init__.py
│  │  └─ pipeline.py
│  └─ api/
│     ├─ __init__.py
│     └─ app.py
├─ tests/
│  └─ test_features.py
└─ .github/workflows/ci.yml
```

---

## Config & Dialects

- `.env` holds runtime env vars (see `.env.example`).
- Auto-detect DB dialect:
  - **SQLite**: uses `INSERT OR IGNORE/REPLACE` UPSERTs and `sql/create_tables_sqlite.sql`.
  - **PostgreSQL**: uses `ON CONFLICT` UPSERTs and `sql/create_tables_postgres.sql`.

Set `DATABASE_URL` accordingly, e.g.:
- SQLite: `sqlite:///./stocks.db`
- Postgres: `postgresql+psycopg2://ml:ml@localhost:5432/stocks`

---

## AWS (App Runner)

1. Push this repo to GitHub.
2. Create an **RDS Postgres** and note the connection string.
3. Create an **App Runner** service from your repo (uses Dockerfile). Configure environment variables:
   - `DATABASE_URL`
   - `FLASK_ENV=production`
   - `SECRET_KEY=<random>`
4. Health check path: `/health`

**Alternative**: ECS Fargate behind an ALB.

---

## Notes

- This repository is set up for clean iteration: modular code, unit tests for features, and a baseline vs. deep model for fair comparison.
- For real-time ingestion in production, add a small scheduler (APScheduler, Lambda, or GitHub Actions cron) invoking `src.data.ingest` and `src.training.pipeline`.
- MLflow hooks are present (set `MLFLOW_TRACKING_URI` if you want to enable it).

MIT License.

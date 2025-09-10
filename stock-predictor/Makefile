.PHONY: init ingest train ui test docker-up docker-down

init:
	python -m src.storage.db --init

ingest:
	python -m src.data.ingest --symbol AAPL --source yfinance --period "5y" --interval "1d"

train:
	python -m src.training.pipeline --symbol AAPL --epochs 8

ui:
	FLASK_APP=src.api.app:app flask run --port 8000

test:
	pytest -q

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

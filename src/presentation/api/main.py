from fastapi import FastAPI
from src.infra.fetchers.coinbase_price import get_coinbase_price_history
from src.app.defaults import DEFAULT_CONFIG
from dataclasses import replace
from datetime import datetime

app = FastAPI()


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/prices")
def get_prices(coin: str | None = None, start_date: datetime | None = None, end_date: datetime | None = None):
    if not (coin and start_date and end_date):
        return get_coinbase_price_history(DEFAULT_CONFIG).to_dict(orient="records")
    else:
        config = replace(DEFAULT_CONFIG, coin = coin.upper(), start_date = start_date, end_date = end_date)
        return get_coinbase_price_history(config).to_dict(orient="records")
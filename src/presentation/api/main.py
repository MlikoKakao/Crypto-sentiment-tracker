from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.infra.fetchers.coinbase_price import get_coinbase_price_history
from src.app.defaults import DEFAULT_CONFIG
from dataclasses import replace
from datetime import datetime
from src.domain.sentiment.service import add_sentiment_to_df
from src.infra.fetchers.service import fetch_posts
from src.domain.market.indicators import add_indicators_to_df
from src.domain.market.dto import IndicatorConfig
from src.shared.helpers import is_date_correct
from src.infra.storage.db.price_repository import save_price_df
from src.infra.storage.db.sentiment_repository import save_sentiment_df
from src.infra.storage.db.signal_repository import save_signal_df
from src.infra.storage.db.content_repository import save_content_df

app = FastAPI()


class IngestRequest(BaseModel):
    coin: str
    start_date: datetime
    end_date: datetime
    analyzer: str
    sources: tuple[str, ...]


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/prices")
def get_prices(request: IngestRequest):
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    config = replace(
        DEFAULT_CONFIG,
        coin=request.coin.upper(),
        start_date=request.start_date,
        end_date=request.end_date,
    )
    return get_coinbase_price_history(config).to_dict(orient="records")


@app.get("/sentiment")
def get_sentiment(request: IngestRequest):
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    if not (
        request.coin
        and request.start_date
        and request.end_date
        and request.analyzer
        and request.sources
    ):
        config = DEFAULT_CONFIG
    else:
        config = replace(
            DEFAULT_CONFIG,
            coin=request.coin.upper(),
            start_date=request.start_date,
            end_date=request.end_date,
            analyzer=request.analyzer,
            sources=request.sources,
        )
    df = fetch_posts(config)
    sentiment_df = add_sentiment_to_df(df)
    return sentiment_df.to_dict(orient="records")


@app.get("/signals")
def get_signals(request: IndicatorConfig):
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    config = replace(
        DEFAULT_CONFIG,
        coin=request.coin.upper(),
        start_date=request.start_date,
        end_date=request.end_date,
    )

    price_df = get_coinbase_price_history(config)
    signals_df = add_indicators_to_df(price_df, request)

    return signals_df.to_dict(orient="records")


@app.post("/ingest")
def ingest(request: IngestRequest):
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    config = replace(
        DEFAULT_CONFIG,
        coin=request.coin.upper(),
        start_date=request.start_date,
        end_date=request.end_date,
        analyzer=request.analyzer,
        sources=request.sources,
    )

    price_df = get_coinbase_price_history(config)
    save_price_df(price_df, request.coin)
    posts_df = fetch_posts(config)
    save_content_df(posts_df, request.coin)
    sentiment_df = add_sentiment_to_df(posts_df, request.analyzer)
    save_sentiment_df(sentiment_df, request.coin)
    return {
        "status": "ok",
        "coin": config.coin,
        "sources": config.sources,
        "price_points": len(price_df),
        "posts_ingested": len(posts_df),
        "sentiment_rows": len(sentiment_df),
    }

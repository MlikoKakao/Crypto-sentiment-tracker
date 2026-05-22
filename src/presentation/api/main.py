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
from src.infra.storage.db.price_repository import save_price_df, load_price_df
from src.infra.storage.db.sentiment_repository import save_sentiment_df, load_sentiment_df
from src.infra.storage.db.content_repository import save_content_df, load_content_df
from src.infra.storage.db.schema import init_db
from src.app.dto import Analyzer, Source
from src.domain.sentiment.registry import ALL_ANALYZER_NAMES
import pandas as pd

app = FastAPI()


class IngestRequest(BaseModel):
    coin: str
    num_posts: int
    start_date: datetime
    end_date: datetime
    analyzer: Analyzer
    sources: tuple[Source, ...]


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
    return load_price_df(config).to_dict(orient="records")


@app.get("/posts")
def get_posts(request: IngestRequest):
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    config = replace(
        DEFAULT_CONFIG,
        coin=request.coin.upper(),
        start_date=request.start_date,
        end_date=request.end_date,
        sources=request.sources,
        num_posts=request.num_posts,
    )
    posts_df = pd.DataFrame()
    for source in request.sources:
        posts_df = pd.concat([posts_df, load_content_df(config, source)])

    return posts_df.to_dict(orient="records")


@app.get("/sentiment")
def get_sentiment(request: IngestRequest):
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    config = replace(
        DEFAULT_CONFIG,
        coin=request.coin.upper(),
        start_date=request.start_date,
        end_date=request.end_date,
        analyzer=request.analyzer,
        sources=request.sources,
        num_posts=request.num_posts,
    )
    sentiment_df = pd.DataFrame()
    if request.analyzer == "all":
        for analyzer in ALL_ANALYZER_NAMES:
            sentiment_df = load_sentiment_df(config, analyzer)
    else:
        sentiment_df = load_sentiment_df(config, request.analyzer)
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
    init_db()
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    config = replace(
        DEFAULT_CONFIG,
        coin=request.coin.upper(),
        num_posts=request.num_posts,
        start_date=request.start_date,
        end_date=request.end_date,
        analyzer=request.analyzer,
        sources=request.sources,
    )

    price_df = get_coinbase_price_history(config)
    save_price_df(price_df, request.coin)
    posts_df = fetch_posts(config)
    save_content_df(posts_df, request.coin)
    len_sent = 0
    sentiment_df = pd.DataFrame()
    if request.analyzer == "all":
        for analyzer in ALL_ANALYZER_NAMES:
            sentiment_df = add_sentiment_to_df(posts_df, analyzer)
            save_sentiment_df(sentiment_df, request.coin)
            len_sent += len(sentiment_df)
    else:
        sentiment_df = add_sentiment_to_df(posts_df, request.analyzer)
        save_sentiment_df(sentiment_df, request.coin)
    return {
        "status": "ok",
        "coin": config.coin,
        "sources": config.sources,
        "price_points": len(price_df),
        "posts_ingested": len(posts_df),
        "sentiment_rows": len_sent if request.analyzer == "all" else len(sentiment_df),
    }

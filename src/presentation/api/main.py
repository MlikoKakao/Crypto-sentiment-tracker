from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.app.defaults import DEFAULT_CONFIG
from dataclasses import replace
from datetime import datetime
from src.domain.sentiment.service import add_sentiment_to_df
from src.infra.fetchers.service import fetch_posts
from src.app.use_cases.get_indicators import add_indicators_with_cache
from src.domain.market.dto import IndicatorConfig
from src.shared.helpers import is_date_correct
from src.infra.storage.db.sentiment_repository import (
    save_sentiment_df,
    load_sentiment_df,
)
from src.infra.storage.db.content_repository import save_content_df
from src.infra.storage.db.schema import init_db
from src.app.dto import Analyzer, Source
from src.domain.sentiment.registry import ALL_ANALYZER_NAMES
import pandas as pd

from src.presentation.api.routes.health import router as health_router
from src.presentation.api.routes.market import router as market_router
from src.presentation.api.routes.posts import router as posts_router
from src.presentation.api.routes.sentiment import router as sentiment_router

app = FastAPI()

app.include_router(health_router)
app.include_router(market_router, prefix="/market", tags=["market"])
app.include_router(posts_router)
app.include_router(sentiment_router)


class IngestRequest(BaseModel):
    coin: str
    num_posts: int
    start_date: datetime
    end_date: datetime
    analyzer: Analyzer
    sources: tuple[Source, ...]


@app.get("/signals")
def get_signals(
    coin: str,
    start_date: datetime,
    end_date: datetime,
    use_sma: bool = False,
    use_rsi: bool = False,
    use_macd: bool = False,
):
    if not is_date_correct(start_date, end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    config = replace(
        DEFAULT_CONFIG,
        coin=coin.upper(),
        start_date=start_date,
        end_date=end_date,
    )
    request = IndicatorConfig(
        coin=coin.upper(),
        start_date=start_date,
        end_date=end_date,
        use_sma=use_sma,
        use_rsi=use_rsi,
        use_macd=use_macd,
    )

    price_df = load_price_df(config)
    signals_df = add_indicators_with_cache(price_df, request)

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
    if config.analyzer == "all":
        for analyzer_name in ALL_ANALYZER_NAMES:
            one_df = add_sentiment_to_df(posts_df, analyzer_name)
            save_sentiment_df(one_df, config.coin)
            sentiment_df = pd.concat([sentiment_df, one_df], ignore_index=True)
            len_sent += len(one_df)
    else:
        sentiment_df = add_sentiment_to_df(posts_df, request.analyzer)
        save_sentiment_df(sentiment_df, config.coin)
        len_sent = len(sentiment_df)
    return {
        "status": "ok",
        "coin": config.coin,
        "sources": config.sources,
        "price_points": len(price_df),
        "posts_ingested": len(posts_df),
        "sentiment_rows": len_sent if request.analyzer == "all" else len(sentiment_df),
    }

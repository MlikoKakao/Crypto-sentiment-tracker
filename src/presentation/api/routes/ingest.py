from fastapi import APIRouter, HTTPException
import pandas as pd
from pydantic import BaseModel
from dataclasses import replace
from datetime import datetime

from src.app.defaults import default_config
from src.app.dto import Analyzer, Source

from src.infra.storage.db.sentiment_repository import save_sentiment_df
from src.infra.storage.db.content_repository import save_content_df
from src.infra.storage.db.price_repository import save_price_df
from src.infra.storage.db.schema import init_db
from src.shared.helpers import is_date_correct, normalize_coin


class IngestRequest(BaseModel):
    coin: str
    num_posts: int
    start_date: datetime
    end_date: datetime
    analyzer: Analyzer
    sources: list[Source]


router = APIRouter()


@router.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    from src.domain.sentiment.service import add_sentiment_to_df
    from src.infra.fetchers.service import fetch_posts
    from src.infra.fetchers.coinbase_price import get_coinbase_price_history

    init_db()
    if not is_date_correct(request.start_date, request.end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    try:
        coin = normalize_coin(request.coin)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = replace(
        default_config(),
        coin=coin,
        num_posts=request.num_posts,
        start_date=request.start_date,
        end_date=request.end_date,
        analyzer=request.analyzer,
        sources=tuple(request.sources),
    )

    price_df = get_coinbase_price_history(config)
    save_price_df(price_df, request.coin)
    posts_df = fetch_posts(config)
    save_content_df(posts_df, request.coin)

    if request.analyzer == "all":
        from src.domain.sentiment.registry import ALL_ANALYZER_NAMES

        frames = []

        for analyzer_name in ALL_ANALYZER_NAMES:
            one_df = add_sentiment_to_df(posts_df, analyzer_name)
            save_sentiment_df(one_df, config.coin)
            frames.append(one_df)

        sentiment_df = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
    else:
        sentiment_df = add_sentiment_to_df(posts_df, request.analyzer)
        save_sentiment_df(sentiment_df, config.coin)

    return {
        "status": "ok",
        "coin": config.coin,
        "sources": config.sources,
        "price_points": len(price_df),
        "posts_ingested": len(posts_df),
        "sentiment_rows": len(sentiment_df),
    }

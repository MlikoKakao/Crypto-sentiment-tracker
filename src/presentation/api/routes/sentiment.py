from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from dataclasses import replace
import pandas as pd
from typing import Any, cast


from src.app.dto import Analyzer, Source
from src.app.defaults import DEFAULT_CONFIG
from src.shared.helpers import is_date_correct, normalize_coin
from src.infra.storage.db.sentiment_repository import load_sentiment_df
from src.shared.dataframe_utils import format_timestamp_for_api

router = APIRouter()


@router.get("/sentiment")
def get_sentiment(
    coin: str,
    start_date: datetime,
    end_date: datetime,
    sources: list[Source] = Query(...),
    num_posts: int = 10,
    analyzer: Analyzer = "vader",
) -> list[dict[str, Any]]:
    if not is_date_correct(start_date, end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    try:
        coin = normalize_coin(coin)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = replace(
        DEFAULT_CONFIG,
        coin=coin,
        start_date=start_date,
        end_date=end_date,
        analyzer=analyzer,
        sources=tuple(sources),
        num_posts=num_posts,
    )

    if analyzer == "all":
        from src.domain.sentiment.registry import ALL_ANALYZER_NAMES

        frames = [
            load_sentiment_df(config, analyzer_name)
            for analyzer_name in ALL_ANALYZER_NAMES
        ]
        sentiment_df = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
    else:
        sentiment_df = load_sentiment_df(config, analyzer)

    sentiment_df = format_timestamp_for_api(sentiment_df)

    return cast(list[dict[str, Any]], sentiment_df.to_dict(orient="records"))

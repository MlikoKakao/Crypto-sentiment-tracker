from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from dataclasses import replace
from typing import Any, cast
import pandas as pd

from src.app.defaults import DEFAULT_CONFIG
from src.app.dto import Source
from src.infra.storage.db.content_repository import load_content_df
from src.shared.helpers import is_date_correct
from src.shared.dataframe_utils import format_timestamp_for_api

router = APIRouter()


@router.get("/posts")
def get_posts(
    coin: str,
    start_date: datetime,
    end_date: datetime,
    sources: list[Source] = Query(...),
    num_posts: int = 10,
) -> list[dict[str, Any]]:
    if not is_date_correct(start_date, end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    config = replace(
        DEFAULT_CONFIG,
        coin=coin.upper(),
        start_date=start_date,
        end_date=end_date,
        sources=tuple(sources),
        num_posts=num_posts,
    )
    posts = [load_content_df(config, source) for source in sources]
    posts_df = pd.concat(posts, ignore_index=True) if posts else pd.DataFrame()
    posts_df = format_timestamp_for_api(posts_df)

    return cast(list[dict[str, Any]], posts_df.to_dict(orient="records"))

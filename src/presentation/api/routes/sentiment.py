from fastapi import APIRouter, Depends, Query
import pandas as pd


from src.app.dto import Analyzer, Source
from src.presentation.api.helpers.format import dataframe_to_response_models
from src.presentation.api.helpers.prep_config import sentiment_to_config
from src.presentation.api.helpers.validate import DateRangeParams
from src.presentation.api.schemas.sentiment import SentimentResponse
from src.infra.storage.db.sentiment_repository import load_sentiment_df

router = APIRouter()


@router.get("/sentiment", response_model=list[SentimentResponse])
def get_sentiment(
    params: DateRangeParams = Depends(),
    sources: list[Source] = Query(...),
    num_posts: int = 10,
    analyzer: Analyzer = "vader",
) -> list[SentimentResponse]:
    config = sentiment_to_config(params, sources, num_posts, analyzer)

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

    return dataframe_to_response_models(sentiment_df, SentimentResponse)

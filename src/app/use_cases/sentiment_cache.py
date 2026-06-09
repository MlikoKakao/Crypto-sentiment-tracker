import pandas as pd

from src.app.dto import AnalysisConfig
from src.domain.sentiment.registry import ALL_ANALYZER_NAMES
from src.domain.sentiment.service import add_sentiment_to_df
from src.infra.storage.db.sentiment_repository import (
    has_sentiment_coverage,
    load_sentiment_df,
    save_sentiment_df,
)


def get_or_create_sentiment_df(
    config: AnalysisConfig, posts_df: pd.DataFrame
) -> pd.DataFrame:
    if posts_df.empty:
        return posts_df
    if config.analyzer == "all":
        return get_or_create_multiple_sentiment_df(config, posts_df)
    else:
        return get_or_create_single_sentiment_df(config, posts_df)


def get_or_create_single_sentiment_df(
    config: AnalysisConfig, posts_df: pd.DataFrame
) -> pd.DataFrame:
    if posts_df.empty:
        return posts_df

    sentiment_df = load_sentiment_df(config, config.analyzer)
    if not has_sentiment_coverage(config, sentiment_df):
        sentiment_df = add_sentiment_to_df(posts_df, analyzer_name=config.analyzer)

        if not sentiment_df.empty:
            save_sentiment_df(sentiment_df, coin=config.coin)
    return sentiment_df


def get_or_create_multiple_sentiment_df(
    config: AnalysisConfig, posts_df: pd.DataFrame
) -> pd.DataFrame:
    if posts_df.empty:
        return posts_df

    score_frames = []

    for analyzer in ALL_ANALYZER_NAMES:
        loaded_df = load_sentiment_df(config, analyzer)

        if not has_sentiment_coverage(config, loaded_df):
            loaded_df = add_sentiment_to_df(posts_df, analyzer_name=analyzer)
            save_sentiment_df(loaded_df, coin=config.coin)

        scores = loaded_df[["source", "content_hash", "sentiment"]].rename(
            columns={"sentiment": f"sentiment_{analyzer}"}
        )

        score_frames.append(scores)

    result_df = posts_df.copy()

    for scores in score_frames:
        result_df = result_df.merge(scores, on=["source", "content_hash"], how="left")

    sentiment_cols = [f"sentiment_{analyzer}" for analyzer in ALL_ANALYZER_NAMES]

    result_df["sentiment"] = result_df[sentiment_cols].mean(axis=1)
    result_df["analyzer"] = "all"

    return result_df

from sqlalchemy import bindparam, text
from typing import Any, cast
import pandas as pd
from datetime import timedelta

from src.app.dto import AnalysisConfig
from src.infra.storage.db.connection import get_engine
from src.shared.dataframe_schema import REQUIRED_SENTIMENT_COLUMNS, require_columns


def save_sentiment_df(sentiment_df: pd.DataFrame, coin: str = "btc") -> None:
    df = sentiment_df.copy()
    df["coin"] = coin.upper()

    require_columns(df, REQUIRED_SENTIMENT_COLUMNS, "sentiment_df")

    rows = cast(
        list[dict[str, Any]],
        df[
            [
                "coin",
                "source",
                "content_hash",
                "analyzer",
                "sentiment",
            ]
        ].to_dict(orient="records"),
    )
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sentiment (
                    coin,
                    source,
                    content_hash,
                    analyzer,
                    sentiment
                )
                VALUES (
                    :coin, 
                    :source, 
                    :content_hash, 
                    :analyzer, 
                    :sentiment
                )
                ON CONFLICT DO NOTHING
                """
            ),
            rows,
        )


def load_sentiment_df(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
    engine = get_engine()

    if not config.sources:
        return pd.DataFrame()

    query = text(
        """
                SELECT
                    c.coin,
                    c.source,
                    c.content_hash,
                    c.timestamp,
                    c.text,
                    c.url,
                    s.analyzer,
                    s.sentiment
                FROM content_items AS c
                JOIN sentiment AS s
                  ON s.coin = c.coin
                 AND s.source = c.source
                 AND s.content_hash = c.content_hash
                WHERE c.coin = :coin
                  AND s.analyzer = :analyzer
                  AND c.timestamp BETWEEN :start_date AND :end_date
                  AND c.source IN :sources
                ORDER BY c.timestamp DESC
                LIMIT :limit
                """
    ).bindparams(bindparam("sources", expanding=True))

    with engine.begin() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(
                dict[str, Any],
                {
                    "coin": config.coin.upper(),
                    "analyzer": analyzer,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "sources": list(config.sources),
                    "limit": config.num_posts,
                },
            ),
        )

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def has_sentiment_coverage(config: AnalysisConfig, sentiment_df: pd.DataFrame) -> bool:
    if sentiment_df.empty:
        return False

    posts_count = len(sentiment_df)
    enough_posts = posts_count >= config.num_posts * 0.9

    tolerance = timedelta(days=1)
    min_time = sentiment_df["timestamp"].min()
    max_time = sentiment_df["timestamp"].max()

    start_date = pd.to_datetime(config.start_date, utc=True).tz_convert(None)
    end_date = pd.to_datetime(config.end_date, utc=True).tz_convert(None)

    starts_near = min_time <= start_date + tolerance
    ends_near = max_time >= end_date - tolerance

    return starts_near and ends_near and enough_posts

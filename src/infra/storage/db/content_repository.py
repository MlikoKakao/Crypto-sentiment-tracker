import pandas as pd
from typing import Any, cast
from datetime import timedelta
from sqlalchemy import text

from src.app.dto import AnalysisConfig
from src.infra.storage.db.connection import get_engine
from src.shared.dataframe_schema import require_columns, REQUIRED_CONTENT_COLUMNS
from src.shared.helpers import normalize_timestamp_column
from src.shared.db_helpers import (
    add_optional_cols_inplace,
    build_content_hash,
)


def save_content_df(content_df: pd.DataFrame, coin: str = "btc") -> pd.DataFrame:
    df = content_df.copy()
    df = df.rename(columns={"id": "source_id"})
    df["coin"] = coin.upper()
    add_optional_cols_inplace(df)
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    require_columns(df, REQUIRED_CONTENT_COLUMNS, "content_df")

    df["content_hash"] = df.apply(build_content_hash, axis=1)

    rows = cast(
        list[dict[str, Any]],
        df[
            ["coin", "source", "source_id", "timestamp", "text", "url", "content_hash"]
        ].to_dict(orient="records"),
    )
    if not rows:
        return df

    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO content_items (
                    coin,
                    source,
                    source_id,
                    timestamp,
                    text,
                    url,
                    content_hash
                )
                VALUES (
                    :coin,
                    :source,
                    :source_id,
                    :timestamp,
                    :text,
                    :url,
                    :content_hash
                )
                ON CONFLICT DO NOTHING
                """
            ),
            rows,
        )

    return df


def load_content_df(config: AnalysisConfig, source: str) -> pd.DataFrame:
    engine = get_engine()

    query = text(
        """
        SELECT *
        FROM content_items
        WHERE coin = :coin
            AND source = :source
            AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "coin": config.coin.upper(),
                "source": source,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "limit": config.num_posts,
            },
        )
    if not df.empty:
        df = normalize_timestamp_column(df)

    return df


def has_content_coverage(config: AnalysisConfig, content_df: pd.DataFrame) -> bool:
    if content_df.empty:
        return False

    posts_count = len(content_df)
    enough_posts = posts_count >= config.num_posts / 2

    tolerance = timedelta(days=1)
    min_time = content_df["timestamp"].min()
    max_time = content_df["timestamp"].max()

    start_date = pd.to_datetime(config.start_date, utc=True).tz_convert(None)
    end_date = pd.to_datetime(config.end_date, utc=True).tz_convert(None)

    starts_near = min_time <= start_date + tolerance
    ends_near = max_time >= end_date - tolerance

    return starts_near and ends_near and enough_posts

from pathlib import Path
from contextlib import closing
import pandas as pd
from datetime import timedelta

from src.app.dto import AnalysisConfig
from src.infra.storage.db.connection import get_connection
from src.shared.dataframe_schema import REQUIRED_SENTIMENT_COLUMNS, require_columns


def save_sentiment_df(
    sentiment_df: pd.DataFrame, db_path: Path | str | None = None, coin: str = "btc"
) -> None:
    require_columns(sentiment_df, REQUIRED_SENTIMENT_COLUMNS, "sentiment_df")
    df = sentiment_df.copy()
    df["coin"] = coin.upper()

    rows = df[
        [
            "coin",
            "source",
            "source_id",
            "analyzer",
            "sentiment",
        ]
    ].itertuples(index=False, name=None)
    with closing(get_connection(db_path)) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO sentiment (coin, source, source_id, analyzer, sentiment)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def load_sentiment_df(
    config: AnalysisConfig, analyzer: str, db_path: Path | str | None = None
) -> pd.DataFrame:
    if not config.sources:
        return pd.DataFrame()
    start_date = config.start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = config.end_date.strftime("%Y-%m-%d %H:%M:%S")
    source_placeholders = ",".join("?" for _ in config.sources)

    with closing(get_connection(db_path)) as conn:
        df = pd.read_sql_query(
            f"""
                               SELECT c.coin, c.source, c.source_id, c.timestamp, c.text, c.url, s.analyzer, s.sentiment 
                               FROM content_items AS c
                               JOIN sentiment as s
                               ON s.coin = c.coin
                               AND s.source = c.source
                               AND s.source_id = c.source_id
                               WHERE c.coin = ? AND s.analyzer = ? AND c.timestamp BETWEEN ? AND ? AND c.source IN ({source_placeholders})
                               ORDER BY c.timestamp DESC
                               LIMIT ?
                               """,
            conn,
            params=(
                config.coin.upper(),
                analyzer,
                start_date,
                end_date,
                *config.sources,
                config.num_posts,
            ),
        )
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

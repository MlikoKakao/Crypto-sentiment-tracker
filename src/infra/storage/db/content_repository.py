from src.infra.storage.db.connection import get_connection
import pandas as pd
from src.shared.helpers import normalize_timestamp_column
from src.app.dto import AnalysisConfig
from datetime import timedelta


def save_content_df(content_df: pd.DataFrame, coin: str = "btc") -> None:
    df = content_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["coin"] = coin.upper()

    rows = df[
        [
            "coin",
            "source",
            "id",
            "timestamp",
            "text",
            "url",
        ]
    ].itertuples(index=False, name=None)
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO content_items (coin, source, source_id, timestamp, text, url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    conn.close()


def load_content_df(config: AnalysisConfig, source: str) -> pd.DataFrame:
    start_date = config.start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = config.end_date.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        df = pd.read_sql_query(
            """
                               SELECT * FROM content_items 
                               WHERE coin = ? AND source = ? AND timestamp BETWEEN ? AND ?
                               """,
            conn,
            params=(config.coin.upper(), source, start_date, end_date),
        )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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

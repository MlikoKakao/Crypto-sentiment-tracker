from src.infra.storage.db.connection import get_connection
import pandas as pd
from src.shared.helpers import normalize_timestamp_column
from src.app.dto import AnalysisConfig
from datetime import timedelta

def save_news_df(news_df: pd.DataFrame, coin: str = "btc") -> None:
    df = news_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["coin"] = coin.upper()

    rows = df[["coin", "timestamp", "title", "summary", "text", "source", "url"]].itertuples(index=False, name=None)
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO news (coin, timestamp, title, summary, text, source, url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    conn.close()

def load_news_df(config: AnalysisConfig) -> pd.DataFrame:
    start_date = config.start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = config.end_date.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        df = pd.read_sql_query("""
                               SELECT * FROM news 
                               WHERE coin = ? AND timestamp BETWEEN ? AND ?
                               """,
                               conn,
                               params=(config.coin.upper(), start_date, end_date)
        )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def has_news_coverage(config: AnalysisConfig, news_df: pd.DataFrame) -> bool:
    if news_df.empty:
        return False
    
    posts_count = len(news_df)
    enough_posts = posts_count >= config.num_posts / 2

    tolerance = timedelta(days=1)
    min_time = news_df["timestamp"].min()
    max_time = news_df["timestamp"].max()
    
    start_date = pd.to_datetime(config.start_date, utc=True).tz_convert(None)
    end_date = pd.to_datetime(config.end_date, utc=True).tz_convert(None)


    starts_near = min_time <= start_date + tolerance
    ends_near = max_time >= end_date - tolerance
    
    return starts_near and ends_near and enough_posts

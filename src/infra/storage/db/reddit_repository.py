from src.infra.storage.db.connection import get_connection
import pandas as pd
from src.shared.helpers import normalize_timestamp_column
from src.app.dto import AnalysisConfig
from datetime import timedelta

def save_reddit_df(reddit_df: pd.DataFrame, coin: str = "btc") -> None:
    df = reddit_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["coin"] = coin.upper()

    rows = df[["coin", "timestamp", "text", "url", "score", "num_comments", "upvote_ratio", "id", "source", "subreddit"]].itertuples(index=False, name=None)
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO reddit (coin, timestamp, text, url, score, num_comments, upvote_ratio, id, source, subreddit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    conn.close()

def load_reddit_df(config: AnalysisConfig) -> pd.DataFrame:
    start_date = config.start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = config.end_date.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        df = pd.read_sql_query("""
                               SELECT * FROM reddit 
                               WHERE coin = ? AND timestamp BETWEEN ? AND ?
                               """,
                               conn,
                               params=(config.coin.upper(), start_date, end_date)
        )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def has_reddit_coverage(config: AnalysisConfig, reddit_df: pd.DataFrame) -> bool:
    if reddit_df.empty:
        return False
    
    posts_count = len(reddit_df)
    enough_posts = posts_count >= config.num_posts / 2

    tolerance = timedelta(days=1)
    min_time = reddit_df["timestamp"].min()
    max_time = reddit_df["timestamp"].max()

    starts_near = min_time <= config.start_date + tolerance
    ends_near = max_time >= config.end_date - tolerance

    return starts_near and ends_near and enough_posts

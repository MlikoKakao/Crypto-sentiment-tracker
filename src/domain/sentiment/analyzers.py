import pandas as pd

from src.domain.sentiment.registry import ALL_ANALYZER_NAMES, ANALYZERS
from src.utils.cache import load_cached_csv, cache_csv
from src.utils.helpers import load_csv, save_csv
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)



def add_sentiment_to_file(
    input_csv: str,
    output_csv: str,
    analyzer_name: str = "vader",
    cache_settings: Optional[Dict[str, Any]] = None,
    freshness_minutes: int = 30,
) -> None:

    if cache_settings:
        cached = load_cached_csv(cache_settings, freshness_minutes=freshness_minutes)
        if cached is not None:
            save_csv(cached, output_csv)
            logger.info(f"Loaded cached sentiment to {output_csv}")
            return

    df = load_csv(input_csv)

    name = analyzer_name.lower()
    analyzer_func = ANALYZERS.get(name)

    if name == "all":
        for analyzer in ALL_ANALYZER_NAMES:
            func = ANALYZERS[analyzer]
            col_name = f"sentiment_{analyzer}"
            df[col_name] = df["text"].apply(func)

        sentiment_cols = [f"sentiment_{analyzer}" for analyzer in ALL_ANALYZER_NAMES]
        df["sentiment"] = df[sentiment_cols].mean(axis=1)

    else:
        analyzer_func = ANALYZERS.get(name)
        if analyzer_func is None:
            raise ValueError(f"Unknown analyzer: {analyzer_name}")
        
        df["sentiment"] = df["text"].apply(analyzer_func)

    save_csv(df, output_csv)
    if cache_settings:
        cache_csv(df, cache_settings)
    logger.info(
        f"Sentiment added using {analyzer_name}. Saved to {output_csv}. Total records: {len(df)}"
    )
    print("Sentiment added. Preview:")
    print(df.head())

def load_sentiment_df(
    news_path: Optional[str], reddit_path: Optional[str], twitter_path: Optional[str], posts_choice: str
) -> pd.DataFrame:
    if posts_choice == "News":
        if not news_path:
            return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
        return pd.read_csv(news_path, parse_dates=["timestamp"]).copy()
    if posts_choice == "Reddit":
        if not reddit_path:
            return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
        return pd.read_csv(reddit_path, parse_dates=["timestamp"]).copy()
    if posts_choice in ("Twitter", "Twitter/X"):
        if not twitter_path:
            return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
        return pd.read_csv(twitter_path, parse_dates=["timestamp"]).copy()

    n = pd.read_csv(news_path, parse_dates=["timestamp"]) if news_path else None
    r = pd.read_csv(reddit_path, parse_dates=["timestamp"]) if reddit_path else None
    t = pd.read_csv(twitter_path, parse_dates=["timestamp"]) if twitter_path else None

    frames = [df for df in (n, r, t) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
    
    all_cols = sorted(set().union(*[f.columns for f in frames]))
    frames = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    add_sentiment_to_file("data/bitcoin_posts.csv","data/bitcoin_posts_with_sentiment.csv")
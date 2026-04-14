    

import pandas as pd

from src.app.dto import AnalysisConfig
from src.infra.fetchers.news import fetch_news_posts
from src.infra.fetchers.reddit import fetch_reddit_posts
from src.infra.fetchers.youtube import fetch_youtube_posts


FETCHERS = {
    "reddit": fetch_reddit_posts,
    "news": fetch_news_posts,
    "youtube": fetch_youtube_posts,
}

def fetch_posts(config: AnalysisConfig) -> pd.DataFrame:
    frames = []
    for source in config.sources:
        fetcher = FETCHERS.get(source)
        if fetcher:
            df = fetcher(config)

            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])

    all_cols = sorted(set().union(*(frame.columns for frame in frames)))
    frames = [frame.reindex(columns=all_cols) for frame in frames]

    return pd.concat(frames, ignore_index=True)

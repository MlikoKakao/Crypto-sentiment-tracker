import pandas as pd


from src.app.dto import AnalysisConfig


def fetch_posts(config: AnalysisConfig) -> pd.DataFrame:
    fetchers = get_fetchers()

    frames = []
    for source in config.sources:
        try:
            fetcher = fetchers.get(source)
            if fetcher:
                df = fetcher(config)

                if not df.empty:
                    frames.append(df)
        except Exception as e:
            print(f"Source {source} failed: {e}")
            continue
    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "text", "sentiment", "source", "source_id", "url"]
        )

    all_cols = sorted(set().union(*(frame.columns for frame in frames)))
    frames = [frame.reindex(columns=all_cols) for frame in frames]

    return pd.concat(frames, ignore_index=True)

def get_fetchers():
    from src.infra.fetchers.news import fetch_news_posts
    from src.infra.fetchers.reddit import fetch_reddit_posts
    from src.infra.fetchers.youtube import fetch_youtube_posts

    return {
        "reddit": fetch_reddit_posts,
        "news": fetch_news_posts,
        "youtube": fetch_youtube_posts,
    }
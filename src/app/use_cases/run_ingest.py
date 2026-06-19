import pandas as pd
from dataclasses import dataclass

from src.app.dto import AnalysisConfig
from src.app.dto import Source


@dataclass
class IngestResult:
    status: str
    coin: str
    sources: list[Source]
    price_df: pd.DataFrame
    posts_df: pd.DataFrame
    sentiment_df: pd.DataFrame


def run_ingest(config: AnalysisConfig) -> IngestResult:
    price_df = fetch_and_save_price(config)
    posts_df = fetch_and_save_posts(config)
    sentiment_df = analyze_and_save_sentiment(config, posts_df)
    response = IngestResult(
        status="ok",
        coin=config.coin,
        sources=list(config.sources),
        price_df=price_df,
        posts_df=posts_df,
        sentiment_df=sentiment_df,
    )
    return response


def fetch_and_save_price(config: AnalysisConfig) -> pd.DataFrame:
    from src.infra.fetchers.coinbase_price import get_coinbase_price_history
    from src.infra.storage.db.price_repository import save_price_df

    price_df = get_coinbase_price_history(config)
    save_price_df(price_df, config.coin)
    return price_df


def fetch_and_save_posts(config: AnalysisConfig) -> pd.DataFrame:
    from src.infra.fetchers.service import fetch_posts
    from src.infra.storage.db.content_repository import save_content_df

    posts_df = fetch_posts(config)
    save_content_df(posts_df, config.coin)
    return posts_df


def analyze_and_save_sentiment(
    config: AnalysisConfig, posts_df: pd.DataFrame
) -> pd.DataFrame:
    from src.domain.sentiment.service import add_sentiment_to_df
    from src.infra.storage.db.sentiment_repository import save_sentiment_df

    if config.analyzer == "all":
        from src.domain.sentiment.registry import ALL_ANALYZER_NAMES

        frames = []

        for analyzer_name in ALL_ANALYZER_NAMES:
            one_df = add_sentiment_to_df(posts_df, analyzer_name)
            save_sentiment_df(one_df, config.coin)
            frames.append(one_df)

        sentiment_df = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
    else:
        sentiment_df = add_sentiment_to_df(posts_df, config.analyzer)
        save_sentiment_df(sentiment_df, config.coin)
    return sentiment_df

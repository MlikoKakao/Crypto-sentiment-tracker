from src.app.dto import AnalysisConfig, Source
from datetime import datetime, timedelta, timezone


DEFAULT_SUBREDDITS = ("CryptoCurrency", "CryptoCurrencyTrading", "CryptoMarkets")

DEMO_CONFIG = AnalysisConfig(
    coin="BTC",
    start_date=datetime(2025, 8, 14),
    end_date=datetime(2025, 8, 25),
    analyzer=("vader",),
    sources=("reddit",),
    num_posts=100,
    subreddits=DEFAULT_SUBREDDITS,
)


DEFAULT_CONFIG = AnalysisConfig(
    coin="BTC",
    start_date=datetime.now(timezone.utc) - timedelta(days=7),
    end_date=datetime.now(timezone.utc),
    analyzer=("vader",),
    sources=("reddit",),
    num_posts=100,
    subreddits=DEFAULT_SUBREDDITS,
)

POSTS_CHOICE_TO_SOURCES: dict[str, tuple[Source, ...]] = {
    "All": ("reddit", "twitter", "news"),
    "Reddit": ("reddit",),
    "Twitter/X": ("twitter",),
    "News": ("news",),
}

from src.app.dto import AnalysisConfig
from datetime import datetime, timedelta, timezone


DEFAULT_SUBREDDITS = ("CryptoCurrency", "CryptoCurrencyTrading", "CryptoMarkets")

def default_config() -> AnalysisConfig:
    now = datetime.now(timezone.utc)
    return AnalysisConfig(
        coin="BTC",
        start_date=now - timedelta(days=7),
        end_date=now,
        analyzer="vader",
        sources=("reddit",),
        num_posts=100,
        subreddits=DEFAULT_SUBREDDITS,
    )

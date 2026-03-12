from app.dto import AnalysisConfig
from datetime import datetime, timedelta


DEFAULT_SUBREDDITS = ("CryptoCurrency", "CryptoCurrencyTrading", "CryptoMarkets")

DEMO_CONFIG = AnalysisConfig(
    coin="BTC",
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    analyzer=("vader",),
    sources=("reddit",),
    num_posts=100,
    subreddits=DEFAULT_SUBREDDITS,
)

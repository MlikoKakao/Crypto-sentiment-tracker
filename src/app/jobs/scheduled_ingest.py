from datetime import datetime, timedelta, timezone

from src.app.dto import AnalysisConfig, Analyzer, Source, Coin
from src.app.use_cases.get_indicators import add_indicators_with_cache
from src.app.use_cases.run_ingest import run_ingest
from src.domain.market.dto import IndicatorConfig


def run_scheduled_ingest() -> None:
    failed_coins = []
    summary_posts = 0
    summary_sentiment = 0
    summary_signals = 0
    coins: tuple[Coin, ...] = ("BTC", "ETH", "XMR")
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(hours=3)
    end_date = now

    analyzer: Analyzer = "all"
    sources: tuple[Source, ...] = ("reddit", "news", "youtube")
    num_posts = 1000
    subreddits_by_coin = {
    "BTC": ("CryptoCurrency", "CryptoMarkets", "Bitcoin", "btc", "BitcoinMarkets"),
    "ETH": ("CryptoCurrency", "CryptoMarkets", "ethereum", "ethtrader", "eth"),
    "XMR": ("CryptoCurrency", "CryptoMarkets", "monero", "xmrtrader"),
}

    for coin in coins:
        try:
            config, signals_config = build_configs(
                coin, start_date, end_date, analyzer, sources, num_posts, subreddits_by_coin[coin], True
            )

            result = run_ingest(config)
            indicators = add_indicators_with_cache(result.price_df, signals_config)

            print(
                f"{datetime.now(timezone.utc)}"
                f"Ingested {len(result.posts_df)} posts, for coin {coin}, "
                f"{len(result.sentiment_df)} sentiment rows. "
                f"Signal length: {len(indicators)} posts, for coin {coin}, "
            )
            summary_posts += len(result.posts_df)
            summary_sentiment += len(result.sentiment_df)
            summary_signals += len(indicators)
        except Exception as e:
            print(f"Failed for {coin}: {e}")
            failed_coins.append(coin)
            continue
    print(f"Scheduled ingest finished. Failed coins: {failed_coins}")
    print(f"{summary_posts} posts ingested.")
    print(f"{summary_sentiment} sentiment scored.")
    print(f"{summary_signals} signals added.")


def build_configs(
    coin: Coin,
    start_date: datetime,
    end_date: datetime,
    analyzer: Analyzer,
    sources: tuple[Source, ...],
    num_posts: int,
    subreddits: tuple[str, ...],
    use_sma: bool = True,
    use_rsi: bool = True,
    use_macd: bool = True,
    force_refresh: bool = False
) -> tuple[AnalysisConfig, IndicatorConfig]:
    configs = (
        (
            build_ingest_config(
                coin, start_date, end_date, analyzer, sources, num_posts, subreddits, force_refresh
            )
        ),
        (build_signal_config(coin, start_date, end_date, use_sma, use_rsi, use_macd)),
    )

    return configs


def build_ingest_config(
    coin: Coin,
    start_date: datetime,
    end_date: datetime,
    analyzer: Analyzer,
    sources: tuple[Source, ...],
    num_posts: int,
    subreddits: tuple[str, ...],
    force_refresh: bool = False,
) -> AnalysisConfig:
    return AnalysisConfig(
        coin=coin,
        start_date=start_date,
        end_date=end_date,
        analyzer=analyzer,
        sources=sources,
        num_posts=num_posts,
        subreddits=subreddits,
        force_refresh=force_refresh,
    )


def build_signal_config(
    coin: Coin,
    start_date: datetime,
    end_date: datetime,
    use_sma: bool = True,
    use_rsi: bool = True,
    use_macd: bool = True,
) -> IndicatorConfig:
    return IndicatorConfig(
        coin=coin,
        start_date=start_date,
        end_date=end_date,
        use_sma=use_sma,
        use_rsi=use_rsi,
        use_macd=use_macd,
    )


if __name__ == "__main__":
    run_scheduled_ingest()

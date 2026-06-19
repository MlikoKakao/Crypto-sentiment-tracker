from datetime import datetime, timedelta

from src.app.dto import AnalysisConfig, Analyzer, Source, Coin
from src.app.use_cases.get_indicators import add_indicators_with_cache
from src.app.use_cases.run_ingest import run_ingest
from src.domain.market.dto import IndicatorConfig


def run_scheduled_ingest() -> None:
    coins: tuple[Coin, ...] = ("BTC", "ETH", "XMR")
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    analyzer: Analyzer = "vader"
    sources: tuple[Source, ...] = ("reddit", "news", "youtube")
    num_posts = 1000
    subreddits: tuple[str, ...] = ("CryptoCurrency", "CryptocurrencyTrading", "CryptoMarkets")

    for coin in coins:
        config, signals_config = build_configs(
            coin, start_date, end_date, analyzer, sources, num_posts, subreddits
        )

        result = run_ingest(config)
        indicators = add_indicators_with_cache(result.price_df, signals_config)

        print(
            f"Ingested {len(result.posts_df)} posts, for coin {coin}, "
            f"{len(result.sentiment_df)} sentiment rows. "
            f"Signal length: {len(indicators)} posts, for coin {coin}, "
        )


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
) -> tuple[AnalysisConfig, IndicatorConfig]:
    configs = (
        (
            build_ingest_config(
                coin, start_date, end_date, analyzer, sources, num_posts, subreddits
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
) -> AnalysisConfig:
    return AnalysisConfig(
        coin=coin,
        start_date=start_date,
        end_date=end_date,
        analyzer=analyzer,
        sources=sources,
        num_posts=num_posts,
        subreddits=subreddits,
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
